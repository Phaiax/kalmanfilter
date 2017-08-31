
Kalman Filter
=============

Crate status: unstable

## Goal of this crate

Provide implementations for a Kalman Filter, Extended Kalman Filter and an efficient Stochastic Cloning Kalman Filter.

## Symbols and Variable Names

Vectors are represented by lowercase letters like a, matrices are written as uppercase letters like A. Additional information may be added after an underscore and optionally grouped with `{}`, for example `v_{bullet}`.

In source code, matrices are prefixed with `mat_`, column vectors with  `vec_` and row vectors with  `rvec_`. All symbols are converted to lowercase. (Only to keep the compiler from complaining). A describing index may be
added as well, for example `vec_v_bullet` for the vector v that describes the bullet speed.

A discrete time-invariant linear system uses the following matrix and vector letters:

```math
    k       : current timestep
    x_{k}   : system state at timestep k
    F       : discrete system matrix
    H       : discrete input matrix
    u_{k}   : input vector at timestep k (interpreted stepwise constant)
    w_{k}   : system noise at timestep k
    y_{k}   : measurements at timestep k
    C       : measurement matrix
    r_{k}   : measurement noise at timestep k

    Discrete system equation:
    x_{k}  =   F x_{k-1}   +   H u_{k}  + w_{k}

    Discrete measurement equation:
    y_{k}  =   C x_{k} + r_{k}

```

```block
                   system                                   measurement
                   noise           x_init                      noise
                     │                │                          │
                     │                ▼                          │
         ┌─────┐     ▼ x_{k+1}  ┌──────────┐    x_k    ┌─────┐   ▼
u_k ────▶│  H  │────▶⊕─────────>│  delay   │─────┬────▶│  C  │───⊕────▶ y_k
         └─────┘     ▲          └──────────┘     │     └─────┘
                     │                           │
                     │             ┌─────┐       │
                     └─────────────│  F  │◀──────┘
                                   └─────┘

```

Since the kalman filter estimates the system state, the measurement equation represents the available measurements (y_k), not compulsorily some other system output that has to be controlled by the controller in a later step.

For a linear model given in continuous form, the following symbols will be used:

```math
    t       : current time
    x_{t}   : system state at time t
    A       : continuous system matrix
    B       : continuous measurement matrix
    u_{t}   : input vector at time t
    w_{t}   : system noise at time t
    y_{t}   : measurements at time t
    C       : measurement matrix (same as in discrete form)
    r_{t}   : measurement noise at time t
    d/dt(...): derivation after time

    Differential system equation:
    d/dt( x_{t} )  =  A x_{t}  +  B  u_{t}  +  v_{t}

    Measurement equation:
    y_{t}  =   C x_{t} + r_{t}

```

```block
                   system                                   measurement
                   noise                   x_init              noise
                     │                        │                  │
                     │                        │                  │
         ┌─────┐     ▼          ┌──────────┐  ▼  x     ┌─────┐   ▼
 u  ────▶│  B  │────▶⊕─────────>│Integrator│─▶⊕──┬────▶│  C  │───⊕────▶ y
         └─────┘     ▲          └──────────┘     │     └─────┘
                     │                           │
                     │             ┌─────┐       │
                     └─────────────│  A  │◀──────┘
                                   └─────┘

```

Note that A and B from the continuous form can be transformed into F and H from the discrete form. The current kalman filter implementations need the discrete model, but the model in the tests can also be fed with the continuous matrices. These forms will be integrated for calculation:

```math
    dt     : time step for approximation

    Approximated system equation:
    x_{t+dt}  =  x_{t}  +  dt * ( A x_{t}  +  B u_{t}  +  v_{t} )

```

The conversion from (A, B) to (F, H) for a certain contant timestep dt can be done as follows. See `kalmanfilter::systems::continuous_to_discrete`.

```math
    SUM_variable=start...inclusiveend
    INTEGRAL_variable=start...inclusiveend

    I    : Unit matrix (ones in diagonal)

    F = F(dt) = exp(A*dt) = SUM_i=0...infinite ( (A*dt)^i / i! )
                          = I + A*dt
                              + ( (A*dt)^2 / 2 )
                              + ( (A*dt)^3 / 6 ) 
                              + ...
    H = H(dt) = INTEGRAL_v=0...T ( F(v) )   * B
```

