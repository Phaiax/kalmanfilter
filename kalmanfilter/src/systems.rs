
use na::{DMatrix, Real};

use nt::{DiscreteSystemMatrix, ContinuousSystemMatrix, DiscreteInputMatrix, ContinuousInputMatrix};

pub struct DiscreteSystemEqMatrices<N : Real> {
    pub mat_f : DiscreteSystemMatrix<N>,
    pub mat_h : DiscreteInputMatrix<N>,
}

/// ```math
/// F = F(dt) = exp(A*dt) = SUM_i=0...infinite ( (A*dt)^i / i! )
///                       = I + A*dt
///                           + ( (A*dt)^2 / 2 )
///                           + ( (A*dt)^3 / 6 )
///                           + ...
/// H = H(dt) = INTEGRAL_v=0...T ( F(v) )   * B
/// ```
pub fn continuous_to_discrete<N : Real>(mat_a : &ContinuousSystemMatrix<N>,
    mat_b: &ContinuousInputMatrix<N>, dt : N, eps: N)
    -> DiscreteSystemEqMatrices<N> {

    assert_eq!(mat_a.0.nrows(), mat_a.0.ncols());

    let mat_f = mat_a_to_mat_f(&mat_a, dt, eps);
    let mat_h;

    let mut inverse_mat_a = mat_a.0.clone_owned();
    if inverse_mat_a.try_inverse_mut() {
        // A is regular
        // H(dt) = A^-1 [ F(dt) - I ] B
        let mat_i = DMatrix::from_diagonal_element(mat_a.0.nrows(), mat_a.0.nrows(), N::one());
        mat_h = DiscreteInputMatrix( inverse_mat_a * ( &mat_f.0 - &mat_i ) * &mat_b.0 );
    } else {
        // A is singular
        // H(dt) = SUM_v=1...infinite ( A^(v-1) * dt^v / v! )
        mat_h = calc_mat_h(mat_a, mat_b, dt, eps);
    }

    DiscreteSystemEqMatrices {
        mat_f : mat_f,
        mat_h : mat_h,
    }
}

/// Expects a to be square
fn mat_a_to_mat_f<N : Real>(mat_a : &ContinuousSystemMatrix<N>, dt : N, eps: N) -> DiscreteSystemMatrix<N> {
    let adt = &mat_a.0 * dt;
    // tmp will be (A*dt), then (A*dt)^2, then (A*dt)^3 and so on
    let mut tmp = adt.clone();

    // i = 0
    let mut factorial : N = N::one();
    let mut mat_f = DMatrix::from_diagonal_element(mat_a.0.nrows(), mat_a.0.nrows(), N::one());

    // i = 1
    let mut factorial_cnt : N = N::one();
    mat_f += &tmp;

    for i in 2..20 {
        tmp *= &adt;

        factorial_cnt += N::one();
        factorial *= factorial_cnt;

        let tmp2 = &tmp / factorial;
        mat_f += &tmp2;

        let max = tmp2.iter().fold(N::zero(), |store, item| { store.max(*item) });
        if max <= eps {
            break;
        }
    }

    DiscreteSystemMatrix(mat_f)
}


/// H(dt) = SUM_v=1...infinite ( A^(v-1) * dt^v / v! )
///       =    I * dt / 1!
///         +  (A*dt)   * dt / 2!
///         +  (A*dt)^2 * dt / 3!
///         +  ...
fn calc_mat_h<N : Real>(mat_a : &ContinuousSystemMatrix<N>,
                        mat_b : &ContinuousInputMatrix<N>, dt : N, eps: N) -> DiscreteInputMatrix<N> {

    let adt = &mat_a.0 * dt;
    // tmp will be (I*dt), then dt*(A*dt)^2, then dt*(A*dt)^3 and so on
    let mut tmp = DMatrix::from_diagonal_element(mat_a.0.nrows(), mat_a.0.nrows(), N::one());
    tmp *= dt;

    // v = 1
    let mut factorial : N = N::one();
    let mut factorial_cnt : N = N::one();
    let mut mat_h = tmp.clone(); // I * dt

    for v in 2..20 {
        tmp *= &adt;

        factorial_cnt += N::one();
        factorial *= factorial_cnt;

        let tmp2 = &tmp / factorial;
        mat_h += &tmp2;

        let max = tmp2.iter().fold(N::zero(), |store, item| { store.max(*item) });
        if max <= eps {
            break;
        }
    }

    mat_h *= &mat_b.0;

    DiscreteInputMatrix(mat_h)
}