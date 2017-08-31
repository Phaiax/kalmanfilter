#![allow(dead_code)]

extern crate kalmanfilter;
extern crate nalgebra as na;
extern crate num;
extern crate rand;

mod helpers;

use helpers::types::*;
use helpers::model::*;
use kalmanfilter::{KalmanFilterBuilder, KalmanFilter};

/// # Modell
///
/// x = [ 2  1 ] x  + [ 1 ] u + w
///     [ 3  0 ]      [ 0 ]
///
/// # Measurement
///
/// z = [ 0 2 ] x + r
///
pub fn new_model_1() -> DiscreteLinearModel {
    DiscreteLinearModelBuilder {
        vec_x_init : SystemState::from_row_slice(2, &[
                0.,
                0.,
            ]),
        // has eigenvalues -3.5 and -1.5
        mat_f : SystemMatrix::from_row_slice(2, 2, &[
                -1., 1.5,
                0.5, -2.,
            ]),
        mat_h : InputMatrix::from_row_slice(2, 1, &[
                1.,
                0.,
            ]),
        vec_w : SystemNoise::from_row_slice(2, &[0., 0.,]),
        mat_c : MeasurementMatrix::from_row_slice(1, 2, &[0., 2.,]),
        vec_r : MeasurementNoise::from_row_slice(1, &[0.]),
        // h : MeasurementMatrix::from_row_slice(2, 2, &[0., 2., 1., 0.]),
        // r : MeasurementNoise::from_row_slice(2, &[0., 0.]),

    }.into()
}


#[test]
fn simple_linear_model() {
    let dt : TimeStep = 0.01;
    let mut rw : DiscreteLinearModel = example_model_2states_regular_stable().into_discrete(dt, 1e-5).into();
    let sim_time : usize = 2;
    let steps = (sim_time as f64 / dt) as usize;

    //println!("Is observable: {:?}",
    //    helpers::analysis::kalman_observability_index(rw.get_f(), rw.get_c(), 0.1));

    let mut kf : KalmanFilter<f64> = KalmanFilterBuilder
        ::with_numstates_and_numinputs(rw.get_num_states(), rw.get_num_inputs())
        .with_f(rw.get_f().clone())
        .with_h(rw.get_h().clone())
        .with_q(SystemMatrix::from_row_slice(2, 2, &[0.01, 0., 0., 0.01]))
        .with_initial_state(SystemState::from_row_slice(2, &[0., 0.]),
                            SystemMatrix::from_row_slice(2, 2, &[100., 0., 0., 100.]))
        .into();

    for i in 0..steps {
        let t = i as f64 * dt;
        let u = if t <= 1. { 0. } else { 1. };
        let u = InputVector::from_row_slice(1, &[u,]);
        let y = rw.step(&u);
        kf.predict(&u);
        let pred = kf.measure(y[(0, 0)], rw.get_c().row(0).clone_owned(), 0.1);
        let diff = pred.vec_state - rw.get_state();

        if helpers::max(&diff) > 0.01 {
            println!("This KalmanFilter is not good.");
        }
        // if (t*10. - (t*10.).trunc()) < dt/2. {
        //     println!("--------- t={}, diff={}, covar={}", t, diff, pred.mat_covariances);
        // }

    }

}