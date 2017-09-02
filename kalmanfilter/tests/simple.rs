#![allow(dead_code)]

extern crate kalmanfilter;
extern crate nalgebra as na;
extern crate num;
extern crate rand;

mod helpers;

use helpers::types::*;
use helpers::model::*;
use kalmanfilter::kf::{KalmanFilterBuilder, KalmanFilter};
use kalmanfilter::nt;

use na::{DMatrix, DVector};



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
        .with_system_matrix(rw.get_system_matrix().clone())
        .with_input_matrix(rw.get_input_matrix().clone())
        .with_system_noise_variances(nt::SystemNoiseVarianceMatrix(
            DMatrix::from_row_slice(2, 2, &[0.01, 0., 0., 0.01])))
        .with_initial_state(nt::StateVector(DVector::from_row_slice(2, &[0., 0.])),
                            nt::CovarianceMatrix(
                                DMatrix::from_row_slice(2, 2, &[100., 0., 0., 100.])))
        .into();

    for i in 0..steps {
        let t = i as f64 * dt;
        let u = if t <= 1. { 0. } else { 1. };
        let u = nt::InputVector(DVector::from_row_slice(1, &[u,]));
        let y = rw.step(&u);
        kf.predict(&u);
        let pred = kf.measure(nt::Measurement(y.0[(0, 0)]),
            nt::MeasurementMatrixRow( rw.get_measurement_matrix().0.row(0).clone_owned() ),
            nt::MeasurementNoiseVariance( 0.1 ));
        let diff = &pred.vec_state.0 - &rw.get_state().0;

        if helpers::max(&diff) > 0.01 {
            println!("This KalmanFilter is not good.");
        }
        // if (t*10. - (t*10.).trunc()) < dt/2. {
        //     println!("--------- t={}, diff={}, covar={}", t, diff, pred.mat_covariances);
        // }

    }

}