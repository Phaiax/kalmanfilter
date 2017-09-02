#![allow(dead_code)]

extern crate kalmanfilter;
extern crate nalgebra as na;
extern crate num;
extern crate rand;

mod helpers;

use kalmanfilter::nt;
use na::{Real, DVector};
use helpers::model::*;

#[test]
fn compare_continuous_discrete() {
    let dt = 0.001;
    let eps = 1e-12;
    let sim_time = 20usize;
    let steps = (sim_time as f64 / dt) as usize;

    let mut cont : ContinuousLinearModel = example_model_2states_regular_stable().into();
    let mut discr : DiscreteLinearModel = example_model_2states_regular_stable()
                                            .into_discrete(dt, eps).into();

    for i in 0..steps {
        let t = i as f64 * dt;
        let u = if t < 5. { 0. } else { 1. };
        let u = nt::InputVector(DVector::from_row_slice(1, &[u,]));

        cont.step(&u, dt);
        discr.step(&u);

        let diff = &discr.get_state().0 - &cont.get_state().0;
        let max = diff.iter().fold(0., |store, item| { store.max(*item) });
        if max > 0.0001 {
            panic!("Continuous does not equal discrete form");
        }
    }
}

#[test]
fn compare_continuous_discrete_async() {
    let dt_discr = 0.1;
    let dt_cont = 0.001;
    let eps = 1e-12;
    let sim_time = 20usize;
    let steps = (sim_time as f64 / dt_cont) as usize;
    let discr_every = (dt_discr / dt_cont) as usize;

    let mut cont : ContinuousLinearModel = example_model_2states_regular_stable().into();
    let mut discr : DiscreteLinearModel = example_model_2states_regular_stable()
                                            .into_discrete(dt_discr, eps).into();

    for i in 0..steps {
        let t = i as f64 * dt_cont;
        // Note the <=: Make sure the continuous model does not get the step earlier
        let u = if t <= 5. { 0. } else { 1. };
        let u = nt::InputVector(DVector::from_row_slice(1, &[u,]));

        cont.step(&u, dt_cont);

        if i % discr_every == 0 && i != 0 {
            discr.step(&u);

            let diff = &discr.get_state().0 - &cont.get_state().0;
            let max = diff.iter().fold(0., |store, item| { store.max(*item) });
            if max > 0.0001 {
                panic!("Continuous does not equal discrete form");
            }

            // if (t - t.trunc()) < dt_cont/2. {
            //     //println!("t={}, discr={}, cont={}", t, discr.get_state(), cont.get_state());
            //     println!("t={}, diff={}", t, diff);
            // }
        }


    }
}


/// Tests `kalmanfilter::systems::calc_mat_h`
#[test]
fn compare_continuous_discrete_async_singular() {
    let dt_discr = 0.1;
    let dt_cont = 0.001;
    let eps = 1e-5;
    let sim_time = 20usize;
    let steps = (sim_time as f64 / dt_cont) as usize;
    let discr_every = (dt_discr / dt_cont) as usize;

    let mut cont : ContinuousLinearModel = example_model_2states_singular_stable().into();
    let mut discr : DiscreteLinearModel = example_model_2states_singular_stable()
                                            .into_discrete(dt_discr, eps).into();

    for i in 0..steps {
        let t = i as f64 * dt_cont;
        // Note the <=: Make sure the continuous model does not get the step earlier
        let u = if t <= 5. { 0. } else { 1. };
        let u = nt::InputVector(DVector::from_row_slice(1, &[u,]));

        cont.step(&u, dt_cont);

        if i % discr_every == 0 && i != 0 {
            discr.step(&u);

            let diff = &discr.get_state().0 - &cont.get_state().0;
            let max = diff.iter().fold(0., |store, item| { store.max(*item) });
            if max > 0.0001 {
                panic!("Continuous does not equal discrete form");
            }

            // if (t - t.trunc()) < dt_cont/2. {
            //     //println!("t={}, discr={}, cont={}", t, discr.get_state(), cont.get_state());
            //     println!("t={}, diff={}", t, diff);
            // }
        }


    }
}
