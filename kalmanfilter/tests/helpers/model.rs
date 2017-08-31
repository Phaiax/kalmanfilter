
use na::DVector;
use rand::distributions::{Normal, IndependentSample};
use rand::thread_rng;

use kalmanfilter::systems::continuous_to_discrete;

use super::types::*;


pub struct ContinuousLinearModelBuilder {
    pub vec_x_init : SystemState,
    pub mat_a : SystemMatrix,
    pub mat_b : InputMatrix,
    pub vec_w : SystemNoise,
    pub mat_c : MeasurementMatrix,
    pub vec_r : MeasurementNoise,
}

pub struct ContinuousLinearModel {
    vec_x : SystemState,
    mat_a : SystemMatrix,
    mat_b : InputMatrix,
    vec_w : SystemNoise, // will also be used to hold the drawn samples
    mat_c : MeasurementMatrix,
    vec_r : MeasurementNoise, // will also be used to hold the drawn samples
    num_inputs : usize,
    num_states : usize,
    system_noise_gen : Vec<Normal>,
    measurement_noise_gen : Vec<Normal>,
}

impl From<ContinuousLinearModelBuilder> for ContinuousLinearModel {
    fn from(builder : ContinuousLinearModelBuilder) -> ContinuousLinearModel {
        let num_states = builder.mat_a.nrows();
        let num_measurements = builder.mat_c.nrows();
        let num_inputs = builder.mat_b.ncols();
        assert_eq!(num_states, builder.mat_a.ncols());
        assert_eq!(num_states, builder.mat_b.nrows());
        assert_eq!(num_states, builder.vec_w.nrows());
        assert_eq!(num_states, builder.mat_c.ncols());
        assert_eq!(num_measurements, builder.vec_r.nrows());
        let system_noise_gen = mk_noise_generators(&builder.vec_w);
        let measurement_noise_gen = mk_noise_generators(&builder.vec_r);
        ContinuousLinearModel {
            vec_x : builder.vec_x_init,
            mat_a : builder.mat_a,
            mat_b : builder.mat_b,
            vec_w : builder.vec_w,
            mat_c : builder.mat_c,
            vec_r : builder.vec_r,
            num_inputs : num_inputs,
            num_states : num_states,
            system_noise_gen : system_noise_gen,
            measurement_noise_gen : measurement_noise_gen,
        }
    }
}

impl ContinuousLinearModel {

    pub fn step(&mut self, u : &InputVector, dt : f64) -> Measurements {
        assert_eq!(self.num_inputs, u.nrows());

        for (gen, n) in self.system_noise_gen.iter().zip(self.vec_w.iter_mut()) {
            *n = gen.ind_sample(&mut thread_rng());
        }

        for (gen, n) in self.measurement_noise_gen.iter().zip(self.vec_r.iter_mut()) {
            *n = gen.ind_sample(&mut thread_rng());
        }

        self.vec_x += ( &self.mat_a * &self.vec_x + &self.mat_b * u + &self.vec_w ) * dt;

        &self.mat_c * &self.vec_x + &self.vec_r
    }

    pub fn get_a(&self) -> &SystemMatrix {
        &self.mat_a
    }

    pub fn get_b(&self) -> &InputMatrix {
        &self.mat_b
    }

    pub fn get_state(&self) -> &SystemState {
        &self.vec_x
    }

    pub fn get_c(&self) -> &MeasurementMatrix {
        &self.mat_c
    }

    pub fn get_num_states(&self) -> usize {
        self.num_states
    }

    pub fn get_num_inputs(&self) -> usize {
        self.num_inputs
    }
}



pub struct DiscreteLinearModelBuilder {
    pub vec_x_init : SystemState,
    pub mat_f : SystemMatrix,
    pub mat_h : InputMatrix,
    pub vec_w : SystemNoise,
    pub mat_c : MeasurementMatrix,
    pub vec_r : MeasurementNoise,
}

impl ContinuousLinearModelBuilder {
    pub fn into_discrete(self, dt : TimeStep, eps : f64) -> DiscreteLinearModelBuilder {
        let disc_sys = continuous_to_discrete(&self.mat_a, &self.mat_b, dt, eps);
        DiscreteLinearModelBuilder {
            vec_x_init : self.vec_x_init,
            mat_f : disc_sys.mat_f,
            mat_h : disc_sys.mat_h,
            vec_w : self.vec_w,
            mat_c : self.mat_c,
            vec_r : self.vec_r,
        }
    }
}

pub struct DiscreteLinearModel {
    vec_x : SystemState,
    mat_f : SystemMatrix,
    mat_h : InputMatrix,
    vec_w : SystemNoise, // will also be used to hold the drawn samples
    mat_c : MeasurementMatrix,
    vec_r : MeasurementNoise, // will also be used to hold the drawn samples
    num_inputs : usize,
    num_states : usize,
    system_noise_gen : Vec<Normal>,
    measurement_noise_gen : Vec<Normal>,
}

fn mk_noise_generators(variances : &DVector<f64>) -> Vec<Normal> {
    let mut gen = vec![];
    for w in variances.iter() {
        gen.push(Normal::new(0.0, *w));
    }
    gen
}

impl From<DiscreteLinearModelBuilder> for DiscreteLinearModel {
    fn from(builder : DiscreteLinearModelBuilder) -> DiscreteLinearModel {
        let num_states = builder.mat_f.nrows();
        let num_measurements = builder.mat_c.nrows();
        let num_inputs = builder.mat_h.ncols();
        assert_eq!(num_states, builder.mat_f.ncols());
        assert_eq!(num_states, builder.mat_h.nrows());
        assert_eq!(num_states, builder.vec_w.nrows());
        assert_eq!(num_states, builder.mat_c.ncols());
        assert_eq!(num_measurements, builder.vec_r.nrows());
        let system_noise_gen = mk_noise_generators(&builder.vec_w);
        let measurement_noise_gen = mk_noise_generators(&builder.vec_r);
        DiscreteLinearModel {
            vec_x : builder.vec_x_init,
            mat_f : builder.mat_f,
            mat_h : builder.mat_h,
            vec_w : builder.vec_w,
            mat_c : builder.mat_c,
            vec_r : builder.vec_r,
            num_inputs : num_inputs,
            num_states : num_states,
            system_noise_gen : system_noise_gen,
            measurement_noise_gen : measurement_noise_gen,
        }
    }
}

impl DiscreteLinearModel {

    pub fn step(&mut self, u : &InputVector) -> Measurements {
        assert_eq!(self.num_inputs, u.nrows());

        for (gen, n) in self.system_noise_gen.iter().zip(self.vec_w.iter_mut()) {
            *n = gen.ind_sample(&mut thread_rng());
        }

        for (gen, n) in self.measurement_noise_gen.iter().zip(self.vec_r.iter_mut()) {
            *n = gen.ind_sample(&mut thread_rng());
        }

        self.vec_x = &self.mat_f * &self.vec_x + &self.mat_h * u + &self.vec_w;

        &self.mat_c * &self.vec_x + &self.vec_r
    }

    pub fn get_f(&self) -> &SystemMatrix {
        &self.mat_f
    }

    pub fn get_h(&self) -> &InputMatrix {
        &self.mat_h
    }

    pub fn get_state(&self) -> &SystemState {
        &self.vec_x
    }

    pub fn get_c(&self) -> &MeasurementMatrix {
        &self.mat_c
    }

    pub fn get_num_states(&self) -> usize {
        self.num_states
    }

    pub fn get_num_inputs(&self) -> usize {
        self.num_inputs
    }
}


pub fn example_model_2states_regular_stable() -> ContinuousLinearModelBuilder {
    ContinuousLinearModelBuilder {
        vec_x_init : SystemState::from_row_slice(2, &[
                0.,
                0.,
            ]),
        // has eigenvalues -3.5 and -1.5
        mat_a : SystemMatrix::from_row_slice(2, 2, &[
                -3., 1.5,
                0.5, -2.,
            ]),
        mat_b : InputMatrix::from_row_slice(2, 1, &[
                1.,
                0.,
            ]),
        vec_w : SystemNoise::from_row_slice(2, &[0., 0.,]),
        mat_c : MeasurementMatrix::from_row_slice(1, 2, &[0., 2.,]),
        vec_r : MeasurementNoise::from_row_slice(1, &[0.]),
        // mat_c : MeasurementMatrix::from_row_slice(2, 2, &[0., 2., 1., 0.]),
        // vec_r : MeasurementNoise::from_row_slice(2, &[0., 0.]),
    }
}

pub fn example_model_2states_singular_stable() -> ContinuousLinearModelBuilder {
    ContinuousLinearModelBuilder {
        vec_x_init : SystemState::from_row_slice(2, &[
                0.,
                0.,
            ]),
        // has eigenvalues -3.5 and -1.5
        mat_a : SystemMatrix::from_row_slice(2, 2, &[
                -3., 1.5,
                1.5, -0.75,
            ]),
        mat_b : InputMatrix::from_row_slice(2, 1, &[
                1.,
                0.,
            ]),
        vec_w : SystemNoise::from_row_slice(2, &[0., 0.,]),
        mat_c : MeasurementMatrix::from_row_slice(1, 2, &[0., 2.,]),
        vec_r : MeasurementNoise::from_row_slice(1, &[0.]),
        // mat_c : MeasurementMatrix::from_row_slice(2, 2, &[0., 2., 1., 0.]),
        // vec_r : MeasurementNoise::from_row_slice(2, &[0., 0.]),
    }
}