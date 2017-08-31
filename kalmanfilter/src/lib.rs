#![allow(unused_variables)]

extern crate alga;
extern crate nalgebra as na;
extern crate num;
extern crate generic_array;

pub mod systems;

use std::convert::From;

use alga::general::Real;
use na::{DMatrix, DVector, RowDVector};
use std::ops::Mul;


pub struct KalmanFilter<N : Real>
{
    num_states : usize,
    num_inputs : usize,
    /// System matrix
    mat_f : DMatrix<N>,
    // input matrix
    mat_h : DMatrix<N>,
    // state noise matrix
    mat_q : DMatrix<N>,
    // current state
    vec_state : DVector<N>,
    // covariance matrix
    mat_p : DMatrix<N>,
}

pub struct KalmanFilterBuilder<N : Real>
{
    filter : KalmanFilter<N>,
}

impl<N : Real> KalmanFilterBuilder<N> {
    pub fn with_numstates_and_numinputs(num_states : usize, num_inputs : usize) -> KalmanFilterBuilder<N> {
        KalmanFilterBuilder {
            filter : KalmanFilter {
                num_states : num_states,
                num_inputs : num_inputs,
                mat_f : DMatrix::identity(num_states, num_states),
                mat_h : DMatrix::zeros(num_states, num_inputs),
                mat_q : DMatrix::zeros(num_states, num_states),
                vec_state : DVector::zeros(num_states),
                mat_p : DMatrix::identity(num_inputs, num_inputs),
            }
        }
    }

    pub fn with_f(mut self, mat_f : DMatrix<N>) -> Self {
        assert_eq!(self.filter.num_states, mat_f.ncols());
        assert_eq!(self.filter.num_states, mat_f.nrows());
        self.filter.mat_f = mat_f;
        self
    }

    pub fn with_h(mut self, mat_h : DMatrix<N>) -> Self {
        assert_eq!(self.filter.num_inputs, mat_h.ncols());
        assert_eq!(self.filter.num_states, mat_h.nrows());
        self.filter.mat_h = mat_h;
        self
    }

    pub fn with_q(mut self, mat_q : DMatrix<N>) -> Self {
        assert_eq!(self.filter.num_states, mat_q.ncols());
        assert_eq!(self.filter.num_states, mat_q.nrows());
        self.filter.mat_q = mat_q;
        self
    }

    pub fn with_initial_state(mut self, vec_state : DVector<N>, mat_covariances : DMatrix<N>) -> Self {
        assert_eq!(self.filter.num_states, vec_state.len());
        assert_eq!(self.filter.num_states, mat_covariances.ncols());
        assert_eq!(self.filter.num_states, mat_covariances.nrows());
        self.filter.vec_state = vec_state;
        self.filter.mat_p = mat_covariances;
        self
    }
}

impl<N : Real> From<KalmanFilterBuilder<N>> for KalmanFilter<N> {
    fn from(builder : KalmanFilterBuilder<N>) -> KalmanFilter<N> {
        builder.filter
    }
}

pub struct BorrowedSystemState<'a, N : Real + 'a> {
    pub vec_state : &'a DVector<N>,
    pub mat_covariances : &'a DMatrix<N>,
}

impl<N : Real> KalmanFilter<N> {

    pub fn predict<'a>(&'a mut self, u : &DVector<N>) -> BorrowedSystemState<'a, N> {
        assert_eq!(self.num_inputs, u.len());
        self.vec_state = &self.mat_f * &self.vec_state + &self.mat_h * u;
        self.mat_p = &self.mat_f * &self.mat_p * &self.mat_f.transpose() + &self.mat_q;
        BorrowedSystemState {
            vec_state : &self.vec_state,
            mat_covariances : &self.mat_p,
        }
    }

    pub fn measure<'a>(&'a mut self, y : N, rvec_c : RowDVector<N>, r : N) -> BorrowedSystemState<'a, N> {
        assert_eq!(self.num_states, rvec_c.len());
        // The matrix measurement formula is : vec_y = mat_c vec_x + vec_r
        // If we measure only one line of that Z vector, named z, we get
        //                                  scalar_y = rvec_c * vec_x + scalar_r

        // S = C P C^T + R
        let s : N = (&rvec_c).mul(&self.mat_p).dot(&rvec_c) + r;
        // K = P C^T S^-1 // k is column vector
        let vec_k : DVector<N> = (&self.mat_p * &rvec_c.transpose()) * s.recip();
        // residual = y - C x
        let residual = y - (&rvec_c * &self.vec_state)[(0, 0)];
        // x = x + K residual
        self.vec_state += &vec_k * residual;
        // P = P - K C P
        self.mat_p = &self.mat_p - &vec_k * &rvec_c * &self.mat_p;

        // S = C P C^T + R
        // K = P C^T S^-1
        // P = P - K C P

        BorrowedSystemState {
            vec_state : &self.vec_state,
            mat_covariances : &self.mat_p,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
