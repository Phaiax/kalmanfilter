
use std::convert::From;

use alga::general::Real;
use na::{DMatrix, DVector};
use std::ops::Mul;

use nt::{DiscreteSystemMatrix, DiscreteInputMatrix, SystemNoiseVarianceMatrix, StateVector,
         CovarianceMatrix, InputVector, Measurement, MeasurementMatrixRow,
         MeasurementNoiseVariance};


pub struct KalmanFilter<N : Real>
{
    num_states : usize,
    num_inputs : usize,
    mat_f : DiscreteSystemMatrix<N>,
    mat_h : DiscreteInputMatrix<N>,
    mat_q : SystemNoiseVarianceMatrix<N>,
    vec_state : StateVector<N>,
    mat_p : CovarianceMatrix<N>,
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
                mat_f : DiscreteSystemMatrix(DMatrix::identity(num_states, num_states)),
                mat_h : DiscreteInputMatrix(DMatrix::zeros(num_states, num_inputs)),
                mat_q : SystemNoiseVarianceMatrix(DMatrix::zeros(num_states, num_states)),
                vec_state : StateVector(DVector::zeros(num_states)),
                mat_p : CovarianceMatrix(DMatrix::identity(num_inputs, num_inputs)),
            }
        }
    }

    pub fn with_system_matrix(mut self, mat_f : DiscreteSystemMatrix<N>) -> Self {
        assert_eq!(self.filter.num_states, mat_f.ncols());
        assert_eq!(self.filter.num_states, mat_f.nrows());
        self.filter.mat_f = mat_f;
        self
    }

    pub fn with_input_matrix(mut self, mat_h : DiscreteInputMatrix<N>) -> Self {
        assert_eq!(self.filter.num_inputs, mat_h.ncols());
        assert_eq!(self.filter.num_states, mat_h.nrows());
        self.filter.mat_h = mat_h;
        self
    }

    pub fn with_system_noise_variances(mut self, mat_q : SystemNoiseVarianceMatrix<N>) -> Self {
        assert_eq!(self.filter.num_states, mat_q.ncols());
        assert_eq!(self.filter.num_states, mat_q.nrows());
        self.filter.mat_q = mat_q;
        self
    }

    pub fn with_initial_state(mut self, vec_state : StateVector<N>, mat_covariances : CovarianceMatrix<N>) -> Self {
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
    pub vec_state : &'a StateVector<N>,
    pub mat_covariances : &'a CovarianceMatrix<N>,
}

impl<N : Real> KalmanFilter<N> {

    pub fn predict<'a>(&'a mut self, u : &InputVector<N>) -> BorrowedSystemState<'a, N> {
        assert_eq!(self.num_inputs, u.0.len());
        self.vec_state = StateVector( &self.mat_f.0 * &self.vec_state.0 + &self.mat_h.0 * &u.0 );
        self.mat_p = CovarianceMatrix( &self.mat_f.0 * &self.mat_p.0 * &self.mat_f.0.transpose()
                                     + &self.mat_q.0 );
        BorrowedSystemState {
            vec_state : &self.vec_state,
            mat_covariances : &self.mat_p,
        }
    }

    pub fn measure<'a>(&'a mut self,
                       y : Measurement<N>,
                       rvec_c : MeasurementMatrixRow<N>,
                       r : MeasurementNoiseVariance<N>)
                    -> BorrowedSystemState<'a, N> {

        assert_eq!(self.num_states, rvec_c.0.len());
        // The matrix measurement formula is : vec_y = mat_c vec_x + vec_r
        // If we measure only one line of that y vector, named scalar_y, we get
        //                                  scalar_y = rvec_c * vec_x + scalar_r

        // S = C P C^T + R
        let s : N = (&rvec_c.0).mul(&self.mat_p.0).dot(&rvec_c.0) + r.0;

        // K = P C^T S^-1 // k is column vector
        let vec_k : DVector<N> = (&self.mat_p.0 * &rvec_c.0.transpose()) * s.recip();

        // residual = y - C x
        let residual = y.0 - (&rvec_c.0 * &self.vec_state.0)[(0, 0)];

        // x = x + K residual
        self.vec_state.0 += &vec_k * residual;

        // P = P - K C P
        self.mat_p.0 = &self.mat_p.0 - &vec_k * &rvec_c.0 * &self.mat_p.0;

        BorrowedSystemState {
            vec_state : &self.vec_state,
            mat_covariances : &self.mat_p,
        }
    }
}

