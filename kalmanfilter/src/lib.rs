#![allow(unused_variables)]

extern crate alga;
extern crate nalgebra as na;
extern crate num;
extern crate generic_array;

pub mod systems;
pub mod kf;

pub mod nt {
    use na::{Real, DMatrix, DVector, RowDVector};
    use std::ops::{Deref, DerefMut};

    macro_rules! newtype {
        ($newt:tt) => (
            newtype!($newt, DMatrix);
        );
        ($newt:tt, N) => (
            #[derive(Clone)]
            pub struct $newt<N : Real>(pub N);

            impl<N : Real> Deref for $newt<N> {
                type Target = N;
                fn deref(&self) -> &N { &self.0 }
            }
            impl<N : Real> DerefMut for $newt<N> {
                fn deref_mut(&mut self) -> &mut N { &mut self.0 }
            }
        );
        ($newt:tt, $mattype:tt) => (
            #[derive(Clone)]
            pub struct $newt<N : Real>(pub $mattype<N>);

            impl<N : Real> Deref for $newt<N> {
                type Target = $mattype<N>;
                fn deref(&self) -> &$mattype<N> { &self.0 }
            }
            impl<N : Real> DerefMut for $newt<N> {
                fn deref_mut(&mut self) -> &mut $mattype<N> { &mut self.0 }
            }
        );
    }

    newtype!(DiscreteSystemMatrix);
    newtype!(ContinuousSystemMatrix);

    newtype!(DiscreteInputMatrix);
    newtype!(ContinuousInputMatrix);
    newtype!(InputVector, DVector);

    newtype!(SystemNoiseVarianceMatrix);
    newtype!(StateVector, DVector);
    newtype!(CovarianceMatrix);

    newtype!(Measurement, N);
    newtype!(MeasurementNoiseVariance, N);
    newtype!(MeasurementMatrixRow, RowDVector);

    pub trait SystemMatrix<N : Real> { fn matrix(&self) -> &DMatrix<N>; }
    impl<N:Real> SystemMatrix<N> for DiscreteSystemMatrix<N> {
        fn matrix(&self) -> &DMatrix<N> { &self.0 }
    }
    impl<N:Real> SystemMatrix<N> for ContinuousSystemMatrix<N> {
        fn matrix(&self) -> &DMatrix<N> { &self.0 }
    }

    pub trait InputMatrix<N : Real> { fn matrix(&self) -> &DMatrix<N>; }
    impl<N:Real> InputMatrix<N> for DiscreteInputMatrix<N> {
        fn matrix(&self) -> &DMatrix<N> { &self.0 }
    }
    impl<N:Real> InputMatrix<N> for ContinuousInputMatrix<N> {
        fn matrix(&self) -> &DMatrix<N> { &self.0 }
    }


}