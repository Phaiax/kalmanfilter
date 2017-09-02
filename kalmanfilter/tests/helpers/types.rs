
use std::ops::{Deref, DerefMut};
use na::{DMatrix, DVector};
use kalmanfilter::nt;

macro_rules! newtype {
    ($newt:tt) => (
        newtype!($newt, DMatrix);
    );
    ($newt:tt, n) => (
        #[derive(Clone)]
        pub struct $newt(pub f64);

        impl Deref for $newt {
            type Target = f64;
            fn deref(&self) -> &f64 { &self.0 }
        }
        impl DerefMut for $newt {
            fn deref_mut(&mut self) -> &mut f64 { &mut self.0 }
        }
    );
    ($newt:tt, $mattype:tt) => (
        #[derive(Clone)]
        pub struct $newt(pub $mattype<f64>);

        impl Deref for $newt {
            type Target = $mattype<f64>;
            fn deref(&self) -> &$mattype<f64> { &self.0 }
        }
        impl DerefMut for $newt {
            fn deref_mut(&mut self) -> &mut $mattype<f64> { &mut self.0 }
        }
    );
}

pub type DiscreteSystemMatrix = nt::DiscreteSystemMatrix<f64>;
pub type ContinuousSystemMatrix = nt::ContinuousSystemMatrix<f64>;
newtype!(SystemNoiseVariances, DVector);
pub type SystemMatrix = nt::SystemMatrix<f64>;

pub type InputVector = nt::InputVector<f64>;
pub type ContinuousInputMatrix = nt::ContinuousInputMatrix<f64>;
pub type DiscreteInputMatrix = nt::DiscreteInputMatrix<f64>;

newtype!(Measurements, DVector);
newtype!(MeasurementMatrix);
newtype!(MeasurementNoiseVariances, DVector);

pub type SystemState = nt::StateVector<f64>;
pub type TimeStep = f64;
