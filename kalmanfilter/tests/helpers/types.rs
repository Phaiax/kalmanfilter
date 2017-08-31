
use na::{DMatrix, DVector};

pub type Measurements = DVector<f64>;
pub type SystemMatrix = DMatrix<f64>;
pub type InputVector = DVector<f64>;
pub type InputMatrix = DMatrix<f64>;
pub type SystemNoise = DVector<f64>;
pub type MeasurementMatrix = DMatrix<f64>;
pub type MeasurementNoise = DVector<f64>;
pub type SystemState = DVector<f64>;
pub type TimeStep = f64;
