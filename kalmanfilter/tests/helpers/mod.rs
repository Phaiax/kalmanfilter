
pub mod model;
pub mod analysis;
pub mod types;

use na::{Matrix, Real, Dim};
use na::storage::Storage;

pub fn max<N : Real, C : Dim, R: Dim, S : Storage<N, C, R>>(mat : &Matrix<N, C, R, S>) -> N {
    mat.iter().fold(N::zero(), |store, item| { store.max(*item) })
}