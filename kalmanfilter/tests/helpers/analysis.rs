#![allow(dead_code)]

use super::types::{SystemMatrix, MeasurementMatrix};
use na::DMatrix;

pub fn is_observable(mat_f : &SystemMatrix, mat_c : &MeasurementMatrix, eps: f64) -> bool {
    kalman_observability_index(mat_f, mat_c, eps).is_some()
}

pub fn is_observable2(mat_f : &SystemMatrix, mat_c : &MeasurementMatrix, eps: f64) -> bool {
    hautus_observable_eigenvalues(mat_f, mat_c, eps).len() > 0
}

fn validate(mat_f : &SystemMatrix, mat_c : &MeasurementMatrix) -> (usize, usize) {
    let num_states = mat_f.nrows();
    assert!(num_states >= 1);
    assert_eq!(num_states, mat_f.ncols());
    assert_eq!(num_states, mat_c.ncols());
    let num_measurements = mat_c.nrows();
    assert!(num_measurements >= 1);
    (num_states, num_measurements)
}

pub fn kalman_observability_index(mat_f : &SystemMatrix, mat_c : &MeasurementMatrix, eps: f64) -> Option<usize> {
    let (num_states, num_measurements) = validate(mat_f, mat_c);

    //
    // Q_observable = [   H       ]   |  extend step by step
    //                [ H F       ]   v
    //                [  ...      ]
    //                [ H F^(m-1) ]
    //
    // Let m go from 1 to num_states. If Q_observable has rank num_states, the corresponding m
    // in the current extend step is the observability index.

    let rows = num_measurements * num_states;
    let mut q : DMatrix<f64> = DMatrix::zeros(rows, num_states);

    let mut sub_q = mat_c.clone();
    q.rows_mut(0, num_measurements).copy_from(&mat_c);

    if num_measurements >= num_states {
        let rank = q.rows(0, 1 * num_measurements).rank(eps);
        if rank == num_states {
            return Some(1)
        }
    }

    for i in 1..num_states {
        sub_q *= mat_f;
        q.rows_mut(i*num_measurements, num_measurements).copy_from(&sub_q);

        if (i+1) * num_measurements >= num_states {
            let rank = q.rows(0, (i+1) * num_measurements).rank(eps);
            if rank == num_states {
                return Some(i+1);
            }
        }
    }
    None
}

///
/// # Criteria of Hautus
///
/// Observable if for each eigenvalue l the matrix
///
/// [ l I - F ]
/// [    H    ]
///
/// has rank `num_states`.
///
pub fn hautus_observable_eigenvalues(mat_f : &SystemMatrix, mat_c : &MeasurementMatrix, eps: f64) -> Vec<f64> {
    let (num_states, num_measurements) = validate(mat_f, mat_c);
    if let Some(eigenvalues) = mat_f.eigenvalues() {
        let mut evm = DMatrix::zeros(num_states, num_states);
        let mut m = DMatrix::zeros(num_states + num_measurements, num_states);
        m.rows_mut(num_states, num_measurements).copy_from(&mat_c);
        let mut observable_eigenvalues = Vec::with_capacity(num_states);
        for &l in eigenvalues.iter() {
            evm.fill_diagonal(l);
            m.rows_mut(0, num_states).copy_from( &(&evm - mat_f) );
            if m.rank(eps) == num_states {
                observable_eigenvalues.push(l);
            }
        }
        observable_eigenvalues
    } else {
        vec![]
    }
}

#[test]
fn test_observability() {
    let f1 = &SystemMatrix::from_row_slice(2, 2, &[2., 1., 3., 0.]);
    let h1 = &MeasurementMatrix::from_row_slice(1, 2, &[0., 2.]);

    let f2 = &SystemMatrix::from_row_slice(2, 2, &[2., 1., 3., 0.]);
    let h2 = &MeasurementMatrix::from_row_slice(1, 2, &[1., 2.]);

    let f3 = &SystemMatrix::from_row_slice(2, 2, &[2., 1., 3., 0.]);
    let h3 = &MeasurementMatrix::from_row_slice(1, 2, &[0., 0.]);

    let f4 = &SystemMatrix::from_row_slice(2, 2, &[2., 1., 3., 0.]);
    let h4 = &MeasurementMatrix::from_row_slice(2, 2, &[0., 2., 1., 0.]);

    let f5 = &SystemMatrix::from_row_slice(2, 2, &[2., 1., 1., 2.]);
    let h5 = &MeasurementMatrix::from_row_slice(2, 2, &[-1., 1., 1., -1.]);


    assert_eq!(Some(2), kalman_observability_index(f1, h1, 0.0001));
    assert_eq!(Some(2), kalman_observability_index(f2, h2, 0.0001));
    assert_eq!(None, kalman_observability_index(f3, h3, 0.0001));
    assert_eq!(Some(1), kalman_observability_index(f4, h4, 0.0001));
    assert_eq!(None, kalman_observability_index(f5, h5, 0.0001));

    assert_eq!(2, hautus_observable_eigenvalues(f1, h1, 0.0001).len());
    assert_eq!(2, hautus_observable_eigenvalues(f2, h2, 0.0001).len());
    assert_eq!(0, hautus_observable_eigenvalues(f3, h3, 0.0001).len());
    assert_eq!(2, hautus_observable_eigenvalues(f4, h4, 0.0001).len());
    assert_eq!(1, hautus_observable_eigenvalues(f5, h5, 0.0001).len());


}