//! End-to-end C ABI tests of the selected-pole DP and binary128 rescue paths.
use super::*;
use ninjars::fast::tensor_layout::{monomial_index, ncoeff};

struct Handle(*mut Context);
impl Handle {
    fn new() -> Self {
        let mut pointer = std::ptr::null_mut();
        assert_eq!(
            unsafe { ninjars_context_create(ABI_VERSION, &mut pointer) },
            SUCCESS
        );
        Self(pointer)
    }
}
impl Drop for Handle {
    fn drop(&mut self) {
        assert_eq!(unsafe { ninjars_context_destroy(self.0) }, SUCCESS);
    }
}

fn records(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let mut record = [b' '; RECORD];
            let text = format!("{value:.39e}");
            record[..text.len()].copy_from_slice(text.as_bytes());
            record
        })
        .collect()
}
fn call(
    handle: &Handle,
    rank: usize,
    coefficient: f64,
    mode: i32,
    mp: bool,
    compensated: bool,
) -> (i32, [C64; 3]) {
    // Exact lightlike shift: every ordinary bubble master vanishes, but the
    // q0^rank numerator has UV residue (-2)^rank / (rank+1).
    let mut tensor = vec![C64::ZERO; ncoeff(rank)];
    tensor[monomial_index([rank as u32, 0, 0, 0])] = C64::real(coefficient);
    let matrix = [0.; 4];
    let momenta = [0., 0., 0., 0., 2., 0., 0., 2.];
    let masses = [C64::ZERO; 2];
    let mut output = [C64::real(991.); 3];
    let mut rational = C64::real(992.);
    let status = if mp {
        let coeff_text = records(&tensor.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>());
        let matrix_text = records(&matrix);
        let momenta_text = records(&momenta);
        let masses_text = records(&[0.; 4]);
        let scale_text = records(&[7.]);
        let mut result = vec![b'?'; 8 * RECORD];
        let status = unsafe {
            ninjars_evaluate_mp_with_poles(
                handle.0,
                2,
                rank as i32,
                tensor.len(),
                coeff_text.as_ptr(),
                coeff_text.len(),
                matrix_text.as_ptr(),
                matrix_text.len(),
                momenta_text.as_ptr(),
                momenta_text.len(),
                masses_text.as_ptr(),
                masses_text.len(),
                scale_text.as_ptr(),
                scale_text.len(),
                result.as_mut_ptr(),
                result.len(),
                mode,
            )
        };
        if status == SUCCESS {
            let values: Vec<f64> = result
                .chunks_exact(RECORD)
                .map(|text| std::str::from_utf8(text).unwrap().trim().parse().unwrap())
                .collect();
            for k in 0..3 {
                output[k] = C64::new(values[2 * k], values[2 * k + 1]);
            }
        } else {
            assert!(result.iter().all(|&v| v == b'?'));
        }
        status
    } else {
        let evaluate = if compensated {
            ninjars_evaluate_dp_compensated_with_poles
        } else {
            ninjars_evaluate_dp_with_poles
        };
        unsafe {
            evaluate(
                handle.0,
                2,
                rank as i32,
                tensor.len(),
                tensor.as_ptr(),
                matrix.as_ptr(),
                matrix.len(),
                momenta.as_ptr(),
                momenta.len(),
                masses.as_ptr(),
                masses.len(),
                7.,
                output.as_mut_ptr(),
                output.len(),
                &mut rational,
                1,
                mode,
            )
        }
    };
    (status, output)
}

#[test]
fn selected_poles_survive_scaleless_reduction_dp_mp_and_mode_changes() {
    let handle = Handle::new();
    for mp in [false, true] {
        for rank in 0..=4 {
            // Reuse one context across topologies, changing mode and numerator.
            for (mode, coefficient, sign) in [(1, 1., 1.), (2, 3., -1.), (1, -2., 1.)] {
                let (status, value) = call(&handle, rank, coefficient, mode, mp, false);
                assert_eq!(status, SUCCESS, "rank={rank}, mode={mode}, mp={mp}");
                let expected =
                    coefficient * sign * (-2.0_f64).powi(rank as i32) / (rank + 1) as f64;
                assert!(value[0].abs() < 1e-10, "finite changed: {value:?}");
                assert!(
                    (value[1] - C64::real(expected)).abs() < 1e-10,
                    "rank={rank} mode={mode} mp={mp}: {value:?}, expected {expected}"
                );
                assert!(value[2].abs() < 1e-10);
            }
        }
        // NONE and BOTH use the ordinary combined result (zero here).
        for mode in [3, 0] {
            let (status, result) = call(&handle, 0, 1., mode, mp, false);
            assert_eq!(status, SUCCESS);
            assert!(result.iter().all(|v| v.abs() < 1e-10));
        }
    }
    let (status, result) = call(&handle, 2, 1., 1, false, true);
    assert_eq!(status, SUCCESS);
    assert!((result[1] - C64::real(4. / 3.)).abs() < 1e-10);
}

#[test]
fn invalid_pole_mode_preserves_output_buffers() {
    let handle = Handle::new();
    for mp in [false, true] {
        for mode in [-1, 4, i32::MAX] {
            let (status, result) = call(&handle, 0, 1., mode, mp, false);
            assert_eq!(status, ERROR);
            assert_eq!(result, [C64::real(991.); 3]);
        }
    }
}
