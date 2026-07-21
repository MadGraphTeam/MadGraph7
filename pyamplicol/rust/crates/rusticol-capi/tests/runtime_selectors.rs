// SPDX-License-Identifier: 0BSD

use rusticol_capi::{
    RUSTICOL_STATUS_INVALID_ARGUMENT, RUSTICOL_STATUS_OK, RusticolRuntimeHandle,
    rusticol_last_error_message, rusticol_runtime_color_count, rusticol_runtime_color_id,
    rusticol_runtime_evaluate_resolved_f64, rusticol_runtime_evaluate_selected_f64,
    rusticol_runtime_free, rusticol_runtime_helicity_count, rusticol_runtime_helicity_id,
    rusticol_runtime_load,
};
use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};
use std::ptr;

fn fixture_root() -> Option<PathBuf> {
    std::env::var_os("RUSTICOL_SELECTOR_ARTIFACT").map(PathBuf::from)
}

fn contracted_fixture_root() -> Option<PathBuf> {
    std::env::var_os("RUSTICOL_CONTRACTED_SELECTOR_ARTIFACT").map(PathBuf::from)
}

fn last_error() -> String {
    let mut required = 0;
    // SAFETY: This is a zero-capacity query with writable required-size storage.
    let status = unsafe { rusticol_last_error_message(ptr::null_mut(), 0, &mut required) };
    if status != RUSTICOL_STATUS_OK || required == 0 {
        return format!("could not query Rusticol error (status {status})");
    }
    let mut buffer = vec![0 as c_char; required];
    // SAFETY: The output buffer has the queried capacity.
    let status =
        unsafe { rusticol_last_error_message(buffer.as_mut_ptr(), buffer.len(), &mut required) };
    if status != RUSTICOL_STATUS_OK {
        return format!("could not copy Rusticol error (status {status})");
    }
    // SAFETY: Successful Rusticol string getters write a trailing NUL.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn load_fixture(root: &Path) -> *mut RusticolRuntimeHandle {
    let root = CString::new(root.to_string_lossy().as_bytes()).expect("fixture C path");
    let mut handle = ptr::null_mut();
    // SAFETY: The path and output handle storage remain valid for the call.
    let status =
        unsafe { rusticol_runtime_load(root.as_ptr(), ptr::null(), ptr::null(), &mut handle) };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    assert!(!handle.is_null());
    handle
}

fn validation_momenta(root: &Path) -> Vec<f64> {
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("artifact.json")).expect("read artifact manifest"),
    )
    .expect("parse artifact manifest");
    let process_id = manifest["default_process_id"]
        .as_str()
        .expect("default process id");
    let payload: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            root.join("processes")
                .join(process_id)
                .join("validation-momenta.json"),
        )
        .expect("read validation momenta"),
    )
    .expect("parse validation momenta");
    payload["points"][0]
        .as_array()
        .expect("validation point")
        .iter()
        .flat_map(|leg| {
            leg["momentum"]
                .as_array()
                .expect("validation momentum")
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .expect("string momentum component")
                        .parse::<f64>()
                        .expect("f64 momentum component")
                })
        })
        .collect()
}

fn repeated_momenta(root: &Path, point_count: usize) -> Vec<f64> {
    let point = validation_momenta(root);
    point.repeat(point_count)
}

fn physical_counts(handle: *mut RusticolRuntimeHandle) -> (usize, usize) {
    let mut helicity_count = 0;
    let mut color_count = 0;
    // SAFETY: Both outputs are writable and the runtime handle remains live.
    assert_eq!(
        unsafe { rusticol_runtime_helicity_count(handle, &mut helicity_count) },
        RUSTICOL_STATUS_OK,
        "{}",
        last_error(),
    );
    // SAFETY: Both outputs are writable and the runtime handle remains live.
    assert_eq!(
        unsafe { rusticol_runtime_color_count(handle, &mut color_count) },
        RUSTICOL_STATUS_OK,
        "{}",
        last_error(),
    );
    (helicity_count, color_count)
}

fn helicity_id(handle: *mut RusticolRuntimeHandle, index: usize) -> CString {
    let mut required = 0;
    // SAFETY: This is a zero-capacity query against a live runtime.
    assert_eq!(
        unsafe { rusticol_runtime_helicity_id(handle, index, ptr::null_mut(), 0, &mut required) },
        RUSTICOL_STATUS_OK,
        "{}",
        last_error(),
    );
    let mut buffer = vec![0 as c_char; required];
    // SAFETY: The buffer has the queried capacity.
    assert_eq!(
        unsafe {
            rusticol_runtime_helicity_id(
                handle,
                index,
                buffer.as_mut_ptr(),
                buffer.len(),
                &mut required,
            )
        },
        RUSTICOL_STATUS_OK,
        "{}",
        last_error(),
    );
    // SAFETY: A successful getter writes a trailing NUL.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_owned()
}

fn color_id(handle: *mut RusticolRuntimeHandle, index: usize) -> CString {
    let mut required = 0;
    // SAFETY: This is a zero-capacity query against a live runtime.
    assert_eq!(
        unsafe { rusticol_runtime_color_id(handle, index, ptr::null_mut(), 0, &mut required) },
        RUSTICOL_STATUS_OK,
        "{}",
        last_error(),
    );
    let mut buffer = vec![0 as c_char; required];
    // SAFETY: The buffer has the queried capacity.
    assert_eq!(
        unsafe {
            rusticol_runtime_color_id(
                handle,
                index,
                buffer.as_mut_ptr(),
                buffer.len(),
                &mut required,
            )
        },
        RUSTICOL_STATUS_OK,
        "{}",
        last_error(),
    );
    // SAFETY: A successful getter writes a trailing NUL.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_owned()
}

fn resolved_values(
    handle: *mut RusticolRuntimeHandle,
    momenta: &[f64],
    point_count: usize,
    helicity_count: usize,
    color_count: usize,
) -> Vec<f64> {
    let mut output_helicity_count = 0;
    let mut output_color_count = 0;
    let mut values = vec![f64::NAN; point_count * helicity_count * color_count];
    // SAFETY: Every buffer remains live and has the declared length. Null selectors request all
    // physical components.
    let status = unsafe {
        rusticol_runtime_evaluate_resolved_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            values.as_mut_ptr(),
            values.len(),
            &mut output_helicity_count,
            &mut output_color_count,
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    assert_eq!(output_helicity_count, helicity_count);
    assert_eq!(output_color_count, color_count);
    values
}

fn expected_component(
    resolved: &[f64],
    point: usize,
    helicity: usize,
    color: usize,
    helicity_count: usize,
    color_count: usize,
) -> f64 {
    resolved[(point * helicity_count + helicity) * color_count + color]
}

fn expected_selected_sum(
    resolved: &[f64],
    point: usize,
    helicity: Option<usize>,
    color: Option<usize>,
    helicity_count: usize,
    color_count: usize,
) -> f64 {
    (0..helicity_count)
        .filter(|index| helicity.is_none_or(|selected| *index == selected))
        .flat_map(|helicity_index| {
            (0..color_count)
                .filter(move |index| color.is_none_or(|selected| *index == selected))
                .map(move |color_index| {
                    expected_component(
                        resolved,
                        point,
                        helicity_index,
                        color_index,
                        helicity_count,
                        color_count,
                    )
                })
        })
        .sum()
}

#[test]
fn homogeneous_and_alternating_per_point_selectors_match_resolved_components() {
    let Some(root) = fixture_root() else {
        return;
    };
    let handle = load_fixture(&root);
    let point_count = 4;
    let momenta = repeated_momenta(&root, point_count);
    let (helicity_count, color_count) = physical_counts(handle);
    assert!(
        helicity_count >= 2,
        "fixture must expose at least two helicities"
    );
    assert!(
        color_count >= 2,
        "fixture must expose at least two color flows"
    );
    let resolved = resolved_values(handle, &momenta, point_count, helicity_count, color_count);

    let helicity = helicity_id(handle, 0);
    let color = color_id(handle, 0);
    let global_helicities = [helicity.as_ptr()];
    let global_colors = [color.as_ptr()];
    let mut global = vec![f64::NAN; point_count];
    // SAFETY: All selector, momentum, and output arrays have their declared lengths.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            global_helicities.as_ptr(),
            global_helicities.len(),
            global_colors.as_ptr(),
            global_colors.len(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            global.as_mut_ptr(),
            global.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    for (point, value) in global.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            expected_component(&resolved, point, 0, 0, helicity_count, color_count).to_bits(),
        );
    }

    let mut global_helicity_only = vec![f64::NAN; point_count];
    // SAFETY: All selector, momentum, and output arrays have their declared lengths.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            global_helicities.as_ptr(),
            global_helicities.len(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            ptr::null(),
            0,
            global_helicity_only.as_mut_ptr(),
            global_helicity_only.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    for (point, value) in global_helicity_only.iter().enumerate() {
        let expected =
            expected_selected_sum(&resolved, point, Some(0), None, helicity_count, color_count);
        assert_eq!(value.to_bits(), expected.to_bits());
    }

    let mut global_color_only = vec![f64::NAN; point_count];
    // SAFETY: All selector, momentum, and output arrays have their declared lengths.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            global_colors.as_ptr(),
            global_colors.len(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            global_color_only.as_mut_ptr(),
            global_color_only.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    for (point, value) in global_color_only.iter().enumerate() {
        let expected =
            expected_selected_sum(&resolved, point, None, Some(0), helicity_count, color_count);
        assert_eq!(value.to_bits(), expected.to_bits());
    }

    let homogeneous_helicities = vec![0_u32; point_count];
    let homogeneous_colors = vec![0_u32; point_count];
    let mut homogeneous = vec![f64::NAN; point_count];
    // SAFETY: All selector, momentum, and output arrays have their declared lengths.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            homogeneous_helicities.as_ptr(),
            homogeneous_helicities.len(),
            homogeneous_colors.as_ptr(),
            homogeneous_colors.len(),
            homogeneous.as_mut_ptr(),
            homogeneous.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    for (point, value) in homogeneous.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            expected_component(&resolved, point, 0, 0, helicity_count, color_count).to_bits(),
        );
    }

    let pooled_helicities = [0_u32, 0, 1, 1];
    let pooled_colors = [1_u32, 1, 0, 0];
    let mut pooled = vec![f64::NAN; point_count];
    // SAFETY: All selector, momentum, and output arrays have their declared lengths.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            pooled_helicities.as_ptr(),
            pooled_helicities.len(),
            pooled_colors.as_ptr(),
            pooled_colors.len(),
            pooled.as_mut_ptr(),
            pooled.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    for (point, value) in pooled.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            expected_component(
                &resolved,
                point,
                pooled_helicities[point] as usize,
                pooled_colors[point] as usize,
                helicity_count,
                color_count,
            )
            .to_bits(),
        );
    }

    let alternating_helicities = [0_u32, 1, 0, 1];
    let alternating_colors = [1_u32, 0, 1, 0];
    let mut alternating = vec![f64::NAN; point_count];
    // SAFETY: All selector, momentum, and output arrays have their declared lengths.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            alternating_helicities.as_ptr(),
            alternating_helicities.len(),
            alternating_colors.as_ptr(),
            alternating_colors.len(),
            alternating.as_mut_ptr(),
            alternating.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    for (point, value) in alternating.iter().enumerate() {
        let helicity = alternating_helicities[point] as usize;
        let color = alternating_colors[point] as usize;
        assert_eq!(
            value.to_bits(),
            expected_component(
                &resolved,
                point,
                helicity,
                color,
                helicity_count,
                color_count,
            )
            .to_bits(),
        );
    }

    // First four entries from a selector stream seeded with 0xC0FFEE.
    let random_helicities = [0_u32, 0, 0, 1];
    let random_colors = [1_u32, 1, 0, 1];
    let mut seeded_random = vec![f64::NAN; point_count];
    // SAFETY: All selector, momentum, and output arrays have their declared lengths.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            random_helicities.as_ptr(),
            random_helicities.len(),
            random_colors.as_ptr(),
            random_colors.len(),
            seeded_random.as_mut_ptr(),
            seeded_random.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    for (point, value) in seeded_random.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            expected_component(
                &resolved,
                point,
                random_helicities[point] as usize,
                random_colors[point] as usize,
                helicity_count,
                color_count,
            )
            .to_bits(),
        );
    }

    let mut helicity_only = vec![f64::NAN; point_count];
    // SAFETY: All selector, momentum, and output arrays have their declared lengths.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            alternating_helicities.as_ptr(),
            alternating_helicities.len(),
            ptr::null(),
            0,
            helicity_only.as_mut_ptr(),
            helicity_only.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    for (point, value) in helicity_only.iter().enumerate() {
        let expected = expected_selected_sum(
            &resolved,
            point,
            Some(alternating_helicities[point] as usize),
            None,
            helicity_count,
            color_count,
        );
        assert_eq!(value.to_bits(), expected.to_bits());
    }

    let mut color_only = vec![f64::NAN; point_count];
    // SAFETY: All selector, momentum, and output arrays have their declared lengths.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            ptr::null(),
            0,
            alternating_colors.as_ptr(),
            alternating_colors.len(),
            color_only.as_mut_ptr(),
            color_only.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    for (point, value) in color_only.iter().enumerate() {
        let expected = expected_selected_sum(
            &resolved,
            point,
            None,
            Some(alternating_colors[point] as usize),
            helicity_count,
            color_count,
        );
        assert_eq!(value.to_bits(), expected.to_bits());
    }

    // SAFETY: The handle is consumed exactly once after all evaluations finish.
    assert_eq!(
        unsafe { rusticol_runtime_free(handle) },
        RUSTICOL_STATUS_OK,
        "{}",
        last_error(),
    );
}

#[test]
fn malformed_selector_buffers_fail_without_touching_output() {
    let Some(root) = fixture_root() else {
        return;
    };
    let handle = load_fixture(&root);
    let point_count = 2;
    let momenta = repeated_momenta(&root, point_count);
    let valid = [0_u32, 0];
    let mut output = [123.0, 456.0];

    // SAFETY: A nonzero length with a null selector pointer is intentionally malformed.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            ptr::null(),
            point_count,
            ptr::null(),
            0,
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    assert!(last_error().contains("helicity_by_point is null"));
    assert_eq!(output, [123.0, 456.0]);

    // SAFETY: The selector pointer is readable, but its declared length is intentionally short.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            valid.as_ptr(),
            1,
            ptr::null(),
            0,
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    assert!(last_error().contains("expected 2"));
    assert_eq!(output, [123.0, 456.0]);

    let helicity = helicity_id(handle, 0);
    let global_helicities = [helicity.as_ptr()];
    // SAFETY: Buffers are valid, but the same selector axis is intentionally supplied twice.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            global_helicities.as_ptr(),
            global_helicities.len(),
            ptr::null(),
            0,
            valid.as_ptr(),
            valid.len(),
            ptr::null(),
            0,
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    assert!(last_error().contains("mutually exclusive"));
    assert_eq!(output, [123.0, 456.0]);

    let color = color_id(handle, 0);
    let global_colors = [color.as_ptr()];
    // SAFETY: Buffers are valid, but the same color selector axis is supplied twice.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            global_colors.as_ptr(),
            global_colors.len(),
            ptr::null(),
            0,
            valid.as_ptr(),
            valid.len(),
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    assert!(last_error().contains("mutually exclusive"));
    assert_eq!(output, [123.0, 456.0]);

    let out_of_range = [u32::MAX, u32::MAX];
    // SAFETY: Buffers are valid; selector values are intentionally outside the physical domain.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            out_of_range.as_ptr(),
            out_of_range.len(),
            ptr::null(),
            0,
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    assert!(last_error().contains("out of range"));
    assert_eq!(output, [123.0, 456.0]);

    // SAFETY: The color selector is readable, but its declared length is intentionally short.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            ptr::null(),
            0,
            valid.as_ptr(),
            1,
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    assert!(last_error().contains("expected 2"));
    assert_eq!(output, [123.0, 456.0]);

    // SAFETY: Buffers are valid; color selector values are outside the physical domain.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            ptr::null(),
            0,
            out_of_range.as_ptr(),
            out_of_range.len(),
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    assert!(last_error().contains("out of range"));
    assert_eq!(output, [123.0, 456.0]);

    // SAFETY: The handle is consumed exactly once after all error cases finish.
    assert_eq!(
        unsafe { rusticol_runtime_free(handle) },
        RUSTICOL_STATUS_OK,
        "{}",
        last_error(),
    );
}

#[test]
fn contracted_color_rejects_global_and_per_point_flow_selectors() {
    let Some(root) = contracted_fixture_root() else {
        return;
    };
    let handle = load_fixture(&root);
    let point_count = 2;
    let momenta = repeated_momenta(&root, point_count);
    let color = color_id(handle, 0);
    let global_colors = [color.as_ptr()];
    let point_colors = [0_u32, 0];
    let mut output = [123.0, 456.0];

    // SAFETY: Buffers are valid; a global physical-flow selection is intentionally requested
    // from a contracted NLC/full color axis.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            global_colors.as_ptr(),
            global_colors.len(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    assert!(last_error().contains("LC color-flow selection is unavailable"));
    assert_eq!(output, [123.0, 456.0]);

    // SAFETY: Buffers are valid; per-point physical-flow selection is intentionally requested
    // from the same contracted color axis.
    let status = unsafe {
        rusticol_runtime_evaluate_selected_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            point_count,
            ptr::null(),
            0,
            ptr::null(),
            0,
            ptr::null(),
            0,
            point_colors.as_ptr(),
            point_colors.len(),
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    assert!(last_error().contains("LC color-flow selection is unavailable"));
    assert_eq!(output, [123.0, 456.0]);

    // SAFETY: The handle is consumed exactly once after both rejection cases finish.
    assert_eq!(
        unsafe { rusticol_runtime_free(handle) },
        RUSTICOL_STATUS_OK,
        "{}",
        last_error(),
    );
}
