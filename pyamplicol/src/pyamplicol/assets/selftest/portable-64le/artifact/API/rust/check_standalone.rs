// SPDX-License-Identifier: 0BSD

use std::env;
use std::ffi::{CStr, CString};
use std::fmt::Write as _;
use std::fs;
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};
use std::ptr;

const RUSTICOL_STATUS_OK: c_int = 0;

#[repr(C)]
struct RusticolRuntimeHandle {
    _private: [u8; 0],
}

type StringGetter =
    unsafe extern "C" fn(*const RusticolRuntimeHandle, *mut c_char, usize, *mut usize) -> c_int;
type IndexedStringGetter = unsafe extern "C" fn(
    *const RusticolRuntimeHandle,
    usize,
    *mut c_char,
    usize,
    *mut usize,
) -> c_int;

unsafe extern "C" {
    fn rusticol_last_error_message(
        buffer: *mut c_char,
        capacity: usize,
        required: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_load(
        process_dir: *const c_char,
        process_key: *const c_char,
        model_parameters_path: *const c_char,
        output: *mut *mut RusticolRuntimeHandle,
    ) -> c_int;
    fn rusticol_runtime_free(handle: *mut RusticolRuntimeHandle) -> c_int;
    fn rusticol_runtime_process(
        handle: *const RusticolRuntimeHandle,
        buffer: *mut c_char,
        capacity: usize,
        required: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_process_key(
        handle: *const RusticolRuntimeHandle,
        buffer: *mut c_char,
        capacity: usize,
        required: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_color_accuracy(
        handle: *const RusticolRuntimeHandle,
        buffer: *mut c_char,
        capacity: usize,
        required: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_external_count(
        handle: *const RusticolRuntimeHandle,
        output: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_external_pdg(
        handle: *const RusticolRuntimeHandle,
        index: usize,
        output: *mut i32,
    ) -> c_int;
    fn rusticol_runtime_helicity_count(
        handle: *const RusticolRuntimeHandle,
        output: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_helicity_id(
        handle: *const RusticolRuntimeHandle,
        index: usize,
        buffer: *mut c_char,
        capacity: usize,
        required: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_helicity_vector(
        handle: *const RusticolRuntimeHandle,
        index: usize,
        output: *mut i32,
        capacity: usize,
        required: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_color_count(
        handle: *const RusticolRuntimeHandle,
        output: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_color_id(
        handle: *const RusticolRuntimeHandle,
        index: usize,
        buffer: *mut c_char,
        capacity: usize,
        required: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_color_kind(
        handle: *const RusticolRuntimeHandle,
        index: usize,
        buffer: *mut c_char,
        capacity: usize,
        required: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_color_word(
        handle: *const RusticolRuntimeHandle,
        index: usize,
        output: *mut usize,
        capacity: usize,
        required: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_resolved_shape(
        handle: *const RusticolRuntimeHandle,
        helicity_ids: *const *const c_char,
        helicity_count: usize,
        color_ids: *const *const c_char,
        color_count: usize,
        output_helicity_count: *mut usize,
        output_color_count: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_evaluate_f64(
        handle: *mut RusticolRuntimeHandle,
        momenta: *const f64,
        momentum_count: usize,
        point_count: usize,
        output: *mut f64,
        output_capacity: usize,
    ) -> c_int;
    fn rusticol_runtime_evaluate_resolved_f64(
        handle: *mut RusticolRuntimeHandle,
        momenta: *const f64,
        momentum_count: usize,
        point_count: usize,
        helicity_ids: *const *const c_char,
        helicity_count: usize,
        color_ids: *const *const c_char,
        color_count: usize,
        output: *mut f64,
        output_capacity: usize,
        output_helicity_count: *mut usize,
        output_color_count: *mut usize,
    ) -> c_int;
    fn rusticol_runtime_set_model_parameters(
        handle: *mut RusticolRuntimeHandle,
        names: *const *const c_char,
        real: *const f64,
        imaginary: *const f64,
        count: usize,
    ) -> c_int;
}

#[derive(Default)]
struct Options {
    process: Option<String>,
    model_parameters: Option<String>,
    overrides: Vec<ParameterOverride>,
    precision: u32,
    json: bool,
    help: bool,
}

struct ParameterOverride {
    name: String,
    real: f64,
    imaginary: f64,
}

struct ExternalParticle {
    index: usize,
    pdg: i32,
}

struct HelicityConfiguration {
    id: String,
    helicities: Vec<i32>,
}

struct ColorComponent {
    id: String,
    kind: String,
    word: Vec<usize>,
}

struct Metadata {
    process: String,
    process_key: String,
    color_accuracy: String,
    particles: Vec<ExternalParticle>,
    helicities: Vec<HelicityConfiguration>,
    colors: Vec<ColorComponent>,
}

struct ResolvedEvaluation {
    values: Vec<f64>,
    helicity_count: usize,
    color_count: usize,
}

struct Runtime {
    handle: *mut RusticolRuntimeHandle,
}

impl Runtime {
    fn load(
        root: &Path,
        process: Option<&str>,
        model_parameters: Option<&str>,
    ) -> Result<Self, String> {
        let root = path_cstring(root)?;
        let process = optional_cstring(process, "process ID")?;
        let model_parameters = optional_cstring(model_parameters, "model-parameter path")?;
        let mut handle = ptr::null_mut();
        check(unsafe {
            rusticol_runtime_load(
                root.as_ptr(),
                optional_pointer(&process),
                optional_pointer(&model_parameters),
                &mut handle,
            )
        })?;
        if handle.is_null() {
            return Err("Rusticol returned a null runtime handle".to_owned());
        }
        Ok(Self { handle })
    }

    fn metadata(&self) -> Result<Metadata, String> {
        let mut external_count = 0;
        check(unsafe { rusticol_runtime_external_count(self.handle, &mut external_count) })?;
        let mut particles = Vec::with_capacity(external_count);
        for index in 0..external_count {
            let mut pdg = 0;
            check(unsafe { rusticol_runtime_external_pdg(self.handle, index, &mut pdg) })?;
            particles.push(ExternalParticle { index, pdg });
        }

        let mut helicity_count = 0;
        check(unsafe { rusticol_runtime_helicity_count(self.handle, &mut helicity_count) })?;
        let mut helicities = Vec::with_capacity(helicity_count);
        for index in 0..helicity_count {
            let mut required = 0;
            check(unsafe {
                rusticol_runtime_helicity_vector(
                    self.handle,
                    index,
                    ptr::null_mut(),
                    0,
                    &mut required,
                )
            })?;
            let mut vector = vec![0; required];
            if required != 0 {
                check(unsafe {
                    rusticol_runtime_helicity_vector(
                        self.handle,
                        index,
                        vector.as_mut_ptr(),
                        vector.len(),
                        &mut required,
                    )
                })?;
            }
            helicities.push(HelicityConfiguration {
                id: self.get_indexed_string(rusticol_runtime_helicity_id, index)?,
                helicities: vector,
            });
        }

        let mut color_count = 0;
        check(unsafe { rusticol_runtime_color_count(self.handle, &mut color_count) })?;
        let mut colors = Vec::with_capacity(color_count);
        for index in 0..color_count {
            let mut required = 0;
            check(unsafe {
                rusticol_runtime_color_word(self.handle, index, ptr::null_mut(), 0, &mut required)
            })?;
            let mut word = vec![0; required];
            if required != 0 {
                check(unsafe {
                    rusticol_runtime_color_word(
                        self.handle,
                        index,
                        word.as_mut_ptr(),
                        word.len(),
                        &mut required,
                    )
                })?;
            }
            colors.push(ColorComponent {
                id: self.get_indexed_string(rusticol_runtime_color_id, index)?,
                kind: self.get_indexed_string(rusticol_runtime_color_kind, index)?,
                word,
            });
        }

        Ok(Metadata {
            process: self.get_string(rusticol_runtime_process)?,
            process_key: self.get_string(rusticol_runtime_process_key)?,
            color_accuracy: self.get_string(rusticol_runtime_color_accuracy)?,
            particles,
            helicities,
            colors,
        })
    }

    fn set_model_parameters(&mut self, overrides: &[ParameterOverride]) -> Result<(), String> {
        if overrides.is_empty() {
            return Ok(());
        }
        let names = overrides
            .iter()
            .map(|item| cstring(&item.name, "model-parameter name"))
            .collect::<Result<Vec<_>, _>>()?;
        let name_pointers = names.iter().map(|name| name.as_ptr()).collect::<Vec<_>>();
        let real = overrides.iter().map(|item| item.real).collect::<Vec<_>>();
        let imaginary = overrides
            .iter()
            .map(|item| item.imaginary)
            .collect::<Vec<_>>();
        check(unsafe {
            rusticol_runtime_set_model_parameters(
                self.handle,
                name_pointers.as_ptr(),
                real.as_ptr(),
                imaginary.as_ptr(),
                overrides.len(),
            )
        })
    }

    fn evaluate(&mut self, momenta: &[f64]) -> Result<f64, String> {
        let mut output = 0.0;
        check(unsafe {
            rusticol_runtime_evaluate_f64(
                self.handle,
                momenta.as_ptr(),
                momenta.len(),
                1,
                &mut output,
                1,
            )
        })?;
        Ok(output)
    }

    fn evaluate_resolved(&mut self, momenta: &[f64]) -> Result<ResolvedEvaluation, String> {
        let mut helicity_count = 0;
        let mut color_count = 0;
        check(unsafe {
            rusticol_runtime_resolved_shape(
                self.handle,
                ptr::null(),
                0,
                ptr::null(),
                0,
                &mut helicity_count,
                &mut color_count,
            )
        })?;
        let mut values = vec![0.0; helicity_count * color_count];
        check(unsafe {
            rusticol_runtime_evaluate_resolved_f64(
                self.handle,
                momenta.as_ptr(),
                momenta.len(),
                1,
                ptr::null(),
                0,
                ptr::null(),
                0,
                values.as_mut_ptr(),
                values.len(),
                &mut helicity_count,
                &mut color_count,
            )
        })?;
        Ok(ResolvedEvaluation {
            values,
            helicity_count,
            color_count,
        })
    }

    fn get_string(&self, getter: StringGetter) -> Result<String, String> {
        let mut required = 0;
        check(unsafe { getter(self.handle, ptr::null_mut(), 0, &mut required) })?;
        copy_string(required, |buffer, capacity, output_required| unsafe {
            getter(self.handle, buffer, capacity, output_required)
        })
    }

    fn get_indexed_string(
        &self,
        getter: IndexedStringGetter,
        index: usize,
    ) -> Result<String, String> {
        let mut required = 0;
        check(unsafe { getter(self.handle, index, ptr::null_mut(), 0, &mut required) })?;
        copy_string(required, |buffer, capacity, output_required| unsafe {
            getter(self.handle, index, buffer, capacity, output_required)
        })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                rusticol_runtime_free(self.handle);
            }
        }
    }
}

fn last_error() -> String {
    let mut required = 0;
    unsafe {
        rusticol_last_error_message(ptr::null_mut(), 0, &mut required);
    }
    if required == 0 {
        return "unknown Rusticol error".to_owned();
    }
    let mut buffer = vec![0_u8; required];
    let status = unsafe {
        rusticol_last_error_message(buffer.as_mut_ptr().cast(), buffer.len(), &mut required)
    };
    if status != RUSTICOL_STATUS_OK {
        return format!("Rusticol error (status {status})");
    }
    c_buffer_string(&buffer).unwrap_or_else(|error| error)
}

fn check(status: c_int) -> Result<(), String> {
    if status == RUSTICOL_STATUS_OK {
        Ok(())
    } else {
        Err(last_error())
    }
}

fn copy_string(
    required: usize,
    copy: impl FnOnce(*mut c_char, usize, *mut usize) -> c_int,
) -> Result<String, String> {
    if required == 0 {
        return Err("Rusticol returned an empty string buffer".to_owned());
    }
    let mut buffer = vec![0_u8; required];
    let mut copied_required = required;
    check(copy(
        buffer.as_mut_ptr().cast(),
        buffer.len(),
        &mut copied_required,
    ))?;
    c_buffer_string(&buffer)
}

fn c_buffer_string(buffer: &[u8]) -> Result<String, String> {
    CStr::from_bytes_until_nul(buffer)
        .map_err(|_| "Rusticol returned a string without a trailing NUL".to_owned())?
        .to_str()
        .map(str::to_owned)
        .map_err(|_| "Rusticol returned a non-UTF-8 string".to_owned())
}

fn cstring(value: &str, description: &str) -> Result<CString, String> {
    CString::new(value).map_err(|_| format!("{description} contains a NUL byte"))
}

fn path_cstring(path: &Path) -> Result<CString, String> {
    cstring(&path.to_string_lossy(), "artifact path")
}

fn optional_cstring(value: Option<&str>, description: &str) -> Result<Option<CString>, String> {
    value.map(|item| cstring(item, description)).transpose()
}

fn optional_pointer(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |item| item.as_ptr())
}

fn usage() -> &'static str {
    "usage: check_standalone [--process ID] [--model-parameters PATH] \
[--set-parameter NAME REAL IMAG] [--precision 16] [--json]"
}

fn parse_options() -> Result<Options, String> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let mut options = Options {
        precision: 16,
        ..Options::default()
    };
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        match argument.as_str() {
            "--process" => {
                options.process = Some(required_argument(&arguments, index, 1)?.to_owned());
                index += 2;
            }
            "--model-parameters" => {
                options.model_parameters =
                    Some(required_argument(&arguments, index, 1)?.to_owned());
                index += 2;
            }
            "--set-parameter" => {
                let name = required_argument(&arguments, index, 1)?.to_owned();
                let real = parse_f64(
                    required_argument(&arguments, index, 2)?,
                    "real model-parameter component",
                )?;
                let imaginary = parse_f64(
                    required_argument(&arguments, index, 3)?,
                    "imaginary model-parameter component",
                )?;
                options.overrides.push(ParameterOverride {
                    name,
                    real,
                    imaginary,
                });
                index += 4;
            }
            "--precision" => {
                options.precision = required_argument(&arguments, index, 1)?
                    .parse()
                    .map_err(|_| "invalid --precision value".to_owned())?;
                index += 2;
            }
            "--json" => {
                options.json = true;
                index += 1;
            }
            "--help" | "-h" => {
                options.help = true;
                index += 1;
            }
            _ => return Err(format!("unknown option: {argument}")),
        }
    }
    if options.precision != 16 {
        return Err(
            "the Rust Rusticol API supports only double precision (--precision 16)".to_owned(),
        );
    }
    Ok(options)
}

fn required_argument<'a>(
    arguments: &'a [String],
    option_index: usize,
    offset: usize,
) -> Result<&'a str, String> {
    arguments
        .get(option_index + offset)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value after {}", arguments[option_index]))
}

fn parse_f64(value: &str, description: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("invalid {description}"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(format!("invalid {description}"))
    }
}

fn artifact_root() -> Result<PathBuf, String> {
    let current =
        env::current_dir().map_err(|error| format!("cannot read current directory: {error}"))?;
    if current.join("artifact.json").is_file() {
        return Ok(current);
    }

    let executable = env::current_exe()
        .map_err(|error| format!("cannot locate check_standalone executable: {error}"))?;
    for ancestor in executable.ancestors().skip(1) {
        if ancestor.join("artifact.json").is_file() {
            return Ok(ancestor.to_path_buf());
        }
    }

    let language_dir = executable.parent();
    let build_artifact = language_dir.and_then(Path::parent);
    let build_root = build_artifact.and_then(Path::parent);
    if build_root.and_then(Path::file_name) == Some(".pyamplicol-api-build".as_ref()) {
        if let (Some(parent), Some(name)) = (
            build_root.and_then(Path::parent),
            build_artifact.and_then(Path::file_name),
        ) {
            let candidate = parent.join(name);
            if candidate.join("artifact.json").is_file() {
                return Ok(candidate);
            }
        }
    }

    Err("run check_standalone from a generated artifact directory".to_owned())
}

fn load_validation_point(
    path: &Path,
    process_key: &str,
    external_count: usize,
) -> Result<Option<Vec<f64>>, String> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot read {}: {error}", path.display())),
    };
    let mut lines = contents.lines();
    if lines.next() != Some("RUSTICOL_VALIDATION_POINTS_V1") {
        return Err("unsupported validation_points.dat format".to_owned());
    }
    for line in lines {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() < 2 || fields[0] != process_key {
            continue;
        }
        let row_count = fields[1]
            .parse::<usize>()
            .map_err(|_| "invalid validation point row".to_owned())?;
        if row_count != external_count || fields.len() != 2 + 4 * external_count {
            return Err("validation point has an incompatible external-particle count".to_owned());
        }
        let momenta = fields[2..]
            .iter()
            .map(|value| parse_f64(value, "validation-point component"))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Some(momenta));
    }
    Ok(None)
}

fn json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{1f}' => {
                write!(&mut output, "\\u{:04x}", character as u32)
                    .expect("writing to a String cannot fail");
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn write_i32_array(output: &mut String, values: &[i32]) {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").expect("writing to a String cannot fail");
    }
    output.push(']');
}

fn write_usize_array(output: &mut String, values: &[usize]) {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").expect("writing to a String cannot fail");
    }
    output.push(']');
}

fn write_f64_array(output: &mut String, values: &[f64]) -> Result<(), String> {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err("matrix-element output is not finite".to_owned());
        }
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").expect("writing to a String cannot fail");
    }
    output.push(']');
    Ok(())
}

fn write_common_json(output: &mut String, metadata: &Metadata) {
    write!(
        output,
        "\"process\":{},\"process_key\":{},\"color_accuracy\":{}",
        json_string(&metadata.process),
        json_string(&metadata.process_key),
        json_string(&metadata.color_accuracy),
    )
    .expect("writing to a String cannot fail");

    output.push_str(",\"external_particles\":[");
    for (index, particle) in metadata.particles.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"index\":{},\"pdg\":{}}}",
            particle.index, particle.pdg
        )
        .expect("writing to a String cannot fail");
    }

    output.push_str("],\"helicities\":[");
    for (index, helicity) in metadata.helicities.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"id\":{},\"helicities\":",
            json_string(&helicity.id)
        )
        .expect("writing to a String cannot fail");
        write_i32_array(output, &helicity.helicities);
        output.push('}');
    }

    output.push_str("],\"colors\":[");
    for (index, color) in metadata.colors.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"id\":{},\"kind\":{},\"word\":",
            json_string(&color.id),
            json_string(&color.kind),
        )
        .expect("writing to a String cannot fail");
        write_usize_array(output, &color.word);
        output.push('}');
    }
    output.push(']');
}

fn available_json(
    metadata: &Metadata,
    resolved: &ResolvedEvaluation,
    explicit_total: f64,
    total: f64,
) -> Result<String, String> {
    let mut output = "{\"language\":\"rust\",\"available\":true,\"precision\":16,".to_owned();
    write_common_json(&mut output, metadata);
    write!(
        &mut output,
        ",\"shape\":[1,{},{}],\"values\":",
        resolved.helicity_count, resolved.color_count
    )
    .expect("writing to a String cannot fail");
    write_f64_array(&mut output, &resolved.values)?;
    output.push_str(",\"resolved_sum\":");
    write_f64_array(&mut output, &[explicit_total])?;
    output.push_str(",\"compatibility_total\":");
    write_f64_array(&mut output, &[total])?;
    output.push_str("}\n");
    Ok(output)
}

fn unavailable_json(metadata: &Metadata) -> String {
    let mut output = "{\"language\":\"rust\",\"available\":false,".to_owned();
    write_common_json(&mut output, metadata);
    output.push_str(",\"diagnostic\":\"no bundled validation point is available\"}\n");
    output
}

fn run() -> Result<(), String> {
    let options = parse_options()?;
    if options.help {
        println!("{}", usage());
        return Ok(());
    }
    let root = artifact_root()?;
    let mut runtime = Runtime::load(
        &root,
        options.process.as_deref(),
        options.model_parameters.as_deref(),
    )?;
    runtime.set_model_parameters(&options.overrides)?;
    let metadata = runtime.metadata()?;
    let point = load_validation_point(
        &root.join("API/validation_points.dat"),
        &metadata.process_key,
        metadata.particles.len(),
    )?;
    let Some(momenta) = point else {
        if options.json {
            print!("{}", unavailable_json(&metadata));
        } else {
            println!("process: {}", metadata.process);
            println!("no bundled validation point is available; metadata load succeeded");
        }
        return Ok(());
    };

    let total = runtime.evaluate(&momenta)?;
    let resolved = runtime.evaluate_resolved(&momenta)?;
    if resolved.helicity_count != metadata.helicities.len()
        || resolved.color_count != metadata.colors.len()
    {
        return Err("resolved shape does not match runtime metadata".to_owned());
    }
    let explicit_total = resolved.values.iter().sum::<f64>();
    let scale = total.abs().max(1.0);
    if (explicit_total - total).abs() > 1.0e-12 * scale {
        return Err("resolved components do not reproduce the compatibility total".to_owned());
    }

    if options.json {
        print!(
            "{}",
            available_json(&metadata, &resolved, explicit_total, total)?
        );
        return Ok(());
    }

    println!("process: {} [{}]", metadata.process, metadata.process_key);
    println!(
        "resolved shape: (1, {}, {})",
        resolved.helicity_count, resolved.color_count
    );
    for (helicity_index, helicity) in metadata.helicities.iter().enumerate() {
        for (color_index, color) in metadata.colors.iter().enumerate() {
            let offset = helicity_index * resolved.color_count + color_index;
            println!(
                "  {}  {}  {}",
                helicity.id, color.id, resolved.values[offset]
            );
        }
    }
    println!("explicit resolved sum: {explicit_total}");
    println!("compatibility total:   {total}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("check_standalone: {error}");
        std::process::exit(1);
    }
}
