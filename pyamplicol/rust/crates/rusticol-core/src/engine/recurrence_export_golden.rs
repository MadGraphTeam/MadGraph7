// SPDX-License-Identifier: 0BSD

//! Golden vectors of the recurrence export's execution semantics.
//!
//! The parity test compares Rusticol with the export interpreter, but the
//! interpreter reuses Rusticol's own vertex, propagator and wavefunction
//! formulas: a formula change keeps that test green while external runners,
//! which implement the published semantics, silently stop matching. These
//! vectors pin the bits of every exported instruction kind that carries
//! arithmetic (each vertex and propagator template, every source family and
//! orientation with their branches, scale, copy, the scatter shortcuts,
//! closure, the kernel input projections and output adapter, momentum forms,
//! reductions and the group factor) to the `execution_semantics` string.
//!
//! The committed fixture `tests/fixtures/recurrence_export_golden.json` holds
//! the inputs (instruction operands, initial arenas and momenta as f64 bit
//! patterns) and the expected output bits. The checker drives the export
//! interpreter (`ExportMachine`) from the fixture's inputs alone. A changed bit
//! fails with "the execution semantics changed: bump the version string and
//! regenerate". To regenerate after bumping `EXECUTION_SEMANTICS`:
//!
//! ```console
//! cargo test -p rusticol-core --lib regenerate_recurrence_export_golden_vectors -- --ignored
//! ```
//!
//! The generator refuses to change an existing case's bits while the version
//! string is unchanged; adding cases is allowed.
//!
//! Platforms: the pinned arithmetic is IEEE-754 binary64 `+ - * /`, `sqrt`,
//! `abs`, sign operations and the explicit `mul_add` of the reduction, all
//! correctly rounded on every IEEE platform. Rust never contracts `a * b + c`
//! into an FMA, the paired vertex path (`wide::f64x2`) performs the same
//! IEEE operations lane by lane, and no transcendental libm function is
//! involved. Every input and output is finite (the generator refuses NaN,
//! whose payload bits are platform specific) and no value is subnormal-flushed
//! (Rust runs without FTZ/DAZ). So the bits are platform independent; the
//! fixture was generated on x86_64 Linux, the platform of this repository's
//! CI. The identity kernel used by the kernel cases only copies planes, so
//! these vectors pin the adapter, not SymJIT's code generation.

use super::recurrence_export::EXECUTION_SEMANTICS;
use super::recurrence_export_tests::{ExportMachine, apply_export_group_factor};
use super::*;
use crate::engine::evaluator::recurrence_intrinsic_direct::{
    FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE, RecurrenceContributionIntrinsicKind,
    RecurrenceFinalizationIntrinsicKind, WEYL_PROPAGATOR_NEGATIVE_TEMPLATE,
    WEYL_PROPAGATOR_POSITIVE_TEMPLATE,
};
use crate::engine::evaluator::recurrence_source_direct::{
    DirectSourceOrientation, DirectSourceWavefunctionFamily,
};
use crate::recurrence::direct_export::{
    DirectExportInstruction, DirectExportKernelBinding, DirectExportMode, DirectExportSource,
    DirectExportSourceEmbedding, ExportComplex, RelativeKernelInput,
};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/recurrence_export_golden.json"
);
const POINTS: usize = 5;
const CURRENT_COMPONENTS: usize = 48;
const AMPLITUDES: usize = 2;
const MASS: f64 = 0.75;
// Operand layout of every instruction case: two parents and a destination
// that cannot overlap for any family (a spin-2 current has 16 components).
const LEFT: u32 = 0;
const RIGHT: u32 = 16;
const DESTINATION: u32 = 24;
const FORM_MASSLESS: u32 = 0;
const FORM_MASSLESS_CROSSED: u32 = 1;
const FORM_MASSIVE: u32 = 2;
const FORM_MASSIVE_CROSSED: u32 = 3;
const FORM_OFF_SHELL_LEFT: u32 = 4;
const FORM_OFF_SHELL_RIGHT: u32 = 5;
const FORM_COUNT: usize = 8;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct GoldenFile {
    execution_semantics: String,
    about: String,
    points: usize,
    /// `[form][Lorentz component]`: the points' f64 bit patterns as one hex
    /// string of 16 digits per point.
    momentum_forms: Vec<Vec<String>>,
    /// `[component]`: `re, im` bit patterns of each point, 16 hex digits each.
    currents: Vec<String>,
    amplitudes: Vec<String>,
    cases: Vec<GoldenCase>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct GoldenCase {
    name: String,
    #[serde(flatten)]
    input: GoldenInput,
    expected: GoldenExpected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "case", rename_all = "snake_case")]
enum GoldenInput {
    /// Execute one instruction on the shared initial arenas and momenta.
    /// Kernel bindings name `identity-N`: a saved application whose N complex
    /// outputs copy its N complex inputs.
    Instruction {
        instruction: DirectExportInstruction,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        sources: Vec<DirectExportSource>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        kernel_bindings: Vec<DirectExportKernelBinding>,
    },
    /// Evaluate momentum forms from caller momenta `[point][leg]` (four
    /// components each, hex as above).
    MomentumForms {
        external_count: usize,
        external: Vec<String>,
        forms: Vec<Vec<GoldenTerm>>,
        crossing: Option<Vec<GoldenCrossing>>,
    },
    /// Apply reduction terms to the shared initial amplitudes, then the group
    /// factor, starting from positive-zero outputs.
    Reduction {
        helicity_count: usize,
        reduction: Vec<GoldenReduction>,
        group_factor: GoldenGroupFactor,
    },
}

// Mirrors of the export's serialized structures (the public types are
// serialize-only).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct GoldenTerm {
    leg: u32,
    coefficient: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct GoldenCrossing {
    program_leg: usize,
    caller_leg: usize,
    sign: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct GoldenReduction {
    helicity_output: usize,
    amplitude: u32,
    amplitude_scale: ExportComplex,
    apply_amplitude_scale: bool,
    weight: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct GoldenGroupFactor {
    numerator: u64,
    denominator: u64,
}

/// Bits after the case. Instruction cases list only the arena slots whose
/// bits differ from the initial arenas; every other slot must keep its bits.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct GoldenExpected {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    currents: BTreeMap<u32, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    amplitudes: BTreeMap<u32, String>,
    /// Momentum-form cases: `[form][component]` as `momentum_forms`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    momentum_forms: Vec<Vec<String>>,
    /// Reduction cases: `[point]`, the helicity outputs' bits.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    outputs: Vec<String>,
}

fn hex(values: impl IntoIterator<Item = f64>) -> String {
    values
        .into_iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect()
}

fn unhex(text: &str) -> Vec<f64> {
    assert!(
        text.len().is_multiple_of(16),
        "malformed golden hex {text:?}"
    );
    (0..text.len() / 16)
        .map(|index| {
            f64::from_bits(u64::from_str_radix(&text[index * 16..index * 16 + 16], 16).unwrap())
        })
        .collect()
}

/// `re, im` of every point of slot `slot`.
fn slot_hex(re: &[f64], im: &[f64], slot: usize) -> String {
    hex((0..POINTS).flat_map(|point| [re[slot * POINTS + point], im[slot * POINTS + point]]))
}

fn set_slot(re: &mut [f64], im: &mut [f64], slot: usize, text: &str) {
    let values = unhex(text);
    assert_eq!(
        values.len(),
        2 * POINTS,
        "golden slot {slot} has {POINTS} points"
    );
    for point in 0..POINTS {
        re[slot * POINTS + point] = values[2 * point];
        im[slot * POINTS + point] = values[2 * point + 1];
    }
}

fn complex(re: f64, im: f64) -> ExportComplex {
    ExportComplex { re, im }
}

// ---------------------------------------------------------------- inputs

/// SplitMix64: a fixed, platform-independent bit source for the arenas.
struct Bits(u64);

impl Bits {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        // [-2, 2) with a full mantissa.
        (z >> 11) as f64 / (1_u64 << 51) as f64 - 2.0
    }
}

fn on_shell(mass: f64, p: [f64; 3], energy_sign: f64) -> [f64; 4] {
    let energy = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + mass * mass).sqrt();
    [energy_sign * energy, p[0], p[1], p[2]]
}

/// `[form][point]` momenta. Generic components carry full mantissas, so
/// reassociating any sum changes bits. Each on-shell form has three generic
/// points and the wavefunctions' two axis (or rest and axis) branches; the
/// paired vertex and propagator paths take points 0-1 and 2-3, the scalar
/// tail point 4.
fn generator_momenta() -> Vec<[[f64; 4]; POINTS]> {
    let mut bits = Bits(0x6d6f_6d65_6e74_756d);
    let mut three =
        || -> [[f64; 3]; 3] { std::array::from_fn(|_| std::array::from_fn(|_| bits.next())) };
    let (generic, other) = (three(), three());
    let on_shell_form = |mass: f64, momenta: [[f64; 3]; 3], branches: [[f64; 3]; 2], sign: f64| {
        [
            on_shell(mass, momenta[0], sign),
            on_shell(mass, branches[0], sign),
            on_shell(mass, momenta[1], sign),
            on_shell(mass, branches[1], sign),
            on_shell(mass, momenta[2], sign),
        ]
    };
    let mut forms = vec![
        on_shell_form(0.0, generic, [[0.0, 0.0, -1.4], [0.0, 0.0, 0.9]], 1.0),
        on_shell_form(0.0, other, [[0.0, 0.0, 1.1], [0.0, 0.0, -0.7]], -1.0),
        on_shell_form(MASS, generic, [[0.0, 0.0, 0.0], [0.0, 0.0, -1.2]], 1.0),
        on_shell_form(MASS, other, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]], -1.0),
    ];
    // Off-shell forms for vertex and propagator momenta.
    for _ in FORM_OFF_SHELL_LEFT as usize..FORM_COUNT {
        forms.push(std::array::from_fn(|_| {
            std::array::from_fn(|_| 2.0 * bits.next())
        }));
    }
    forms
}

fn generator_file_inputs() -> (Vec<Vec<String>>, Vec<String>, Vec<String>) {
    let momenta = generator_momenta();
    assert_eq!(momenta.len(), FORM_COUNT);
    let forms = momenta
        .iter()
        .map(|points| {
            (0..4)
                .map(|component| hex(points.iter().map(|p| p[component])))
                .collect()
        })
        .collect();
    let mut bits = Bits(0x5eed_0fec_0ee1_f000);
    let mut values = |count: usize| {
        (0..count * POINTS * 2)
            .map(|_| bits.next())
            .collect::<Vec<_>>()
    };
    let mut currents = values(CURRENT_COMPONENTS);
    // Signed zeros in operands: operand order decides the sign of results.
    currents[(LEFT as usize + 1) * POINTS * 2 + 2] = -0.0;
    currents[(LEFT as usize + 2) * POINTS * 2 + 1] = 0.0;
    currents[(RIGHT as usize + 1) * POINTS * 2 + 5] = -0.0;
    currents[(DESTINATION as usize) * POINTS * 2 + 3] = -0.0;
    let amplitudes = values(AMPLITUDES);
    let encode = |values: &[f64]| -> Vec<String> {
        values
            .chunks(POINTS * 2)
            .map(|slot| hex(slot.iter().copied()))
            .collect()
    };
    (forms, encode(&currents), encode(&amplitudes))
}

// ------------------------------------------------------------ coverage

/// Every contribution template. Adding a kind fails to compile here until it
/// is given a successor, so it cannot escape the golden vectors.
fn contribution_kinds() -> Vec<RecurrenceContributionIntrinsicKind> {
    use RecurrenceContributionIntrinsicKind as Kind;
    let next = |kind: Kind| match kind {
        Kind::WeylVectorToWeylPositive => Some(Kind::WeylVectorToWeylNegative),
        Kind::WeylVectorToWeylNegative => Some(Kind::AntisymmetricTensorVectorToVector),
        Kind::AntisymmetricTensorVectorToVector => {
            Some(Kind::VectorWedgeVectorToAntisymmetricTensor)
        }
        Kind::VectorWedgeVectorToAntisymmetricTensor => Some(Kind::ColorOrderedThreeVector),
        Kind::ColorOrderedThreeVector => None,
    };
    std::iter::successors(Some(Kind::WeylVectorToWeylPositive), |kind| next(*kind)).collect()
}

/// Every propagator template with its component count.
fn propagators() -> Vec<(&'static str, u32)> {
    use RecurrenceFinalizationIntrinsicKind as Kind;
    let describe = |kind: Kind| match kind {
        Kind::WeylPropagatorPositive => (WEYL_PROPAGATOR_POSITIVE_TEMPLATE, 2),
        Kind::WeylPropagatorNegative => (WEYL_PROPAGATOR_NEGATIVE_TEMPLATE, 2),
        Kind::FeynmanVectorPropagator => (FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE, 4),
    };
    let next = |kind: Kind| match kind {
        Kind::WeylPropagatorPositive => Some(Kind::WeylPropagatorNegative),
        Kind::WeylPropagatorNegative => Some(Kind::FeynmanVectorPropagator),
        Kind::FeynmanVectorPropagator => None,
    };
    std::iter::successors(Some(Kind::WeylPropagatorPositive), |kind| next(*kind))
        .map(describe)
        .collect()
}

fn source_families() -> Vec<(&'static str, u32)> {
    use DirectSourceWavefunctionFamily as Family;
    let describe = |family: Family| match family {
        Family::Scalar => ("scalar", 1),
        Family::WeylFermion => ("weyl", 2),
        Family::DiracFermion => ("dirac", 4),
        Family::Vector => ("vector", 4),
        Family::Spin2 => ("spin2", 16),
    };
    let next = |family: Family| match family {
        Family::Scalar => Some(Family::WeylFermion),
        Family::WeylFermion => Some(Family::DiracFermion),
        Family::DiracFermion => Some(Family::Vector),
        Family::Vector => Some(Family::Spin2),
        Family::Spin2 => None,
    };
    std::iter::successors(Some(Family::Scalar), |family| next(*family))
        .map(describe)
        .collect()
}

fn orientations() -> Vec<&'static str> {
    use DirectSourceOrientation as Orientation;
    let describe = |orientation: Orientation| match orientation {
        Orientation::Particle => "particle",
        Orientation::Antiparticle => "antiparticle",
        Orientation::SelfConjugate => "self_conjugate",
    };
    [
        Orientation::Particle,
        Orientation::Antiparticle,
        Orientation::SelfConjugate,
    ]
    .into_iter()
    .map(describe)
    .collect()
}

/// The exported instruction kinds; adding one fails to compile here.
fn instruction_kind(instruction: &DirectExportInstruction) -> &'static str {
    match instruction {
        DirectExportInstruction::Clear { .. } => "clear",
        DirectExportInstruction::ClearAmplitude { .. } => "clear_amplitude",
        DirectExportInstruction::Source { .. } => "source",
        DirectExportInstruction::Vertex { .. } => "vertex",
        DirectExportInstruction::Propagator { .. } => "propagator",
        DirectExportInstruction::Scale { .. } => "scale",
        DirectExportInstruction::Copy { .. } => "copy",
        DirectExportInstruction::Scatter { .. } => "scatter",
        DirectExportInstruction::Closure { .. } => "closure",
        DirectExportInstruction::Kernel { .. } => "kernel",
    }
}

const INSTRUCTION_KINDS: [&str; 10] = [
    "clear",
    "clear_amplitude",
    "source",
    "vertex",
    "propagator",
    "scale",
    "copy",
    "scatter",
    "closure",
    "kernel",
];

// --------------------------------------------------------------- cases

fn instruction_case(name: String, instruction: DirectExportInstruction) -> GoldenCase {
    GoldenCase {
        name,
        input: GoldenInput::Instruction {
            instruction,
            sources: Vec::new(),
            kernel_bindings: Vec::new(),
        },
        expected: GoldenExpected::default(),
    }
}

#[allow(clippy::too_many_arguments)]
fn source_case(
    name: String,
    family: &str,
    component_count: u32,
    orientation: &str,
    form: u32,
    helicity: i32,
    chirality: i32,
    mass: f64,
    embedding: Vec<DirectExportSourceEmbedding>,
) -> GoldenCase {
    GoldenCase {
        name,
        input: GoldenInput::Instruction {
            instruction: DirectExportInstruction::Source { source: 0 },
            sources: vec![DirectExportSource {
                current: DESTINATION,
                component_count,
                family: family.to_owned(),
                orientation: orientation.to_owned(),
                leg: 0,
                momentum_form: form,
                helicity,
                chirality,
                mass: complex(mass, 0.0),
                factor: complex(0.875, -0.375),
                embedding,
            }],
            kernel_bindings: Vec::new(),
        },
        expected: GoldenExpected::default(),
    }
}

fn mode_name(mode: DirectExportMode) -> &'static str {
    match mode {
        DirectExportMode::Set => "set",
        DirectExportMode::Add => "add",
    }
}

fn generator_cases() -> Vec<GoldenCase> {
    let modes = [DirectExportMode::Set, DirectExportMode::Add];
    let mut cases = Vec::new();
    cases.push(instruction_case(
        "clear".into(),
        DirectExportInstruction::Clear {
            current: DESTINATION,
            component_count: 6,
        },
    ));
    cases.push(instruction_case(
        "clear_amplitude".into(),
        DirectExportInstruction::ClearAmplitude { amplitude: 1 },
    ));
    for kind in contribution_kinds() {
        let template = kind.runtime_template();
        for mode in modes {
            cases.push(instruction_case(
                format!("vertex {template} {}", mode_name(mode)),
                DirectExportInstruction::Vertex {
                    vertex: template.to_owned(),
                    current: DESTINATION,
                    mode,
                    left: LEFT,
                    right: Some(RIGHT),
                    left_momentum: FORM_OFF_SHELL_LEFT,
                    right_momentum: Some(FORM_OFF_SHELL_RIGHT),
                    coupling: complex(0.375, -1.21875),
                    factor: complex(-0.5, 0.3125),
                },
            ));
        }
    }
    // Every off-shell form: each gives the paired and the scalar tail path
    // more samples of the propagator denominator.
    for (template, component_count) in propagators() {
        for form in FORM_OFF_SHELL_LEFT..FORM_COUNT as u32 {
            cases.push(instruction_case(
                format!("propagator {template} form {form}"),
                DirectExportInstruction::Propagator {
                    propagator: template.to_owned(),
                    current: LEFT,
                    component_count,
                    momentum_form: form,
                    factor: complex(0.625, -0.4375),
                },
            ));
        }
    }
    // Sources: every family and orientation, both helicity and chirality
    // signs, massless and massive, positive and negative energy.
    let families = source_families();
    let count = |name: &str| {
        families
            .iter()
            .find(|(family, _)| *family == name)
            .expect("known family")
            .1
    };
    for (family, component_count) in &families {
        let mut variants: Vec<(String, u32, i32, i32, f64)> = Vec::new();
        match *family {
            "scalar" => {
                for (form, mass) in [(FORM_MASSLESS, 0.0), (FORM_MASSIVE, MASS)] {
                    variants.push((format!("form {form} mass {mass}"), form, 0, 0, mass));
                }
            }
            "weyl" => {
                for form in [FORM_MASSLESS, FORM_MASSLESS_CROSSED] {
                    for helicity in [-1, 1] {
                        for chirality in [-1, 1] {
                            variants.push((
                                format!("form {form} helicity {helicity} chirality {chirality}"),
                                form,
                                helicity,
                                chirality,
                                0.0,
                            ));
                        }
                    }
                }
            }
            "dirac" | "vector" => {
                for (form, mass) in [
                    (FORM_MASSLESS, 0.0),
                    (FORM_MASSLESS_CROSSED, 0.0),
                    (FORM_MASSIVE, MASS),
                    (FORM_MASSIVE_CROSSED, MASS),
                ] {
                    let helicities: &[i32] = if *family == "vector" && mass != 0.0 {
                        &[-1, 0, 1]
                    } else {
                        &[-1, 1]
                    };
                    for &helicity in helicities {
                        variants.push((
                            format!("form {form} helicity {helicity} mass {mass}"),
                            form,
                            helicity,
                            0,
                            mass,
                        ));
                    }
                }
            }
            "spin2" => {
                for helicity in [-2, 2] {
                    variants.push((
                        format!("form {FORM_MASSLESS} helicity {helicity} mass 0"),
                        FORM_MASSLESS,
                        helicity,
                        0,
                        0.0,
                    ));
                }
                for form in [FORM_MASSIVE, FORM_MASSIVE_CROSSED] {
                    for helicity in -2..=2 {
                        variants.push((
                            format!("form {form} helicity {helicity} mass {MASS}"),
                            form,
                            helicity,
                            0,
                            MASS,
                        ));
                    }
                }
            }
            other => panic!("golden vectors have no source variants for family {other}"),
        }
        // Fermion formulas depend on the orientation (particle or not); the
        // other families' formulas do not, so they use self_conjugate once.
        let fermion = matches!(*family, "weyl" | "dirac");
        for orientation in orientations()
            .into_iter()
            .filter(|orientation| fermion != (*orientation == "self_conjugate"))
        {
            for (label, form, helicity, chirality, mass) in &variants {
                cases.push(source_case(
                    format!("source {family} {orientation} {label}"),
                    family,
                    *component_count,
                    orientation,
                    *form,
                    *helicity,
                    *chirality,
                    *mass,
                    Vec::new(),
                ));
            }
        }
    }
    cases.push(source_case(
        "source vector embedded".into(),
        "vector",
        count("vector"),
        "self_conjugate",
        FORM_MASSLESS,
        1,
        0,
        0.0,
        vec![
            DirectExportSourceEmbedding {
                component: 0,
                source_component: Some(3),
                factor: complex(0.5, 0.5),
            },
            DirectExportSourceEmbedding {
                component: 1,
                source_component: None,
                factor: complex(1.0, 0.0),
            },
            DirectExportSourceEmbedding {
                component: 2,
                source_component: Some(1),
                factor: complex(-1.0, 0.0),
            },
            DirectExportSourceEmbedding {
                component: 5,
                source_component: Some(2),
                factor: complex(0.0, -1.5),
            },
        ],
    ));
    for (label, factor) in [
        ("general", complex(-0.34375, 1.6875)),
        ("unit", complex(1.0, 0.0)),
        ("negative zero imaginary", complex(1.0, -0.0)),
    ] {
        cases.push(instruction_case(
            format!("scale {label}"),
            DirectExportInstruction::Scale {
                current: LEFT,
                component_count: 4,
                factor,
            },
        ));
    }
    for (label, factor) in [
        ("zero", complex(0.0, 0.0)),
        ("general", complex(0.4375, -0.90625)),
        ("unit", complex(1.0, 0.0)),
    ] {
        cases.push(instruction_case(
            format!("copy {label}"),
            DirectExportInstruction::Copy {
                current: DESTINATION,
                source: LEFT,
                component_count: 4,
                factor,
            },
        ));
    }
    for (label, factor) in [
        ("unit", complex(1.0, 0.0)),
        ("minus unit", complex(-1.0, 0.0)),
        ("real", complex(0.3125, 0.0)),
        ("imaginary", complex(0.0, -0.8125)),
        ("general", complex(0.59375, 0.6875)),
    ] {
        for mode in modes {
            cases.push(instruction_case(
                format!("scatter {label} {}", mode_name(mode)),
                DirectExportInstruction::Scatter {
                    current: DESTINATION,
                    source: LEFT,
                    component_count: 4,
                    mode,
                    factor,
                },
            ));
        }
    }
    cases.push(instruction_case(
        "closure four components".into(),
        DirectExportInstruction::Closure {
            amplitude: 1,
            left: LEFT,
            right: RIGHT,
            component_factors: vec![
                complex(1.0, 0.0),
                complex(-1.0, 0.0),
                complex(0.0, 0.5),
                complex(0.75, -1.25),
            ],
            factor: complex(0.8125, -0.28125),
        },
    ));
    cases.push(instruction_case(
        "closure unit factor".into(),
        DirectExportInstruction::Closure {
            amplitude: 0,
            left: RIGHT,
            right: LEFT,
            component_factors: vec![complex(1.0, 0.0); 6],
            factor: complex(1.0, 0.0),
        },
    ));
    cases.extend(kernel_cases());
    cases.extend(momentum_form_cases());
    cases.extend(reduction_cases());
    cases
}

fn parent(parent: u8, component: u16, imaginary: bool) -> RelativeKernelInput {
    RelativeKernelInput::Parent {
        parent,
        component,
        imaginary,
    }
}

fn destination(component: u16, imaginary: bool, amplitude: bool) -> RelativeKernelInput {
    RelativeKernelInput::Destination {
        component,
        imaginary,
        amplitude,
    }
}

fn kernel_cases() -> Vec<GoldenCase> {
    let mut cases = Vec::new();
    // Every input projection kind feeds an identity kernel, so each output
    // plane shows one resolved input, then the output adapter.
    let projections = vec![
        parent(0, 0, false),
        parent(0, 0, true),
        parent(1, 1, false),
        RelativeKernelInput::Momentum {
            operand: 0,
            component: 3,
        },
        RelativeKernelInput::Literal { value: 0.3125 },
        RelativeKernelInput::Factor { imaginary: true },
        destination(1, true, false),
        RelativeKernelInput::Factor { imaginary: false },
    ];
    let outputs = (0..4)
        .flat_map(|component| {
            [
                destination(component, false, false),
                destination(component, true, false),
            ]
        })
        .collect::<Vec<_>>();
    let direct_inputs = vec![
        parent(0, 2, false),
        parent(0, 2, true),
        parent(1, 0, true),
        RelativeKernelInput::Momentum {
            operand: 1,
            component: 0,
        },
    ];
    let direct_outputs = (0..2)
        .flat_map(|component| {
            [
                destination(component, false, false),
                destination(component, true, false),
            ]
        })
        .collect::<Vec<_>>();
    for (label, inputs, outputs, direct) in [
        ("scaled", projections, outputs, false),
        ("direct", direct_inputs, direct_outputs, true),
    ] {
        for mode in [DirectExportMode::Set, DirectExportMode::Add] {
            cases.push(GoldenCase {
                name: format!("kernel {label} {}", mode_name(mode)),
                input: GoldenInput::Instruction {
                    instruction: DirectExportInstruction::Kernel {
                        binding: 0,
                        destination: DESTINATION,
                        amplitude: false,
                        left: Some(LEFT),
                        right: Some(RIGHT),
                        left_momentum: Some(FORM_OFF_SHELL_LEFT),
                        right_momentum: Some(FORM_OFF_SHELL_RIGHT),
                        mode,
                        factor: complex(0.59375, -1.3125),
                    },
                    sources: Vec::new(),
                    kernel_bindings: vec![DirectExportKernelBinding {
                        id: 0,
                        sha256: format!("identity-{}", inputs.len() / 2),
                        inputs: inputs.clone(),
                        outputs: outputs.clone(),
                        direct,
                    }],
                },
                expected: GoldenExpected::default(),
            });
        }
    }
    cases.push(GoldenCase {
        name: "kernel scaled add amplitude".into(),
        input: GoldenInput::Instruction {
            instruction: DirectExportInstruction::Kernel {
                binding: 0,
                destination: 1,
                amplitude: true,
                left: Some(LEFT),
                right: None,
                left_momentum: None,
                right_momentum: None,
                mode: DirectExportMode::Add,
                factor: complex(-0.71875, 0.40625),
            },
            sources: Vec::new(),
            kernel_bindings: vec![DirectExportKernelBinding {
                id: 0,
                // Two complex inputs, one complex output: the output copies
                // the first input.
                sha256: "identity-2-to-1".into(),
                inputs: vec![
                    parent(0, 3, true),
                    parent(0, 3, false),
                    destination(0, false, true),
                    destination(0, true, true),
                ],
                outputs: vec![destination(0, false, true), destination(0, true, true)],
                direct: false,
            }],
        },
        expected: GoldenExpected::default(),
    });
    cases
}

fn momentum_form_cases() -> Vec<GoldenCase> {
    let external_count = 4;
    let mut bits = Bits(0x0f0e_d0c0_b0a0_9080);
    let external = (0..POINTS)
        .map(|_| hex((0..external_count * 4).map(|_| bits.next())))
        .collect::<Vec<_>>();
    let forms = vec![
        vec![
            GoldenTerm {
                leg: 0,
                coefficient: 1.0,
            },
            GoldenTerm {
                leg: 1,
                coefficient: 1.0,
            },
        ],
        vec![
            GoldenTerm {
                leg: 2,
                coefficient: -1.0,
            },
            GoldenTerm {
                leg: 3,
                coefficient: 1.0,
            },
            GoldenTerm {
                leg: 0,
                coefficient: 1.0,
            },
        ],
        vec![GoldenTerm {
            leg: 3,
            coefficient: -1.0,
        }],
    ];
    let crossing = vec![
        GoldenCrossing {
            program_leg: 0,
            caller_leg: 1,
            sign: -1.0,
        },
        GoldenCrossing {
            program_leg: 1,
            caller_leg: 0,
            sign: -1.0,
        },
        GoldenCrossing {
            program_leg: 2,
            caller_leg: 3,
            sign: 1.0,
        },
        GoldenCrossing {
            program_leg: 3,
            caller_leg: 2,
            sign: 1.0,
        },
    ];
    [("uncrossed", None), ("crossed", Some(crossing))]
        .into_iter()
        .map(|(label, crossing)| GoldenCase {
            name: format!("momentum forms {label}"),
            input: GoldenInput::MomentumForms {
                external_count,
                external: external.clone(),
                forms: forms.clone(),
                crossing,
            },
            expected: GoldenExpected::default(),
        })
        .collect()
}

fn reduction_cases() -> Vec<GoldenCase> {
    let reduction = vec![
        GoldenReduction {
            helicity_output: 1,
            amplitude: 0,
            amplitude_scale: complex(0.0, 3.0),
            apply_amplitude_scale: true,
            weight: 1.25,
        },
        GoldenReduction {
            helicity_output: 0,
            amplitude: 1,
            amplitude_scale: complex(1.0, 0.0),
            apply_amplitude_scale: false,
            weight: 0.3671875,
        },
        GoldenReduction {
            helicity_output: 1,
            amplitude: 1,
            amplitude_scale: complex(-0.5, 0.875),
            apply_amplitude_scale: true,
            weight: -0.0625,
        },
    ];
    [(1, 1), (2, 3), (7, 5)]
        .into_iter()
        .map(|(numerator, denominator)| GoldenCase {
            name: format!("reduction group factor {numerator}/{denominator}"),
            input: GoldenInput::Reduction {
                helicity_count: 2,
                reduction: reduction.clone(),
                group_factor: GoldenGroupFactor {
                    numerator,
                    denominator,
                },
            },
            expected: GoldenExpected::default(),
        })
        .collect()
}

// ----------------------------------------------------------- execution

fn identity_kernel(name: &str) -> RecurrenceExportKernel {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    let (inputs, outputs) = match name
        .strip_prefix("identity-")
        .and_then(|sizes| sizes.split_once("-to-"))
    {
        Some((inputs, outputs)) => (inputs.parse().unwrap(), outputs.parse().unwrap()),
        None => {
            let count: usize = name
                .strip_prefix("identity-")
                .unwrap_or_else(|| panic!("golden kernel {name:?} is not an identity kernel"))
                .parse()
                .unwrap();
            (count, count)
        }
    };
    let assignments = (0..outputs)
        .map(|index| format!("('assign', ('out', {index}), ('param', {index}))"))
        .collect::<Vec<_>>()
        .join(", ");
    let bytes = evaluator::symjit_plane::compile_symbolica_program_to_plane_application_bytes(
        &format!("([{assignments}], 0, [])"),
        inputs,
        outputs,
        2,
        false,
    )
    .unwrap();
    RecurrenceExportKernel {
        sha256: name.to_owned(),
        application_base64: STANDARD.encode(bytes),
    }
}

/// The shared initial state of every instruction case.
fn initial_machine(file: &GoldenFile) -> ExportMachine {
    assert_eq!(file.points, POINTS, "golden vectors use {POINTS} points");
    let mut machine = ExportMachine::new(
        file.momentum_forms.len(),
        file.currents.len(),
        file.amplitudes.len(),
        POINTS,
    );
    for (form, components) in file.momentum_forms.iter().enumerate() {
        for (component, text) in components.iter().enumerate() {
            for (point, value) in unhex(text).into_iter().enumerate() {
                machine.momenta[(form * 4 + component) * POINTS + point] = value;
            }
        }
    }
    for (slot, text) in file.currents.iter().enumerate() {
        set_slot(&mut machine.current_re, &mut machine.current_im, slot, text);
    }
    for (slot, text) in file.amplitudes.iter().enumerate() {
        set_slot(
            &mut machine.amplitude_re,
            &mut machine.amplitude_im,
            slot,
            text,
        );
    }
    machine
}

/// Execute a case from its inputs and return its expected-format result.
fn evaluate_case(file: &GoldenFile, input: &GoldenInput) -> GoldenExpected {
    match input {
        GoldenInput::Instruction {
            instruction,
            sources,
            kernel_bindings,
        } => {
            let initial = initial_machine(file);
            let mut machine = initial_machine(file);
            let kernels = kernel_bindings
                .iter()
                .map(|binding| identity_kernel(&binding.sha256))
                .collect::<Vec<_>>();
            machine.execute(instruction, sources, kernel_bindings, &kernels);
            let changed = |before_re: &[f64], before_im: &[f64], re: &[f64], im: &[f64]| {
                (0..re.len() / POINTS)
                    .filter(|&slot| slot_hex(before_re, before_im, slot) != slot_hex(re, im, slot))
                    .map(|slot| (slot as u32, slot_hex(re, im, slot)))
                    .collect::<BTreeMap<_, _>>()
            };
            GoldenExpected {
                currents: changed(
                    &initial.current_re,
                    &initial.current_im,
                    &machine.current_re,
                    &machine.current_im,
                ),
                amplitudes: changed(
                    &initial.amplitude_re,
                    &initial.amplitude_im,
                    &machine.amplitude_re,
                    &machine.amplitude_im,
                ),
                ..GoldenExpected::default()
            }
        }
        GoldenInput::MomentumForms {
            external_count,
            external,
            forms,
            crossing,
        } => {
            let external = external
                .iter()
                .flat_map(|point| unhex(point))
                .collect::<Vec<_>>();
            let forms = forms
                .iter()
                .map(|terms| {
                    terms
                        .iter()
                        .map(|term| RecurrenceExportMomentumTerm {
                            leg: term.leg,
                            coefficient: term.coefficient,
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let crossing = crossing.as_ref().map(|entries| {
                entries
                    .iter()
                    .map(|entry| RecurrenceExportCrossing {
                        program_leg: entry.program_leg,
                        caller_leg: entry.caller_leg,
                        sign: entry.sign,
                    })
                    .collect::<Vec<_>>()
            });
            let mut machine = ExportMachine::new(forms.len(), 0, 0, POINTS);
            machine.evaluate_momentum_forms(
                &forms,
                crossing.as_deref(),
                *external_count,
                &external,
            );
            GoldenExpected {
                momentum_forms: (0..forms.len())
                    .map(|form| {
                        (0..4)
                            .map(|component| {
                                hex((0..POINTS).map(|point| {
                                    machine.momenta[(form * 4 + component) * POINTS + point]
                                }))
                            })
                            .collect()
                    })
                    .collect(),
                ..GoldenExpected::default()
            }
        }
        GoldenInput::Reduction {
            helicity_count,
            reduction,
            group_factor,
        } => {
            let machine = initial_machine(file);
            let reduction = reduction
                .iter()
                .map(|term| RecurrenceExportReduction {
                    helicity_output: term.helicity_output,
                    amplitude: term.amplitude,
                    amplitude_scale: term.amplitude_scale,
                    apply_amplitude_scale: term.apply_amplitude_scale,
                    weight: term.weight,
                })
                .collect::<Vec<_>>();
            let mut output = vec![0.0; POINTS * helicity_count];
            machine.reduce(&reduction, *helicity_count, &mut output);
            apply_export_group_factor(
                &mut output,
                NativeRationalFactor {
                    numerator: group_factor.numerator,
                    denominator: group_factor.denominator,
                },
            );
            GoldenExpected {
                outputs: output
                    .chunks(*helicity_count)
                    .map(|point| hex(point.iter().copied()))
                    .collect(),
                ..GoldenExpected::default()
            }
        }
    }
}

fn all_bits(expected: &GoldenExpected) -> Vec<f64> {
    expected
        .currents
        .values()
        .chain(expected.amplitudes.values())
        .chain(expected.momentum_forms.iter().flatten())
        .chain(expected.outputs.iter())
        .flat_map(|text| unhex(text))
        .collect()
}

fn generate() -> GoldenFile {
    let (momentum_forms, currents, amplitudes) = generator_file_inputs();
    let mut file = GoldenFile {
        execution_semantics: EXECUTION_SEMANTICS.to_owned(),
        about: "Golden vectors of the recurrence export's execution semantics; see \
                docs/reference/recurrence-export.md and \
                rust/crates/rusticol-core/src/engine/recurrence_export_golden.rs. \
                f64 values are IEEE-754 bit patterns, 16 hex digits each."
            .to_owned(),
        points: POINTS,
        momentum_forms,
        currents,
        amplitudes,
        cases: generator_cases(),
    };
    let inputs = file.clone();
    for case in &mut file.cases {
        case.expected = evaluate_case(&inputs, &case.input);
        // An empty result is valid (a unit scale keeps every bit); the
        // checker still compares every arena slot.
        let bits = all_bits(&case.expected);
        // NaN payloads are platform specific; the vectors must avoid them.
        assert!(
            bits.iter().all(|value| value.is_finite()),
            "golden case {:?} produces a non-finite value",
            case.name
        );
    }
    let mut names = BTreeSet::new();
    for case in &file.cases {
        assert!(
            names.insert(case.name.clone()),
            "duplicate golden case {:?}",
            case.name
        );
    }
    for value in file
        .momentum_forms
        .iter()
        .flatten()
        .chain(&file.currents)
        .chain(&file.amplitudes)
        .flat_map(|text| unhex(text))
    {
        assert!(value.is_finite(), "golden inputs must be finite");
    }
    file
}

fn read_fixture() -> GoldenFile {
    let bytes = fs::read(FIXTURE).unwrap_or_else(|error| {
        panic!("read the recurrence export golden vectors {FIXTURE}: {error}")
    });
    serde_json::from_slice(&bytes).expect("parse the recurrence export golden vectors")
}

fn input_kind(input: &GoldenInput) -> &'static str {
    match input {
        GoldenInput::Instruction { instruction, .. } => instruction_kind(instruction),
        GoldenInput::MomentumForms { .. } => "momentum_forms",
        GoldenInput::Reduction { .. } => "reduction",
    }
}

#[test]
fn recurrence_export_golden_vectors_pin_the_execution_semantics() {
    let fixture = read_fixture();
    let regenerate = "regenerate them with `cargo test -p rusticol-core --lib \
                      regenerate_recurrence_export_golden_vectors -- --ignored`";
    assert_eq!(
        fixture.execution_semantics, EXECUTION_SEMANTICS,
        "the golden vectors pin {:?} but the export declares {EXECUTION_SEMANTICS:?}: {regenerate}",
        fixture.execution_semantics
    );
    // Coverage: the fixture's inputs are the generator's, which enumerates
    // every template, family, orientation and instruction kind.
    let generated = generate_inputs_only();
    assert_eq!(
        fixture
            .cases
            .iter()
            .map(|case| (&case.name, &case.input))
            .collect::<Vec<_>>(),
        generated
            .cases
            .iter()
            .map(|case| (&case.name, &case.input))
            .collect::<Vec<_>>(),
        "the golden vector cases differ from the generator's (a template, family or \
         instruction kind was added or a case changed): {regenerate}"
    );
    assert_eq!(
        (
            &fixture.momentum_forms,
            &fixture.currents,
            &fixture.amplitudes
        ),
        (
            &generated.momentum_forms,
            &generated.currents,
            &generated.amplitudes
        ),
        "the golden vectors' shared inputs differ from the generator's: {regenerate}"
    );
    for kind in INSTRUCTION_KINDS
        .iter()
        .copied()
        .chain(["momentum_forms", "reduction"])
    {
        assert!(
            fixture
                .cases
                .iter()
                .any(|case| input_kind(&case.input) == kind),
            "no golden vector covers {kind}"
        );
    }
    let mut checked = 0;
    for case in &fixture.cases {
        let actual = evaluate_case(&fixture, &case.input);
        if actual != case.expected {
            panic!(
                "the execution semantics changed: bump the version string \
                 (EXECUTION_SEMANTICS in engine/recurrence_export.rs, now \
                 {EXECUTION_SEMANTICS:?}) and regenerate the golden vectors. Case {:?}: {}",
                case.name,
                describe_difference(&case.expected, &actual)
            );
        }
        checked += all_bits(&case.expected).len();
    }
    eprintln!(
        "recurrence export golden vectors: {} cases, {checked} f64 values bit for bit",
        fixture.cases.len()
    );
}

fn generate_inputs_only() -> GoldenFile {
    let (momentum_forms, currents, amplitudes) = generator_file_inputs();
    GoldenFile {
        execution_semantics: EXECUTION_SEMANTICS.to_owned(),
        about: String::new(),
        points: POINTS,
        momentum_forms,
        currents,
        amplitudes,
        cases: generator_cases(),
    }
}

fn describe_difference(expected: &GoldenExpected, actual: &GoldenExpected) -> String {
    let maps = [
        ("current", &expected.currents, &actual.currents),
        ("amplitude", &expected.amplitudes, &actual.amplitudes),
    ];
    for (label, expected, actual) in maps {
        for slot in expected.keys().chain(actual.keys()) {
            if expected.get(slot) != actual.get(slot) {
                return format!(
                    "{label} {slot}: expected {} got {} (re, im per point; absent = \
                     unchanged from the initial arena)",
                    describe_values(expected.get(slot)),
                    describe_values(actual.get(slot))
                );
            }
        }
    }
    format!(
        "momentum forms {:?} / outputs {:?} expected, got {:?} / {:?}",
        expected.momentum_forms, expected.outputs, actual.momentum_forms, actual.outputs
    )
}

fn describe_values(text: Option<&String>) -> String {
    text.map_or_else(
        || "unchanged".to_owned(),
        |text| format!("{:?}", unhex(text)),
    )
}

#[test]
#[ignore = "regenerates the committed golden vectors; run after bumping EXECUTION_SEMANTICS"]
fn regenerate_recurrence_export_golden_vectors() {
    let generated = generate();
    if let Ok(bytes) = fs::read(FIXTURE) {
        let existing: GoldenFile =
            serde_json::from_slice(&bytes).expect("parse the existing golden vectors");
        if existing.execution_semantics == generated.execution_semantics {
            // Under an unchanged version only new cases may appear.
            assert_eq!(
                (
                    &existing.momentum_forms,
                    &existing.currents,
                    &existing.amplitudes
                ),
                (
                    &generated.momentum_forms,
                    &generated.currents,
                    &generated.amplitudes
                ),
                "the shared inputs changed under the unchanged execution semantics \
                 {EXECUTION_SEMANTICS:?}: bump the version string first"
            );
            for case in &existing.cases {
                let regenerated = generated
                    .cases
                    .iter()
                    .find(|candidate| candidate.name == case.name);
                assert!(
                    regenerated == Some(case),
                    "golden case {:?} changed under the unchanged execution semantics \
                     {EXECUTION_SEMANTICS:?}: bump the version string first",
                    case.name
                );
            }
        }
    }
    let mut text = serde_json::to_string_pretty(&generated).unwrap();
    text.push('\n');
    fs::write(FIXTURE, text).expect("write the golden vectors");
    eprintln!(
        "wrote {} golden cases for {EXECUTION_SEMANTICS:?} to {FIXTURE}",
        generated.cases.len()
    );
}
