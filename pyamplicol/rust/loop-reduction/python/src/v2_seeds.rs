// SPDX-License-Identifier: 0BSD

//! Python entry point of the native seed combinations and classes (WP3.1
//! I10, package native-seeds; `rusticol_core::loop_v2::generate::seeds`).
//! `loops/v2/native_seeds.py` encodes the stage input (a JSON table of the
//! distinct contents and the skeletons as little-endian content-id spans)
//! and decodes the output. No Symbolica licence call is made.
use super::*;
use crate::v2_generate::V2Generation;
use rusticol_core::loop_v2::generate::FastMap;
use rusticol_core::loop_v2::generate::networks::Network;
use rusticol_core::loop_v2::generate::seeds::{
    self, FactoredSpec, Kind, LivenessSpec, NetworkSpec, Seed, SeedOutput, SeedSpec, Tensor,
    TensorSpec, Word,
};

/// A primitive: `(kind, letters, input descriptor indices)`.
type RawPrimitive = (&'static str, Vec<i64>, Vec<u32>);
/// A ring need: `(input descriptor indices, forward relabelling)`.
type RawRing = (Vec<u32>, Vec<(i64, i64)>);
/// A new class: `("P", primitive)` or `("G", generic need)`.
type RawClass = (&'static str, Option<RawPrimitive>, u32);
type RawOutput = (Vec<RawClass>, Vec<Vec<u32>>, Vec<Vec<(u32, u32)>>, Vec<u32>);
/// A term of a reduced tensor: `(re, im, N_c power, open strings, traces)`.
type RawTerm = (
    String,
    String,
    i64,
    Vec<(i64, Vec<i64>, i64)>,
    Vec<Vec<i64>>,
);

fn raw_tensor(tensor: &Tensor) -> (i64, Vec<RawTerm>) {
    (
        tensor.normalization,
        tensor
            .terms
            .iter()
            .map(|(value, structure, power)| {
                let (re, im) = value.text();
                (
                    re,
                    im,
                    *power,
                    structure
                        .strings
                        .iter()
                        .map(|s| (s.fundamental, s.adjoint.clone(), s.antifundamental))
                        .collect(),
                    structure.traces.clone(),
                )
            })
            .collect(),
    )
}

/// `[[key, network]]` JSON to networks by key.
fn networks(text: &str) -> PyResult<FastMap<u32, Network>> {
    let specs: Vec<(u32, NetworkSpec)> = serde_json::from_str(text).map_err(error)?;
    specs
        .iter()
        .map(|(key, spec)| Ok((*key, seeds::parse_network(spec).map_err(error)?)))
        .collect()
}

/// `{"factors": [...], "contents": [[key, [factor ids]]]}` JSON to networks
/// by key (`seeds::factored_networks`).
fn factored_networks(text: &str) -> PyResult<FastMap<u32, Network>> {
    let spec: FactoredSpec = serde_json::from_str(text).map_err(error)?;
    seeds::factored_networks(&spec).map_err(error)
}

fn words<const N: usize>(bytes: &[u8]) -> PyResult<Vec<[u8; N]>> {
    if !bytes.len().is_multiple_of(N) {
        return Err(error("seed table: a binary table of partial words"));
    }
    Ok(bytes
        .chunks_exact(N)
        .map(|chunk| chunk.try_into().expect("exact chunks"))
        .collect())
}

/// One seed-table stage (`colour.build_seed_table` or
/// `colour.extend_seed_table` on its skeletons): `analyse`, `needs`, then
/// `complete` (see `loops/v2/native_seeds.py`).
#[pyclass(module = "pyamplicol._rusticol")]
struct SeedBuilder {
    core: seeds::SeedBuilder,
    output: Option<SeedOutput>,
}

impl SeedBuilder {
    /// The raw output of a completed stage (kept for `liveness`).
    fn finish(&mut self, output: SeedOutput) -> PyResult<RawOutput> {
        let classes = output
            .classes
            .iter()
            .map(|seed| {
                Ok(match *seed {
                    Seed::Primitive(p) => ("P", Some(self.raw_primitive(p)?), 0),
                    Seed::Generic(g) => ("G", None, g),
                    Seed::Tensor(_) => unreachable!("new classes are never existing tensors"),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let raw = (
            classes,
            output.orders.clone(),
            output.rows.clone(),
            output.skeleton_rows.clone(),
        );
        self.output = Some(output);
        Ok(raw)
    }

    fn raw_primitive(&self, id: u32) -> PyResult<RawPrimitive> {
        let (kind, word) = self.core.primitive(id);
        Ok(match word {
            Word::Letters(letters) => (kind.name(), letters.clone(), Vec::new()),
            Word::Ring(ring) => (
                kind.name(),
                Vec::new(),
                ring.iter()
                    .map(|&d| self.core.input_descriptor(d).map_err(error))
                    .collect::<PyResult<_>>()?,
            ),
        })
    }
}

#[pymethods]
impl SeedBuilder {
    /// `spec`: the JSON table of `native_seeds.py`; `offsets`: u64 content
    /// offsets of the skeletons (one more than skeletons); `contents`: u32
    /// content ids.
    #[new]
    fn new(py: Python<'_>, spec: &str, offsets: &[u8], contents: &[u8]) -> PyResult<Self> {
        let spec: SeedSpec = serde_json::from_str(spec).map_err(error)?;
        let offsets = words::<8>(offsets)?
            .into_iter()
            .map(|w| usize::try_from(u64::from_le_bytes(w)).map_err(error))
            .collect::<PyResult<Vec<_>>>()?;
        let contents: Vec<u32> = words::<4>(contents)?
            .into_iter()
            .map(u32::from_le_bytes)
            .collect();
        let core = py
            .detach(|| seeds::SeedBuilder::new(spec, offsets, contents))
            .map_err(error)?;
        Ok(Self { core, output: None })
    }

    /// The stage fed in-crate by a retained schedule build of `generation`
    /// (`V2Generation`, handle `handle`): its readings
    /// (`reference_executor.reading_skeletons`) are the stage's skeletons,
    /// and `spec`'s contents are the build's distinct reading contents in
    /// the order `V2Generation.readings` lists them.
    #[staticmethod]
    fn from_generation(
        py: Python<'_>,
        generation: &Bound<'_, V2Generation>,
        handle: usize,
        spec: &str,
    ) -> PyResult<Self> {
        let spec: SeedSpec = serde_json::from_str(spec).map_err(error)?;
        let retained = generation.get().retained(handle)?;
        let core = py
            .detach(|| {
                let readings = retained.readings();
                let (_used, dense) = readings.dense();
                seeds::SeedBuilder::new(spec, readings.offsets, dense)
            })
            .map_err(error)?;
        Ok(Self { core, output: None })
    }

    /// Combinations of every skeleton; the certificate candidates as
    /// `[(kind, skeletons of the smallest size)]`.
    fn analyse(&mut self, py: Python<'_>) -> PyResult<Vec<(&'static str, Vec<usize>)>> {
        let core = &mut self.core;
        let candidates = py.detach(|| core.analyse()).map_err(error)?;
        Ok(candidates
            .into_iter()
            .map(|(kind, list)| (kind.name(), list))
            .collect())
    }

    /// The combination of one skeleton: `[(primitive, coefficient id)]`.
    fn combination(&self, skeleton: usize) -> PyResult<Option<Vec<(RawPrimitive, u32)>>> {
        let Some(terms) = self.core.combination_of(skeleton) else {
            return Ok(None);
        };
        terms
            .iter()
            .map(|&(p, c)| Ok((self.raw_primitive(p)?, c)))
            .collect::<PyResult<Vec<_>>>()
            .map(Some)
    }

    /// `(ring needs, generic skeletons)` once `certified` compressions are
    /// known.
    fn needs(
        &mut self,
        py: Python<'_>,
        certified: Vec<String>,
    ) -> PyResult<(Vec<RawRing>, Vec<usize>)> {
        let kinds = certified
            .iter()
            .map(|name| Kind::from_name(name))
            .collect::<Result<Vec<_>, _>>()
            .map_err(error)?;
        let core = &mut self.core;
        let (rings, generics) = py
            .detach(|| {
                core.needs(&kinds)
                    .map(|(rings, generics)| (rings.to_vec(), generics.to_vec()))
            })
            .map_err(error)?;
        let rings = rings
            .into_iter()
            .map(|need| {
                Ok((
                    need.descriptors
                        .iter()
                        .map(|&d| self.core.input_descriptor(d).map_err(error))
                        .collect::<PyResult<Vec<_>>>()?,
                    need.forward,
                ))
            })
            .collect::<PyResult<_>>()?;
        Ok((rings, generics))
    }

    /// The table rows given the reduced tensors of the needs (JSON lists of
    /// `(inverse sqrt(2) power, terms)`): `(new classes, class orders, rows,
    /// row of each skeleton)`.
    fn complete(&mut self, py: Python<'_>, rings: &str, generics: &str) -> PyResult<RawOutput> {
        let parse = |text: &str| -> PyResult<Vec<seeds::Tensor>> {
            let specs: Vec<TensorSpec> = serde_json::from_str(text).map_err(error)?;
            specs
                .iter()
                .map(seeds::parse_tensor)
                .collect::<Result<_, _>>()
                .map_err(error)
        };
        let (rings, generics) = (parse(rings)?, parse(generics)?);
        let core = &mut self.core;
        let output = py
            .detach(|| core.complete(rings, generics))
            .map_err(error)?;
        self.finish(output)
    }

    /// The stage's content ids the generic needs read (after `needs`).
    fn generic_contents(&self) -> Vec<u32> {
        self.core.generic_contents()
    }

    /// Input indices of the exact descriptors of the ring needs.
    fn ring_exact_descriptors(&self) -> PyResult<Vec<u32>> {
        self.core.ring_exact_descriptors().map_err(error)
    }

    /// Reduce the needs natively (after `needs`): `exact` is a JSON list of
    /// `[input descriptor index, network]`, `contents` the factored networks
    /// of the generic contents (`loops/v2/native_seeds.py`); legs are the
    /// tensors' leg labels.
    fn reduce(
        &mut self,
        py: Python<'_>,
        ring_legs: Vec<i64>,
        seed_legs: Vec<i64>,
        exact: &str,
        contents: &str,
    ) -> PyResult<()> {
        let (exact, contents) = (networks(exact)?, factored_networks(contents)?);
        let core = &mut self.core;
        py.detach(|| core.reduce(&ring_legs, &seed_legs, &exact, &contents))
            .map_err(error)
    }

    /// The reduced tensor of ring need `k` (after `reduce`):
    /// `(normalisation, terms)` (`_reduced_primitive`).
    fn ring_tensor(&self, k: usize) -> PyResult<(i64, Vec<RawTerm>)> {
        let (rings, _) = self
            .core
            .reduced()
            .ok_or_else(|| error("no reduced needs"))?;
        rings
            .get(k)
            .map(raw_tensor)
            .ok_or_else(|| error("no such ring need"))
    }

    /// The generic seed of generic need `g` (after `reduce`).
    fn generic_tensor(&self, g: usize) -> PyResult<(i64, Vec<RawTerm>)> {
        let (_, generics) = self
            .core
            .reduced()
            .ok_or_else(|| error("no reduced needs"))?;
        generics
            .get(g)
            .map(raw_tensor)
            .ok_or_else(|| error("no such generic need"))
    }

    /// `complete` with the natively reduced needs.
    fn complete_reduced(&mut self, py: Python<'_>) -> PyResult<RawOutput> {
        let core = &mut self.core;
        let output = py.detach(|| core.complete_reduced()).map_err(error)?;
        self.finish(output)
    }

    /// Every interned coefficient as `(re, im)` fraction texts, by id.
    fn coefficients(&self) -> Vec<(String, String)> {
        (0..self.core.coefficient_count() as u32)
            .map(|id| self.core.coefficient(id).text())
            .collect()
    }

    /// Every order vector by id, names as the caller's ids.
    fn order_vectors(&self) -> Vec<Vec<(u16, i64)>> {
        self.core.order_vectors()
    }

    /// `Skeleton.orders` of every skeleton, as order ids.
    fn skeleton_orders(&self) -> Vec<u32> {
        self.core.skeleton_orders().to_vec()
    }

    /// `_prune_colour`'s liveness (after `complete`): per class the
    /// `class_live` flag of each of its orders (as `complete` lists them)
    /// and `keep` of every skeleton.
    fn liveness(&mut self, py: Python<'_>, spec: &str) -> PyResult<(Vec<Vec<bool>>, Vec<bool>)> {
        let spec: LivenessSpec = serde_json::from_str(spec).map_err(error)?;
        let output = self
            .output
            .take()
            .ok_or_else(|| error("seed table: liveness needs a completed stage"))?;
        let core = &mut self.core;
        let result = py.detach(|| core.liveness(&output, &spec));
        self.output = Some(output);
        result.map_err(error)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SeedBuilder>()?;
    Ok(())
}
