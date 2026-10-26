// SPDX-License-Identifier: 0BSD

//! Python entry points of the native plan model (WP3.1 package D2,
//! `rusticol_core::loop_v2::plan::binary`): the plan v4 model with its
//! byte-identical JSON writer and its binary columnar member, and the loader
//! from either encoding. The model is fed with the records of a Python
//! `ExportedPlan` (`loops/v2/native_lowering.py`): read from the Python
//! objects themselves (`from_program`, the production route of
//! `ExportedPlan.json()` and `ExportedPlan.binary()`), or from a JSON.
use super::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyInt, PyList, PyString, PyTuple};
use rusticol_core::loop_v2::plan::binary::{
    OMITTED_WHEN_ABSENT, PlanModel, column_names, is_binary, program_digest, read_binary, read_pool,
};
use rusticol_core::loop_v2::plan::{FormPolicy, LoopProgramV4};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use std::time::Instant;

/// A plan v4 model (`PlanModel`).
#[pyclass]
struct V2PlanModel(PlanModel);

#[pymethods]
impl V2PlanModel {
    /// The model of a plan v4 JSON (`ExportedPlan.json()`).
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &[u8]) -> PyResult<Self> {
        py.detach(|| PlanModel::from_json(json))
            .map(Self)
            .map_err(error)
    }

    /// The model of an exported plan's records (`ExportedPlan.program`),
    /// read from the Python objects. Only records the writer writes back as
    /// `json.dumps(program, separators=(",", ":"))` are accepted (`Records`:
    /// the loader's types, every record's keys in the writer's order, every
    /// key the writer always writes present, no `None` under a key the writer
    /// omits when absent), so the native JSON of an accepted program is
    /// `json.dumps`'s byte for byte; anything else fails with the record's
    /// path.
    #[staticmethod]
    fn from_program(program: &Bound<'_, PyAny>) -> PyResult<Self> {
        PlanModel::from_records(Records(program))
            .map(Self)
            .map_err(|failure| error(format!("loop v2 plan records: {failure}")))
    }

    /// The model of a binary plan member.
    #[staticmethod]
    fn from_binary(py: Python<'_>, member: &[u8]) -> PyResult<Self> {
        py.detach(|| PlanModel::from_binary(member))
            .map(Self)
            .map_err(error)
    }

    /// The plan JSON, byte-identical to `ExportedPlan.json()`.
    fn json<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let model = &self.0;
        let json = py.detach(|| model.json());
        PyBytes::new(py, &json)
    }

    /// The binary columnar member.
    fn binary<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let model = &mut self.0;
        let member = py.detach(|| model.binary()).map_err(error)?;
        Ok(PyBytes::new(py, &member))
    }

    /// The bound of the loader's derivations for this plan,
    /// `(peak bytes, allocated bytes, iterations)` (`PlanModel::derive_bound`),
    /// after its refusal of untrusted scalars.
    fn derive_bound(&self, py: Python<'_>) -> PyResult<(u64, u64, u64)> {
        let model = &self.0;
        py.detach(|| model.derive_bound())
            .map(|bound| (bound.memory, bound.allocated, bound.work))
            .map_err(error)
    }

    /// The binary member's column names, in member order (diagnostics and
    /// tests of crafted members).
    #[staticmethod]
    fn columns() -> Vec<&'static str> {
        column_names().to_vec()
    }
}

/// Load a plan (binary member or JSON) as `V2Executor` does, timing the
/// decode into the model and the loader's validation and derivations
/// separately: `(form, decode_s, derive_s, digest)`, with the SHA-256 of
/// the loaded program's tables (`program_digest`) when `digest`.
#[pyfunction]
#[pyo3(signature = (data, *, digest=false))]
fn v2_plan_load(
    py: Python<'_>,
    data: &[u8],
    digest: bool,
) -> PyResult<(&'static str, f64, f64, Option<String>)> {
    py.detach(|| {
        let binary = is_binary(data);
        let start = Instant::now();
        let model = if binary {
            PlanModel::from_binary(data)
        } else {
            PlanModel::from_json(data)
        }?;
        let decoded = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let program: LoopProgramV4 = model.program(FormPolicy::Rule)?;
        let derived = start.elapsed().as_secs_f64();
        let digest = digest.then(|| program_digest(&program));
        Ok((
            if binary { "binary" } else { "json" },
            decoded,
            derived,
            digest,
        ))
    })
    .map_err(|e: rusticol_core::RusticolError| error(e))
}

/// The reader's record of the global thread pool (`read_pool`): `None`
/// before a read needed it, else whether it was up. A diagnostic of the
/// pool-failure fallback's test.
#[pyfunction]
fn v2_plan_read_pool() -> Option<bool> {
    read_pool()
}

/// The SHA-256 of the program the production readers load from a plan
/// (`program_digest`): a binary member through `read_binary`, the one reader
/// of the member, and a JSON through `LoopProgramV4::load`. A diagnostic: the
/// digest formats the program beyond the reader's bound (`program_digest`),
/// so fail-closed probes call `v2_plan_load(data, digest=False)` instead.
#[pyfunction]
fn v2_plan_read(py: Python<'_>, data: &[u8]) -> PyResult<String> {
    py.detach(|| {
        let program = if is_binary(data) {
            read_binary(data)
        } else {
            LoopProgramV4::load(data)
        }?;
        Ok(program_digest(&program))
    })
    .map_err(|e: rusticol_core::RusticolError| error(e))
}

// ------------------------------------------------------- Python records

/// A plan's records as Python objects, read as the loader's file form (a
/// `serde` deserializer over them). Only what the JSON writer writes back as
/// `json.dumps` writes it is accepted: an `int` that is not a `bool` for a
/// number, a `bool` for a flag, a `str`, `None` for an absent option written
/// as `null`, a `list` or `tuple` of the expected length, a `dict` with `str`
/// keys for a record, its keys in the record's field order (the writer's,
/// `plan.rs` file form; unknown keys refused), every key the writer always
/// writes present and no `None` under a key the writer omits when absent
/// ([`OMITTED_WHEN_ABSENT`], the only keys a record may lack). Anything else
/// (a float, a `bool` for a number, a number beyond the field's type, keys
/// out of order, a missing key the writer would write, `null` included, an
/// explicit `None` the writer would drop) fails with the record's path: an
/// accepted program's native JSON is `json.dumps`'s byte for byte
/// (`tests/unit/test_loop_v2_native_lowering.py` checks it on every control
/// and refuses each kind of difference).
struct Records<'a, 'py>(&'a Bound<'py, PyAny>);

/// Why the records do not convert, with the path to the value.
#[derive(Debug)]
struct RecordError(String);

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RecordError {}

impl de::Error for RecordError {
    fn custom<T: std::fmt::Display>(message: T) -> Self {
        Self(message.to_string())
    }
}

impl RecordError {
    /// The error with the key or index it was found under prepended (a path
    /// such as `.families[3].leaves[0].weight: expected a str, ...`).
    fn at(self, place: impl std::fmt::Display) -> Self {
        if self.0.starts_with(['.', '[']) {
            Self(format!("{place}{}", self.0))
        } else {
            Self(format!("{place}: {}", self.0))
        }
    }
}

impl<'py> Records<'_, 'py> {
    fn unexpected(&self, expected: &str) -> RecordError {
        let found = self
            .0
            .get_type()
            .name()
            .map_or_else(|_| "?".to_owned(), |name| name.to_string());
        RecordError(format!("expected {expected}, found a Python {found}"))
    }

    fn integer<'de, V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        let value = self.0;
        if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
            return Err(self.unexpected("an integer"));
        }
        if let Ok(small) = value.extract::<i64>() {
            visitor.visit_i64(small)
        } else if let Ok(large) = value.extract::<u64>() {
            visitor.visit_u64(large)
        } else {
            Err(RecordError("an integer beyond 64 bits".to_owned()))
        }
    }

    fn sequence<'de, V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        let value = self.0;
        if let Ok(list) = value.cast::<PyList>() {
            let mut items = Items {
                items: list.iter(),
                index: 0,
            };
            let out = visitor.visit_seq(&mut items)?;
            items.end()?;
            Ok(out)
        } else if let Ok(tuple) = value.cast::<PyTuple>() {
            let mut items = Items {
                items: tuple.iter(),
                index: 0,
            };
            let out = visitor.visit_seq(&mut items)?;
            items.end()?;
            Ok(out)
        } else {
            Err(self.unexpected("a list"))
        }
    }

    /// A record (its name and `fields` in the writer's order) or a map
    /// (`None`).
    fn record<'de, V: Visitor<'de>>(
        self,
        record: Option<(&'static str, &'static [&'static str])>,
        visitor: V,
    ) -> Result<V::Value, RecordError> {
        let dict = self
            .0
            .cast::<PyDict>()
            .map_err(|_| self.unexpected("a dict"))?;
        visitor.visit_map(Fields {
            items: dict.iter(),
            value: None,
            key: String::new(),
            record,
            next: 0,
            missing: None,
        })
    }
}

impl<'de> de::Deserializer<'de> for Records<'_, '_> {
    type Error = RecordError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        let value = self.0;
        if value.is_none() {
            visitor.visit_unit()
        } else if let Ok(flag) = value.cast::<PyBool>() {
            visitor.visit_bool(flag.is_true())
        } else if value.is_instance_of::<PyInt>() {
            self.integer(visitor)
        } else if value.is_instance_of::<PyString>() {
            self.deserialize_str(visitor)
        } else if value.is_instance_of::<PyDict>() {
            self.record(None, visitor)
        } else {
            self.sequence(visitor)
        }
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        match self.0.cast::<PyBool>() {
            Ok(flag) => visitor.visit_bool(flag.is_true()),
            Err(_) => Err(self.unexpected("a bool")),
        }
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        let text = self
            .0
            .cast::<PyString>()
            .map_err(|_| self.unexpected("a str"))?;
        visitor.visit_str(
            text.to_str()
                .map_err(|_| RecordError("a str with lone surrogates".to_owned()))?,
        )
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.deserialize_str(visitor)
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.deserialize_str(visitor)
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        if self.0.is_none() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.sequence(visitor)
    }

    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, RecordError> {
        self.sequence(visitor)
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, RecordError> {
        self.sequence(visitor)
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.record(None, visitor)
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, RecordError> {
        self.record(Some((name, fields)), visitor)
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        visitor.visit_unit()
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.integer(visitor)
    }

    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.integer(visitor)
    }

    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.integer(visitor)
    }

    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.integer(visitor)
    }

    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.integer(visitor)
    }

    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.integer(visitor)
    }

    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.integer(visitor)
    }

    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, RecordError> {
        self.integer(visitor)
    }

    serde::forward_to_deserialize_any! {
        i128 u128 f32 f64 char bytes byte_buf unit unit_struct newtype_struct enum
    }
}

/// The items of a Python list or tuple, each read as one value.
struct Items<I> {
    items: I,
    index: usize,
}

impl<'py, I: ExactSizeIterator<Item = Bound<'py, PyAny>>> Items<I> {
    /// Every item read: a longer list than its type (an array or a tuple)
    /// is refused, as a JSON reader refuses it.
    fn end(self) -> Result<(), RecordError> {
        match self.items.len() {
            0 => Ok(()),
            more => Err(RecordError(format!(
                "{more} more items than expected after item {}",
                self.index
            ))),
        }
    }
}

impl<'de, 'py, I: ExactSizeIterator<Item = Bound<'py, PyAny>>> SeqAccess<'de> for Items<I> {
    type Error = RecordError;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, RecordError> {
        match self.items.next() {
            None => Ok(None),
            Some(item) => {
                let index = self.index;
                self.index += 1;
                seed.deserialize(Records(&item))
                    .map(Some)
                    .map_err(|error| error.at(format!("[{index}]")))
            }
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.items.len())
    }
}

/// The fields of a Python dict (`str` keys), each read as one value.
struct Fields<'py, I> {
    items: I,
    value: Option<Bound<'py, PyAny>>,
    /// The key of the value read next (the error's path).
    key: String,
    /// A record's name (the file form's type) and its fields in the writer's
    /// order (`None` for a map).
    record: Option<(&'static str, &'static [&'static str])>,
    /// The first of the record's fields the next key may be.
    next: usize,
    /// The first field the writer writes that a later key skipped: missing
    /// (it cannot follow in the writer's order), refused at the record's end
    /// unless a key out of order is refused first.
    missing: Option<&'static str>,
}

/// The writer omits `key` of `record` when its value is absent.
fn omitted(record: &str, key: &str) -> bool {
    OMITTED_WHEN_ABSENT.contains(&(record, key))
}

/// The first of `absent`, fields of `record` the dict lacks, that the writer
/// writes (every key but those it omits when absent, an absent option as
/// `null`), where `json.dumps` of the dict writes nothing.
fn first_written(record: &str, absent: &'static [&'static str]) -> Option<&'static str> {
    absent.iter().copied().find(|&key| !omitted(record, key))
}

fn missing(key: &str) -> RecordError {
    RecordError(format!(
        "missing field {key:?}: the writer writes it (json.dumps would not)"
    ))
}

impl<'de, 'py, I: Iterator<Item = (Bound<'py, PyAny>, Bound<'py, PyAny>)>> MapAccess<'de>
    for Fields<'py, I>
{
    type Error = RecordError;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, RecordError> {
        let Some((key, value)) = self.items.next() else {
            // The record's end: the fields skipped and those after the last
            // key are absent.
            if let Some((record, fields)) = self.record {
                if let Some(key) = self
                    .missing
                    .or_else(|| first_written(record, &fields[self.next..]))
                {
                    return Err(missing(key));
                }
                self.next = fields.len();
            }
            return Ok(None);
        };
        let name = key
            .cast::<PyString>()
            .map_err(|_| Records(&key).unexpected("a str key"))?;
        let name = name
            .to_str()
            .map_err(|_| RecordError("a key with lone surrogates".to_owned()))?;
        if let Some((record, fields)) = self.record {
            // The writer's order: each key after the previous one, the
            // fields between them absent. A key of the record met earlier is
            // out of order; an unknown key is the field visitor's refusal
            // below.
            match fields[self.next..].iter().position(|&field| field == name) {
                Some(skipped) => {
                    let absent = &fields[self.next..self.next + skipped];
                    self.missing = self.missing.or_else(|| first_written(record, absent));
                    self.next += skipped + 1;
                }
                None if fields.contains(&name) => {
                    let after = self.key.as_str();
                    return Err(RecordError(format!(
                        "key {name:?} after {after:?}: out of the writer's key order {fields:?}"
                    )));
                }
                None => {}
            }
            if value.is_none() && omitted(record, name) {
                return Err(RecordError(format!(
                    ".{name}: None where the writer omits the key (json.dumps would \
                     write null)"
                )));
            }
        }
        self.key.clear();
        self.key.push_str(name);
        self.value = Some(value);
        seed.deserialize(Records(&key)).map(Some)
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, RecordError> {
        let value = self
            .value
            .take()
            .ok_or_else(|| RecordError("a value without its key".to_owned()))?;
        seed.deserialize(Records(&value))
            .map_err(|error| error.at(format!(".{}", self.key)))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<V2PlanModel>()?;
    module.add_function(wrap_pyfunction!(v2_plan_load, module)?)?;
    module.add_function(wrap_pyfunction!(v2_plan_read, module)?)?;
    module.add_function(wrap_pyfunction!(v2_plan_read_pool, module)?)
}
