// SPDX-License-Identifier: 0BSD
//! One shared exact/rectangular dual-Born operation for prepared and saved calls.
use super::*;
use crate::C64;
use rusticol_core::loop_runtime::RectangularDualWeights;
use rusticol_core::recurrence::{SymmetricGroupFftPlan, SymmetricGroupFftWorkspace};

struct FourierWeights {
    descriptor: RectangularDualWeights,
    plan: SymmetricGroupFftPlan,
    workspace: SymmetricGroupFftWorkspace,
    kernels: Vec<Vec<C64>>,
    residuals: Vec<C64>,
    output_aliases: Vec<(usize, usize)>,
    born_width: usize,
}

pub struct DualBornWeights {
    exact: CoefficientEvaluator,
    fourier: Option<FourierWeights>,
    has_rectangle: bool,
}

impl DualBornWeights {
    pub fn new(
        state: &[u8],
        rectangle: Option<&RectangularDualWeights>,
        constants_state: Option<&[u8]>,
        columns: &[[usize; 3]],
        arithmetic: &str,
        precision: Option<u32>,
    ) -> ExecutionResult<Self> {
        let exact = CoefficientEvaluator::new(state, arithmetic, precision)?;
        let saved = exact_state(state)?;
        if saved.get_output_len() != columns.len() {
            return Err(error("dual Born output columns differ from exact state"));
        }
        let fourier = match (rectangle, constants_state) {
            (None, None) => None,
            (Some(descriptor), Some(constants)) => {
                descriptor
                    .validate(saved.get_input_len(), columns)
                    .map_err(error)?;
                let values = exact_state(constants)?;
                if values.get_input_len() != 0
                    || values.get_output_len() != descriptor.constant_count().map_err(error)?
                {
                    return Err(error("rectangular constant state dimensions differ"));
                }
                if arithmetic == "double" {
                    Some(FourierWeights::new(
                        descriptor,
                        constants,
                        columns,
                        saved.get_input_len(),
                    )?)
                } else {
                    None
                }
            }
            _ => {
                return Err(error(
                    "rectangular descriptor and exact constants must be supplied together",
                ));
            }
        };
        Ok(Self {
            exact,
            fourier,
            has_rectangle: rectangle.is_some(),
        })
    }

    pub fn algorithm(&self) -> &'static str {
        if self.fourier.is_some() {
            "symmetric-group-fft"
        } else {
            "direct"
        }
    }

    pub fn fallback_reason(&self) -> Option<&'static str> {
        (self.has_rectangle && self.fourier.is_none())
            .then_some("exact higher-precision direct contraction")
    }

    pub fn precision_bits(&self) -> u32 {
        self.exact.precision_bits()
    }

    pub fn evaluate_batch(&mut self, input: &Batch) -> ExecutionResult<Batch> {
        match &mut self.fourier {
            Some(weights) => weights.evaluate(input),
            None => self.exact.evaluate_batch(input),
        }
    }
}

impl FourierWeights {
    fn new(
        descriptor: &RectangularDualWeights,
        constants: &[u8],
        columns: &[[usize; 3]],
        born_width: usize,
    ) -> ExecutionResult<Self> {
        let plan = SymmetricGroupFftPlan::new(descriptor.degree).map_err(error)?;
        let mut workspace = plan.workspace(1).map_err(error)?;
        let mut evaluator = CoefficientEvaluator::new(constants, "double", None)?;
        let input = Batch::typed::<f64>(vec![], vec![0], 0, 53)?;
        let evaluated = evaluator.evaluate_batch(&input)?;
        let Data::Double(values) = evaluated.data() else {
            unreachable!()
        };
        if !evaluated.nonfinite_rows().is_empty() {
            return Err(error("nonfinite exact rectangular colour constants"));
        }
        let group = plan.order();
        let kernel_count = shape(descriptor.kernels.len(), group)?;
        if !values[..kernel_count]
            .iter()
            .any(|v| v.re != 0.0 || v.im != 0.0)
        {
            return Err(error("rectangular FFT has no nonzero regular kernel"));
        }
        let mut kernels = Vec::with_capacity(descriptor.kernels.len());
        for values in values[..kernel_count].chunks_exact(group) {
            let reversed = (0..group)
                .map(|g| {
                    let v = &values[plan.inverse_lexicographic_index(g).map_err(error)?];
                    Ok((v.re, v.im))
                })
                .collect::<ExecutionResult<Vec<_>>>()?;
            let mut transformed = vec![(0., 0.); group];
            plan.forward(&reversed, &mut transformed, &mut workspace)
                .map_err(error)?;
            kernels.push(
                transformed
                    .into_iter()
                    .map(|(re, im)| C64::new(re, im))
                    .collect(),
            );
        }
        let residuals = values[kernel_count..]
            .iter()
            .map(|v| C64::new(v.re, v.im))
            .collect();
        let term_count: usize = descriptor
            .loop_aliases
            .iter()
            .map(|a| a.columns.len())
            .sum();
        let mut term_aliases = vec![0; term_count];
        for (alias, item) in descriptor.loop_aliases.iter().enumerate() {
            for &term in &item.columns {
                term_aliases[term] = alias;
            }
        }
        let output_aliases = columns
            .iter()
            .map(|&[direction, sector, term]| {
                let bank = descriptor
                    .banks
                    .iter()
                    .position(|b| *b == [direction, sector])
                    .expect("validated exact bank domain");
                (bank, term_aliases[term])
            })
            .collect();
        Ok(Self {
            descriptor: descriptor.clone(),
            plan,
            workspace,
            kernels,
            residuals,
            output_aliases,
            born_width,
        })
    }

    fn evaluate(&mut self, input: &Batch) -> ExecutionResult<Batch> {
        let Data::Double(values) = input.data() else {
            return Err(error("rectangular Fourier input requires binary64"));
        };
        if input.precision_bits() != 53 || input.columns() != self.born_width {
            return Err(error(
                "rectangular Fourier input dimensions/arithmetic differ",
            ));
        }
        let lanes = input.rows();
        let output_count = self.output_aliases.len();
        if lanes == 0 {
            return Batch::typed::<f64>(vec![], vec![], output_count, 53);
        }
        if self.workspace.lane_capacity() < lanes {
            self.workspace = self.plan.workspace(lanes).map_err(error)?;
        }
        let descriptor = &self.descriptor;
        let group = self.plan.order();
        let mut born = vec![C64::new(0., 0.); shape(descriptor.born_aliases.len(), lanes)?];
        for (alias, entry) in descriptor.born_aliases.iter().enumerate() {
            for &column in &entry.columns {
                for row in 0..lanes {
                    let v = &values[row * self.born_width + column];
                    born[alias * lanes + row] += C64::new(v.re, v.im);
                }
            }
        }
        let transform_size = shape(group, lanes)?;
        let mut born_fourier = Vec::with_capacity(descriptor.born_orbits.len());
        for orbit in &descriptor.born_orbits {
            let ordered = orbit
                .iter()
                .flat_map(|&alias| {
                    born[alias * lanes..(alias + 1) * lanes]
                        .iter()
                        .map(|v| (v.re, v.im))
                })
                .collect::<Vec<_>>();
            let mut output = vec![(0., 0.); transform_size];
            self.plan
                .forward_lanes(lanes, &ordered, &mut output, &mut self.workspace)
                .map_err(error)?;
            born_fourier.push(output);
        }
        let aliases = descriptor.loop_aliases.len();
        let mut weights =
            vec![C64::new(0., 0.); shape(shape(descriptor.banks.len(), aliases)?, lanes)?];
        for bank in 0..descriptor.banks.len() {
            for (loop_orbit, orbit) in descriptor.loop_orbits.iter().enumerate() {
                let mut weighted = vec![(0., 0.); transform_size];
                for (index, &[kernel_bank, born_orbit, kernel_loop]) in
                    descriptor.kernels.iter().enumerate()
                {
                    if kernel_bank != bank || kernel_loop != loop_orbit {
                        continue;
                    }
                    for block in self.plan.blocks() {
                        let d = block.dimension();
                        let offset = block.coefficient_offset();
                        for column in 0..d {
                            for row in 0..d {
                                for inner in 0..d {
                                    let kernel =
                                        self.kernels[index][offset + inner * d + column].conj();
                                    for lane in 0..lanes {
                                        let b = born_fourier[born_orbit]
                                            [(offset + inner * d + row) * lanes + lane];
                                        let value = C64::new(b.0, b.1) * kernel;
                                        let target = &mut weighted
                                            [(offset + column * d + row) * lanes + lane];
                                        target.0 += value.re;
                                        target.1 += value.im;
                                    }
                                }
                            }
                        }
                    }
                }
                let mut output = vec![(0., 0.); transform_size];
                self.plan
                    .inverse_lanes(lanes, &weighted, &mut output, &mut self.workspace)
                    .map_err(error)?;
                for (g, &alias) in orbit.iter().enumerate() {
                    for lane in 0..lanes {
                        let v = output[g * lanes + lane];
                        weights[(bank * aliases + alias) * lanes + lane] = C64::new(v.0, -v.1);
                    }
                }
            }
        }
        for (index, &[bank, i, j]) in descriptor.residuals.iter().enumerate() {
            for lane in 0..lanes {
                weights[(bank * aliases + j) * lanes + lane] +=
                    born[i * lanes + lane].conj() * self.residuals[index];
            }
        }
        let mut output = Vec::with_capacity(shape(lanes, output_count)?);
        for lane in 0..lanes {
            for &(bank, alias) in &self.output_aliases {
                let v = weights[(bank * aliases + alias) * lanes + lane];
                output.push(Complex::new(v.re, v.im));
            }
        }
        Batch::typed(output, input.ids().to_vec(), output_count, 53)
    }
}
