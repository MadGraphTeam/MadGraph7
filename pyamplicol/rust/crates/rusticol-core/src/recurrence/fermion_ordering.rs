// SPDX-License-Identifier: 0BSD

//! Cold construction-time exterior-algebra signs for external fermions.
//!
//! Every binary merge restores the canonical source order of the external
//! fermions of its two parents, and a merge of two fermion currents has one
//! more sign when its oriented inputs are particle--antiparticle.  The product
//! over a tree is its Wick sign, for colored and color-singlet fermions alike.
//!
//! The leading-color contraction multiplies two open-line sectors by the
//! parity of their relative fundamental--antifundamental pairing
//! (`color/contraction_factors.py`).  For gluon exchange that parity is the
//! Wick sign up to a process constant, because every gluon joining two quark
//! lines transposes their color ends.  A color-singlet link (an electroweak
//! boson or the U(1) part of a gluon) leaves the color ends in place, so its
//! diagrams need the Wick sign, not the color parity.  Each closure therefore
//! divides out the color parity of the sector it fills
//! ([`open_line_closure_factor`]), and the U(1) subtraction, whose `-1/N_c`
//! sign the color parity used to supply, carries that sign where its current
//! is created ([`u1_subtraction_factor`]).

use super::ExactComplexRational;
use super::color::{LCColorComponent, LCColorComponentKind};
use super::template::{CurrentOrientation, CurrentStateRow, ParticleStatistics};
use crate::{RusticolError, RusticolResult};

/// Auxiliary kind of the U(1) subtraction current of a color-flow gluon.
pub(super) const U1_SUBTRACTION_AUXILIARY_KIND: &str = "u1-subtraction-color-flow-vector";

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

#[derive(Clone, Debug)]
pub(super) struct FermionOrderingContext {
    source_requires_exterior_sign: Box<[bool]>,
    u1_subtraction_states: Box<[bool]>,
    active: bool,
}

impl FermionOrderingContext {
    /// `u1_subtraction_states` flags, by current-state template ID, the
    /// U(1) subtraction currents of color-flow gluons.
    pub(super) fn new(
        source_requires_exterior_sign: Vec<bool>,
        u1_subtraction_states: Vec<bool>,
    ) -> Self {
        let active = source_requires_exterior_sign
            .iter()
            .copied()
            .any(|value| value);
        Self {
            source_requires_exterior_sign: source_requires_exterior_sign.into_boxed_slice(),
            u1_subtraction_states: u1_subtraction_states.into_boxed_slice(),
            active,
        }
    }

    fn source_requires_exterior_sign(&self, source_slot: u32) -> RusticolResult<bool> {
        self.source_requires_exterior_sign
            .get(source_slot as usize)
            .copied()
            .ok_or_else(|| invalid("fermion-ordering source slot is absent"))
    }
}

fn fermion_color_is_fundamental(color_representation: i32) -> RusticolResult<bool> {
    match color_representation {
        1 => Ok(false),
        3 | -3 => Ok(true),
        value => Err(invalid(format!(
            "external-fermion ordering does not support fermion color representation {value}"
        ))),
    }
}

/// Validate one external source and report whether it is a fermion.
///
/// A colored fermion must be an authenticated pairing endpoint and a
/// color-singlet fermion must not be one; both take part in the exterior sign.
pub(super) fn authenticated_source_requires_exterior_sign(
    is_fermionic: bool,
    is_authenticated_colored_endpoint: bool,
    color_representations: impl IntoIterator<Item = i32>,
) -> RusticolResult<bool> {
    if !is_fermionic {
        if is_authenticated_colored_endpoint {
            return Err(invalid(
                "fermion-ordering endpoint authentication selected a bosonic source",
            ));
        }
        return Ok(false);
    }
    let mut expected_fundamental = None;
    for color_representation in color_representations {
        let fundamental = fermion_color_is_fundamental(color_representation)?;
        if expected_fundamental.is_some_and(|expected| expected != fundamental) {
            return Err(invalid(
                "external-fermion ordering source states have mixed color representations",
            ));
        }
        expected_fundamental = Some(fundamental);
    }
    let fundamental = expected_fundamental.ok_or_else(|| {
        invalid("external-fermion ordering fermion source has no authenticated state")
    })?;
    if is_authenticated_colored_endpoint != fundamental {
        return Err(invalid(if is_authenticated_colored_endpoint {
            "fermion-ordering authenticated colored endpoint has singlet source states"
        } else {
            "fermion-ordering fundamental source lacks authenticated endpoint ownership"
        }));
    }
    Ok(true)
}

fn external_support_sign(
    left_support: &[u32],
    right_support: &[u32],
    context: &FermionOrderingContext,
) -> RusticolResult<i32> {
    let mut odd = false;
    for left in left_support.iter().copied() {
        if !context.source_requires_exterior_sign(left)? {
            continue;
        }
        for right in right_support.iter().copied() {
            if left > right && context.source_requires_exterior_sign(right)? {
                odd = !odd;
            }
        }
    }
    Ok(if odd { -1 } else { 1 })
}

fn input_orientation_sign(left: CurrentStateRow, right: CurrentStateRow) -> RusticolResult<i32> {
    let left_statistics = ParticleStatistics::try_from(left.statistics)?;
    let right_statistics = ParticleStatistics::try_from(right.statistics)?;
    if left_statistics != ParticleStatistics::Fermion
        || right_statistics != ParticleStatistics::Fermion
    {
        return Ok(1);
    }
    fermion_color_is_fundamental(left.color_representation)?;
    fermion_color_is_fundamental(right.color_representation)?;
    let left_orientation = CurrentOrientation::try_from(left.orientation)?;
    let right_orientation = CurrentOrientation::try_from(right.orientation)?;
    match (left_orientation, right_orientation) {
        (CurrentOrientation::Antiparticle, CurrentOrientation::Particle) => Ok(1),
        (CurrentOrientation::Particle, CurrentOrientation::Antiparticle) => Ok(-1),
        (CurrentOrientation::SelfConjugate, _) | (_, CurrentOrientation::SelfConjugate) => Err(
            invalid("external-fermion ordering does not support self-conjugate fermion currents"),
        ),
        _ => Err(invalid(
            "external-fermion ordering requires opposite Dirac orientations",
        )),
    }
}

fn sign_factor(odd: bool) -> RusticolResult<ExactComplexRational> {
    if odd {
        ExactComplexRational::ONE.checked_neg()
    } else {
        Ok(ExactComplexRational::ONE)
    }
}

/// Return the exact sign for one construction-ordered binary current merge.
pub(super) fn fermion_ordering_factor(
    current_states: &[CurrentStateRow],
    parent_state_ids: [u32; 2],
    parent_supports: [&[u32]; 2],
    context: &FermionOrderingContext,
) -> RusticolResult<ExactComplexRational> {
    if !context.active {
        return Ok(ExactComplexRational::ONE);
    }
    let left_state = current_states
        .get(parent_state_ids[0] as usize)
        .copied()
        .ok_or_else(|| invalid("fermion-ordering left current state is absent"))?;
    let right_state = current_states
        .get(parent_state_ids[1] as usize)
        .copied()
        .ok_or_else(|| invalid("fermion-ordering right current state is absent"))?;
    let sign = external_support_sign(parent_supports[0], parent_supports[1], context)?
        * input_orientation_sign(left_state, right_state)?;
    sign_factor(sign < 0)
}

/// Return the `-1` of the U(1) subtraction for a merge creating `result_state_id`.
pub(super) fn u1_subtraction_factor(
    result_state_id: u32,
    context: &FermionOrderingContext,
) -> RusticolResult<ExactComplexRational> {
    if !context.active {
        return Ok(ExactComplexRational::ONE);
    }
    let subtraction = context
        .u1_subtraction_states
        .get(result_state_id as usize)
        .copied()
        .ok_or_else(|| invalid("fermion-ordering result current state is absent"))?;
    sign_factor(subtraction)
}

/// Return the sign that divides the color parity of a closed open-line sector out.
///
/// For `k` open lines the factor is the parity of the fundamental-to-
/// antifundamental pairing, in source-slot order, times `(-1)^(k-1)`, the
/// color parity of a tree whose lines are all joined by gluons.  Closures of
/// gluon-joined lines keep their previous sign.
pub(super) fn open_line_closure_factor(
    closed: &[LCColorComponent],
    context: &FermionOrderingContext,
) -> RusticolResult<ExactComplexRational> {
    if !context.active {
        return Ok(ExactComplexRational::ONE);
    }
    let mut lines = closed
        .iter()
        .filter(|component| component.kind() == LCColorComponentKind::OpenString)
        .map(|component| match component.source_slots() {
            [fundamental, .., antifundamental] => Ok((*fundamental, *antifundamental)),
            _ => Err(invalid(
                "fermion-ordering closure has an open color line without two endpoints",
            )),
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    lines.sort_unstable();
    let mut odd = !lines.is_empty() && lines.len() % 2 == 0;
    for (index, (_, antifundamental)) in lines.iter().enumerate() {
        for (_, later) in &lines[index + 1..] {
            if antifundamental > later {
                odd = !odd;
            }
        }
    }
    sign_factor(odd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(
        id: u32,
        statistics: ParticleStatistics,
        orientation: CurrentOrientation,
        color_representation: i32,
    ) -> CurrentStateRow {
        CurrentStateRow {
            id,
            template_string_id: id,
            particle_id: id as i32,
            anti_particle_id: id as i32,
            species_string_id: id,
            orientation: orientation as u8,
            statistics: statistics as u8,
            color_representation,
            basis_string_id: id,
            tensor_ordering_sequence_id: id,
            dimension: 1,
            chirality: 0,
            lc_color_shape_string_id: id,
            auxiliary_kind_string_id: u32::MAX,
            mass_parameter_id: u32::MAX,
            width_parameter_id: u32::MAX,
            semantic_digest_id: id,
        }
    }

    #[test]
    fn direct_and_exchanged_dirac_pairings_have_opposite_cached_products() {
        // Color-singlet and colored (3, 3bar) fermions follow the same algebra.
        for (anti_color, particle_color) in [(1, 1), (-3, 3)] {
            let anti = state(
                0,
                ParticleStatistics::Fermion,
                CurrentOrientation::Antiparticle,
                anti_color,
            );
            let particle = state(
                1,
                ParticleStatistics::Fermion,
                CurrentOrientation::Particle,
                particle_color,
            );
            let boson = state(
                2,
                ParticleStatistics::Boson,
                CurrentOrientation::SelfConjugate,
                1,
            );
            let states = [anti, particle, boson];
            let context = FermionOrderingContext::new(vec![true; 4], vec![false; 3]);

            let direct_first =
                fermion_ordering_factor(&states, [0, 1], [&[0], &[1]], &context).unwrap();
            let direct_attach =
                fermion_ordering_factor(&states, [2, 0], [&[0, 1], &[2]], &context).unwrap();
            let exchange_first =
                fermion_ordering_factor(&states, [1, 0], [&[1], &[2]], &context).unwrap();
            let exchange_attach =
                fermion_ordering_factor(&states, [0, 2], [&[0], &[1, 2]], &context).unwrap();

            assert_eq!(
                direct_first.checked_mul(direct_attach).unwrap(),
                ExactComplexRational::ONE
            );
            assert_eq!(
                exchange_first.checked_mul(exchange_attach).unwrap(),
                ExactComplexRational::ONE.checked_neg().unwrap()
            );
        }
    }

    #[test]
    fn bosonic_sources_do_not_participate_in_support_parity() {
        let boson = state(
            0,
            ParticleStatistics::Boson,
            CurrentOrientation::SelfConjugate,
            1,
        );
        let context = FermionOrderingContext::new(vec![true, true, true, false], vec![false]);
        let factor = fermion_ordering_factor(&[boson], [0, 0], [&[3], &[0]], &context).unwrap();
        assert_eq!(factor, ExactComplexRational::ONE);
    }

    #[test]
    fn authenticated_source_filter_signs_singlet_and_colored_fermions() {
        assert!(authenticated_source_requires_exterior_sign(true, false, [1, 1]).unwrap());
        assert!(authenticated_source_requires_exterior_sign(true, true, [3, -3]).unwrap());
        assert!(!authenticated_source_requires_exterior_sign(false, false, [8]).unwrap());
    }

    #[test]
    fn authenticated_source_filter_fails_closed_on_mixed_or_unauthenticated_color() {
        for error in [
            authenticated_source_requires_exterior_sign(true, false, [1, 3]).unwrap_err(),
            authenticated_source_requires_exterior_sign(true, false, [3, 3]).unwrap_err(),
            authenticated_source_requires_exterior_sign(true, true, [1, 1]).unwrap_err(),
            authenticated_source_requires_exterior_sign(true, false, [8, 8]).unwrap_err(),
        ] {
            assert!(
                error.to_string().contains("fermion-ordering")
                    || error.to_string().contains("external-fermion ordering")
            );
        }
    }

    #[test]
    fn colored_and_singlet_fermions_are_signed_together() {
        let particle = state(
            0,
            ParticleStatistics::Fermion,
            CurrentOrientation::Particle,
            3,
        );
        let lepton = state(
            1,
            ParticleStatistics::Fermion,
            CurrentOrientation::Antiparticle,
            1,
        );
        let context = FermionOrderingContext::new(vec![true, true, false], vec![false; 3]);
        let negative = ExactComplexRational::ONE.checked_neg().unwrap();
        // A colored particle meeting a singlet antiparticle (a leptoquark-like
        // merge) has the ordinary particle--antiparticle sign.
        assert_eq!(
            fermion_ordering_factor(&[particle, lepton], [0, 1], [&[0], &[1]], &context).unwrap(),
            negative
        );
        // Colored sources count in the support order of any merge.
        let boson = state(
            2,
            ParticleStatistics::Boson,
            CurrentOrientation::SelfConjugate,
            8,
        );
        assert_eq!(
            fermion_ordering_factor(
                &[particle, lepton, boson],
                [0, 2],
                [&[2], &[0, 1]],
                &context
            )
            .unwrap(),
            ExactComplexRational::ONE
        );
        assert_eq!(
            fermion_ordering_factor(&[particle, lepton, boson], [0, 2], [&[1], &[0]], &context)
                .unwrap(),
            negative
        );
    }

    fn line(fundamental: u32, adjoints: &[u32], antifundamental: u32) -> LCColorComponent {
        let mut word = vec![fundamental];
        word.extend_from_slice(adjoints);
        word.push(antifundamental);
        LCColorComponent::new(LCColorComponentKind::OpenString, word).unwrap()
    }

    #[test]
    fn open_line_closure_factor_is_the_pairing_parity_relative_to_gluon_exchange() {
        let context = FermionOrderingContext::new(vec![true; 6], Vec::new());
        let negative = ExactComplexRational::ONE.checked_neg().unwrap();
        let factor = |closed: &[LCColorComponent]| open_line_closure_factor(closed, &context);
        assert_eq!(factor(&[]).unwrap(), ExactComplexRational::ONE);
        assert_eq!(
            factor(&[line(0, &[4], 1)]).unwrap(),
            ExactComplexRational::ONE
        );
        // Two lines: the uncrossed pairing is odd relative to gluon exchange.
        assert_eq!(
            factor(&[line(0, &[], 1), line(2, &[], 3)]).unwrap(),
            negative
        );
        assert_eq!(
            factor(&[line(2, &[], 1), line(0, &[4], 3)]).unwrap(),
            ExactComplexRational::ONE
        );
        // Three lines: even pairings are even, a transposition is odd.
        assert_eq!(
            factor(&[line(0, &[], 1), line(2, &[], 3), line(4, &[], 5)]).unwrap(),
            ExactComplexRational::ONE
        );
        assert_eq!(
            factor(&[line(4, &[], 1), line(2, &[], 3), line(0, &[], 5)]).unwrap(),
            negative
        );
        let error =
            factor(&[LCColorComponent::new(LCColorComponentKind::OpenString, vec![0]).unwrap()])
                .unwrap_err();
        assert!(error.to_string().contains("two endpoints"));
    }

    #[test]
    fn u1_subtraction_currents_carry_their_relative_sign() {
        let context = FermionOrderingContext::new(vec![true, true], vec![false, true]);
        assert_eq!(
            u1_subtraction_factor(0, &context).unwrap(),
            ExactComplexRational::ONE
        );
        assert_eq!(
            u1_subtraction_factor(1, &context).unwrap(),
            ExactComplexRational::ONE.checked_neg().unwrap()
        );
        assert!(u1_subtraction_factor(2, &context).is_err());
    }

    #[test]
    fn same_dirac_orientation_fails_during_construction() {
        for orientation in [
            CurrentOrientation::Particle,
            CurrentOrientation::Antiparticle,
        ] {
            let state = state(0, ParticleStatistics::Fermion, orientation, 1);
            let context = FermionOrderingContext::new(vec![true, true], vec![false]);
            let error =
                fermion_ordering_factor(&[state], [0, 0], [&[0], &[1]], &context).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("requires opposite Dirac orientations")
            );
        }
    }
}
