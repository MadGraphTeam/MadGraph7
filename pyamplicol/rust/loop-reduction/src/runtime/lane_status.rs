// SPDX-License-Identifier: 0BSD

//! The status texts of refused NinjaRS lanes, with what NinjaRS records about
//! the refusal: the fast engine's test, measured quantity and bound
//! (`ninjars::fast::take_instabilities`), and the generic engine's box-Gram
//! bound (`ninjars::conditioning::take_box_gram_refusals`: predicted loss,
//! precision, margin). Both are taken right after the evaluation, on its
//! thread. The texts keep the plain status first (`Unstable`, `CutFailed {
//! .. }`), so matching on it is unchanged:
//! `Unstable (triangle Gram determinant 0 <= 1.82e-12)`,
//! `Unstable (box mu-expansion conditioning 3.10e-9 > 7.60e-11)`,
//! `CutFailed { cut_index: 3, failure: DegenerateGram } (box-Gram bound:
//! predicted loss 348.1 bits > 189 - 28 bits)`.

use crate::EvaluationStatus;
use ninjars::conditioning::{BoxGramRefusal, take_box_gram_refusals};
use ninjars::fast::{FastStatus, Instability, take_instabilities};

/// The fast engine's records of its last evaluation on this thread, per
/// lane of that call (`lanes` of them; the first record of a lane).
pub(super) fn fast_instabilities(lanes: usize) -> Vec<Option<Instability>> {
    let mut by_lane = vec![None; lanes];
    for (lane, record) in take_instabilities() {
        if let Some(slot) = by_lane.get_mut(lane) {
            slot.get_or_insert(record);
        }
    }
    by_lane
}

/// A fast-engine lane's status text: `Unstable (<record>)` when NinjaRS
/// recorded why, else the status's name.
pub(super) fn fast_status(status: FastStatus, record: Option<&Instability>) -> String {
    match (status, record) {
        (FastStatus::Unstable, Some(record)) => format!("Unstable ({record})"),
        _ => format!("{status:?}"),
    }
}

/// The generic engine's box-Gram refusals of its last evaluation on this
/// thread.
pub(super) fn generic_refusals() -> Vec<BoxGramRefusal> {
    take_box_gram_refusals()
}

/// A generic-engine lane's status text (lane `lane` of its call): the
/// status, and the box-Gram bound's record when it refused the lane's cut.
pub(super) fn generic_status(
    status: &EvaluationStatus,
    lane: usize,
    refusals: &[BoxGramRefusal],
) -> String {
    match refusals
        .iter()
        .find(|refusal| refusal.explains(lane, status))
    {
        Some(refusal) => format!("{status:?} ({refusal})"),
        None => format!("{status:?}"),
    }
}
