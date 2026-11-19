// SPDX-License-Identifier: 0BSD

//! The status texts of refused NinjaRS lanes, with what NinjaRS records about
//! the refusal: the fast engine's test, measured quantity and bound
//! (`ninjars::fast::take_instabilities`), and the generic engine's box-Gram
//! bound (`ninjars::conditioning::take_box_gram_refusals`: predicted loss,
//! precision, margin). Both are taken right after the evaluation, on its
//! thread. The texts keep the plain status first (`Unstable`, `CutFailed {
//! .. }`), so matching on it is unchanged, and the record follows it, found
//! first and then what the test required (NinjaRS's `Display`):
//! `Unstable (box normal: found 3.09e-11, required > 3.26e-8)`,
//! `Unstable (box mu-expansion conditioning: predicted error 3.19e-15,
//! allowed <= 2.36e-15)`,
//! `CutFailed { cut_index: 0, failure: DegenerateGram } (box-Gram bound:
//! predicted loss 295.0 bits, allowed <= 228 bits (256 - 28))`: bits of the
//! carrier's working precision, 256 at 57 requested digits.

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
