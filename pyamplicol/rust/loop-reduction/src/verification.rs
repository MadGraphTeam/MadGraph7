// SPDX-License-Identifier: 0BSD
//! Cross-precision verification policies of the SDK's C ABI. The removed
//! legacy loop runtime evaluated them; plan v4 SDK runtimes refuse a policy
//! (stability estimates and escalation go through the Python API), so only
//! the policy records and their validation remain.
use crate::runtime::{Batch, ExecutionError, arithmetic_mode};
use rusticol_core::{RusticolError, RusticolResult};
use serde::Serialize;

fn invalid(e: impl ToString) -> RusticolError {
    RusticolError::invalid_argument(e.to_string())
}
fn execution(e: ExecutionError) -> RusticolError {
    RusticolError::evaluation(e.to_string())
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ArithmeticSpec {
    pub arithmetic: String,
    pub precision: Option<u32>,
}
impl ArithmeticSpec {
    pub fn bits(&self) -> RusticolResult<u32> {
        arithmetic_mode(&self.arithmetic, self.precision)
            .map(|(_, bits)| bits)
            .map_err(invalid)
    }
}
#[derive(Clone, Debug)]
pub struct VerificationPolicy {
    pub higher: Vec<ArithmeticSpec>,
    pub relative_tolerance: String,
    pub absolute_tolerance: String,
    pub verify_successful: bool,
}
impl VerificationPolicy {
    pub fn validate(&self, primary: &ArithmeticSpec) -> RusticolResult<()> {
        if self.higher.is_empty() {
            return Err(invalid("verification requires higher arithmetic levels"));
        }
        let mut bits = primary.bits()?;
        for spec in &self.higher {
            let next = spec.bits()?;
            if next <= bits {
                return Err(invalid(
                    "verification levels must strictly increase actual precision bits",
                ));
            }
            bits = next;
        }
        let empty = Batch::typed::<f64>(vec![], vec![], 0, 53).map_err(execution)?;
        empty
            .agreement_rows(&empty, &self.relative_tolerance, &self.absolute_tolerance)
            .map_err(invalid)?;
        Ok(())
    }
}
