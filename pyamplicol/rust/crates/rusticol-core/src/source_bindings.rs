// SPDX-License-Identifier: 0BSD

//! Checked source scalar ownership over saved dependency evaluators. This module
//! supplies no scalar algebra or renormalization prescription: exact defaults,
//! value/diagnostic pairs and scale substitution are the validated saved programs.

use std::collections::BTreeMap;

/// Model parameter overrides by name: (real, imaginary) decimal strings.
pub type SourceOverrides = BTreeMap<String, [String; 2]>;
