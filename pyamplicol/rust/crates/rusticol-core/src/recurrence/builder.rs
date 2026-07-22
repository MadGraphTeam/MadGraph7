// SPDX-License-Identifier: 0BSD

//! Composite authentication and compact recurrence schedule construction.

use super::process::{ProcessSemanticTemplateReference, ValidatedRecurrenceProcessInput};
use super::template::{RecurrenceSemanticTemplateKind, ValidatedRecurrenceTemplateInput};
use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// Process and prepared-model inputs authenticated as one semantic unit.
#[derive(Clone, Debug)]
pub struct AuthenticatedRecurrenceBuilderInput {
    process: ValidatedRecurrenceProcessInput,
    template: ValidatedRecurrenceTemplateInput,
}

impl AuthenticatedRecurrenceBuilderInput {
    pub fn new(
        process: ValidatedRecurrenceProcessInput,
        template: ValidatedRecurrenceTemplateInput,
    ) -> RusticolResult<Self> {
        let process_identity = process.semantic_identity();
        let template_summary = template.summary();
        if process_identity.model_catalog_digest() != template_summary.compiled_model_digest {
            return Err(invalid(format!(
                "recurrence process model-catalog digest {} does not match prepared model {}",
                process_identity.model_catalog_digest(),
                template_summary.compiled_model_digest,
            )));
        }
        if process_identity.prepared_catalog_digest() != template_summary.catalog_digest {
            return Err(invalid(format!(
                "recurrence process prepared-catalog digest {} does not match template catalog {}",
                process_identity.prepared_catalog_digest(),
                template_summary.catalog_digest,
            )));
        }

        let template_index = template.semantic_index()?;
        for reference in process.template_references() {
            authenticate_template_reference(reference, &template_index)?;
        }

        Ok(Self { process, template })
    }

    pub const fn process(&self) -> &ValidatedRecurrenceProcessInput {
        &self.process
    }

    pub const fn template(&self) -> &ValidatedRecurrenceTemplateInput {
        &self.template
    }

    pub fn into_parts(
        self,
    ) -> (
        ValidatedRecurrenceProcessInput,
        ValidatedRecurrenceTemplateInput,
    ) {
        (self.process, self.template)
    }
}

fn authenticate_template_reference(
    reference: &ProcessSemanticTemplateReference,
    template_index: &super::template::RecurrenceTemplateSemanticIndex,
) -> RusticolResult<()> {
    let kind = RecurrenceSemanticTemplateKind::parse(reference.typed_id.kind.as_str())?;
    let Some(identity) = template_index.record(kind, reference.typed_id.template_id) else {
        return Err(invalid(format!(
            "recurrence process references absent prepared {} template {}",
            kind.as_str(),
            reference.typed_id.template_id,
        )));
    };
    if identity.semantic_digest != reference.semantic_digest {
        return Err(invalid(format!(
            "recurrence process {} template {} has semantic digest {}, expected {}",
            kind.as_str(),
            reference.typed_id.template_id,
            reference.semantic_digest,
            identity.semantic_digest,
        )));
    }
    if identity.prepared_kernel_id != reference.prepared_kernel_id {
        return Err(invalid(format!(
            "recurrence process {} template {} has prepared kernel {:?}, expected {:?}",
            kind.as_str(),
            reference.typed_id.template_id,
            reference.prepared_kernel_id,
            identity.prepared_kernel_id,
        )));
    }
    Ok(())
}
