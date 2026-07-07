use crate::core::types::{ScanRequest, ScanResult};

use super::{Engine, PipelineError, pipeline::run_pipeline};

pub(crate) struct SemanticEngine;

impl Engine for SemanticEngine {
    fn scan(&self, request: &ScanRequest) -> Result<ScanResult, PipelineError> {
        run_pipeline(&request.paths, &request.config)
    }
}
