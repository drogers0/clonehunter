pub(crate) mod pipeline;
mod semantic;
mod sonarqube;

pub(crate) use semantic::SemanticEngine;
pub(crate) use sonarqube::SonarQubeEngine;

use thiserror::Error;

use crate::core::config::EngineName;
use crate::core::types::{ScanRequest, ScanResult};

/// Engine trait — the one ABC actually subclassed in the Rust port.
/// Matches Python's `Engine.scan(request: ScanRequest) -> ScanResult`.
pub(crate) trait Engine {
    fn scan(&self, request: &ScanRequest) -> Result<ScanResult, PipelineError>;
}

/// Create the engine for the given engine name.
/// Infallible: `EngineName` is validated by the type system at config-load time.
/// Rust idiomatic alternative to Python's dynamic HashMap registry.
pub(crate) fn get_engine(name: EngineName) -> Box<dyn Engine> {
    match name {
        EngineName::Semantic => Box::new(SemanticEngine),
        EngineName::Sonarqube => Box::new(SonarQubeEngine),
    }
}

/// All errors that can terminate a scan run.
#[derive(Debug, Error)]
pub(crate) enum PipelineError {
    #[error("file collection: {0}")]
    Fs(#[from] crate::io::fs::FsError),
    #[error("embedding: {0}")]
    Embedding(#[from] crate::embedding::EmbeddingError),
    #[error("sonarqube: {0}")]
    Sonar(#[from] sonarqube::SonarError),
    #[error("config serialization: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::EngineName;

    #[test]
    fn test_get_engine_semantic() {
        let engine = get_engine(EngineName::Semantic);
        let _ = engine; // constructing does not panic
    }

    #[test]
    fn test_get_engine_sonarqube() {
        let engine = get_engine(EngineName::Sonarqube);
        let _ = engine;
    }
}
