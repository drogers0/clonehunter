mod compare;
mod html;
mod json;
mod sarif;
mod schema;

pub(crate) use html::write_html;
pub(crate) use json::write_json;
pub(crate) use sarif::write_sarif;

use thiserror::Error;

/// Shared error type for all reporters.
#[derive(Debug, Error)]
pub(crate) enum ReportError {
    #[error("write report: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialize: {0}")]
    Json(#[from] serde_json::Error),
}
