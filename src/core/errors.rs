use thiserror::Error;

/// Configuration errors. Defined here because config is a core concern.
/// Other error types (EmbeddingError, CacheError, ParseError, etc.)
/// are defined in their respective modules.
#[derive(Debug, Error)]
#[allow(dead_code)] // Used starting T3 (config_loader); until then, allow
pub(crate) enum ConfigError {
    #[error("invalid value for {field}: {reason}")]
    InvalidValue { field: String, reason: String },

    #[error("failed to read config: {0}")]
    ReadError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_formats_include_context() {
        let e = ConfigError::InvalidValue {
            field: "threshold".into(),
            reason: "must be in [0,1]".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("threshold"));
        assert!(msg.contains("must be in [0,1]"));
    }
}
