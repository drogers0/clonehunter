use serde::Deserialize;
use std::path::Path;

use crate::core::config::{
    CloneHunterConfig, DeviceName, EmbedderName, EngineName, IndexName, embedder_preset,
};
use crate::core::errors::ConfigError;

// ─── Override structs ────────────────────────────────────────────────────────
// All fields are `Option<T>` so missing TOML keys → `None` → no clobber (DD3).
// These are the vehicle for CLI→config layering in T12.

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct WindowOverride {
    pub window_lines: Option<usize>,
    pub stride_lines: Option<usize>,
    pub min_nonempty: Option<usize>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct ExpansionOverride {
    pub enabled: Option<bool>,
    pub depth: Option<usize>,
    pub max_chars: Option<usize>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct ThresholdsOverride {
    pub func: Option<f64>,
    pub win: Option<f64>,
    pub exp: Option<f64>,
    pub min_window_hits: Option<usize>,
    pub lexical_min_ratio: Option<f64>,
    pub lexical_weight: Option<f64>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct IndexOverride {
    pub name: Option<IndexName>,
    pub top_k: Option<usize>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct CacheOverride {
    pub path: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct EmbedderOverride {
    pub name: Option<EmbedderName>,
    pub model_name: Option<String>,
    pub revision: Option<String>,
    pub max_length: Option<usize>,
    pub batch_size: Option<usize>,
    pub device: Option<DeviceName>,
}

/// Top-level override struct. Deserialized from `clonehunter.toml` or built from CLI args.
#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct ConfigOverride {
    pub engine: Option<EngineName>,
    /// Accepts both a bare string `"**/*.py"` and an array `["**/*.py"]` in TOML,
    /// matching Python's `_coerce_globs` which coerces scalars to singleton lists.
    #[serde(default, deserialize_with = "deserialize_string_or_vec_opt")]
    pub include_globs: Option<Vec<String>>,
    #[serde(default, deserialize_with = "deserialize_string_or_vec_opt")]
    pub exclude_globs: Option<Vec<String>>,
    pub cluster_findings: Option<bool>,
    pub cluster_min_size: Option<usize>,
    pub windows: Option<WindowOverride>,
    pub expansion: Option<ExpansionOverride>,
    pub thresholds: Option<ThresholdsOverride>,
    pub index: Option<IndexOverride>,
    pub cache: Option<CacheOverride>,
    pub embedder: Option<EmbedderOverride>,
}

/// Deserialize a TOML value that may be a bare string or an array of strings.
/// Bare strings are coerced to a singleton `Vec`, matching Python's `_coerce_globs`.
fn deserialize_string_or_vec_opt<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct StringOrVec;

    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Option<Vec<String>>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a string or array of strings")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(Some(vec![v.to_string()]))
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut v = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                v.push(s);
            }
            Ok(Some(v))
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
    }

    deserializer.deserialize_any(StringOrVec)
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Load config with layering: defaults → `clonehunter.toml` → CLI overrides.
///
/// `root` is the directory where `clonehunter.toml` lives (or would live).
/// Pass `None` for `overrides` when no CLI overrides were provided.
pub(crate) fn load_config(
    root: &Path,
    overrides: Option<&ConfigOverride>,
) -> Result<CloneHunterConfig, ConfigError> {
    let mut config = CloneHunterConfig::default();

    let config_file = root.join("clonehunter.toml");
    if config_file.exists() {
        let content = std::fs::read_to_string(&config_file)
            .map_err(|e| ConfigError::ReadError(format!("{}: {}", config_file.display(), e)))?;
        let file_overrides: ConfigOverride = toml::from_str(&content)
            .map_err(|e| ConfigError::ReadError(format!("{}: {}", config_file.display(), e)))?;
        config = apply_overrides(config, &file_overrides);
    }

    if let Some(cli_overrides) = overrides {
        config = apply_overrides(config, cli_overrides);
    }

    validate_config(&config)?;
    Ok(config)
}

/// Apply a set of overrides onto a base config. `None` fields are skipped — they never clobber.
///
/// For nested sections (thresholds, embedder, etc.), merge is field-level: only `Some` fields
/// in the override section replace the corresponding base field. This is the type-safe Rust
/// translation of Python's `if "key" in cfg` / `clean_overrides` pattern (DD3).
pub(crate) fn apply_overrides(
    mut config: CloneHunterConfig,
    ov: &ConfigOverride,
) -> CloneHunterConfig {
    if let Some(engine) = ov.engine {
        config.engine = engine;
    }
    if let Some(ref globs) = ov.include_globs {
        config.include_globs = globs.clone();
    }
    if let Some(ref globs) = ov.exclude_globs {
        config.exclude_globs = globs.clone();
    }
    if let Some(v) = ov.cluster_findings {
        config.cluster_findings = v;
    }
    if let Some(v) = ov.cluster_min_size {
        config.cluster_min_size = v;
    }

    if let Some(ref w) = ov.windows {
        if let Some(v) = w.window_lines {
            config.windows.window_lines = v;
        }
        if let Some(v) = w.stride_lines {
            config.windows.stride_lines = v;
        }
        if let Some(v) = w.min_nonempty {
            config.windows.min_nonempty = v;
        }
    }

    if let Some(ref e) = ov.expansion {
        if let Some(v) = e.enabled {
            config.expansion.enabled = v;
        }
        if let Some(v) = e.depth {
            config.expansion.depth = v;
        }
        if let Some(v) = e.max_chars {
            config.expansion.max_chars = v;
        }
    }

    if let Some(ref t) = ov.thresholds {
        if let Some(v) = t.func {
            config.thresholds.func = v;
        }
        if let Some(v) = t.win {
            config.thresholds.win = v;
        }
        if let Some(v) = t.exp {
            config.thresholds.exp = v;
        }
        if let Some(v) = t.min_window_hits {
            config.thresholds.min_window_hits = v;
        }
        if let Some(v) = t.lexical_min_ratio {
            config.thresholds.lexical_min_ratio = v;
        }
        if let Some(v) = t.lexical_weight {
            config.thresholds.lexical_weight = v;
        }
    }

    if let Some(ref i) = ov.index {
        if let Some(v) = i.name {
            config.index.name = v;
        }
        if let Some(v) = i.top_k {
            config.index.top_k = v;
        }
    }

    if let Some(ref c) = ov.cache {
        if let Some(ref v) = c.path {
            config.cache.path = v.clone();
        }
    }

    if let Some(ref e) = ov.embedder {
        // Rebase from preset ONLY when the name is explicitly changed to a *different* value.
        // This prevents a later layer (e.g., CLI `--device cpu`) from clobbering
        // prior TOML customizations for model_name/revision/max_length/batch_size.
        //
        // Intentional divergence from Python: Python always consults the preset as a fallback
        // even when the name didn't change, clobbering prior-layer customizations. Rust only
        // rebases from the preset on an actual name change — fixing a Python bug silently.
        let name_changed = e.name.is_some() && e.name != Some(config.embedder.name);
        if let Some(name) = e.name {
            config.embedder.name = name;
        }

        if name_changed {
            // Name was explicitly switched — rebase from the new preset,
            // then apply any explicit overrides on top.
            if let Some(preset) = embedder_preset(config.embedder.name) {
                config.embedder.model_name = e
                    .model_name
                    .clone()
                    .unwrap_or_else(|| preset.model_name.into());
                config.embedder.revision =
                    e.revision.clone().unwrap_or_else(|| preset.revision.into());
                config.embedder.max_length = e.max_length.unwrap_or(preset.max_length);
                config.embedder.batch_size = e.batch_size.unwrap_or(preset.batch_size);
            } else {
                // Switched to a name without a preset (e.g., stub) —
                // keep current values, apply only explicit overrides.
                if let Some(ref v) = e.model_name {
                    config.embedder.model_name = v.clone();
                }
                if let Some(ref v) = e.revision {
                    config.embedder.revision = v.clone();
                }
                if let Some(v) = e.max_length {
                    config.embedder.max_length = v;
                }
                if let Some(v) = e.batch_size {
                    config.embedder.batch_size = v;
                }
            }
        } else {
            // Name not changed — only patch explicitly provided fields.
            if let Some(ref v) = e.model_name {
                config.embedder.model_name = v.clone();
            }
            if let Some(ref v) = e.revision {
                config.embedder.revision = v.clone();
            }
            if let Some(v) = e.max_length {
                config.embedder.max_length = v;
            }
            if let Some(v) = e.batch_size {
                config.embedder.batch_size = v;
            }
        }

        if let Some(v) = e.device {
            config.embedder.device = v;
        }
    }

    config
}

/// Validate a resolved config. Enum fields are guaranteed valid by the type system (DD4);
/// this only checks numeric ranges and unit intervals.
pub(crate) fn validate_config(config: &CloneHunterConfig) -> Result<(), ConfigError> {
    fn positive(field: &str, val: usize) -> Result<(), ConfigError> {
        if val == 0 {
            return Err(ConfigError::InvalidValue {
                field: field.into(),
                reason: "must be > 0".into(),
            });
        }
        Ok(())
    }

    fn at_least(field: &str, val: usize, min: usize) -> Result<(), ConfigError> {
        if val < min {
            return Err(ConfigError::InvalidValue {
                field: field.into(),
                reason: format!("must be >= {min}"),
            });
        }
        Ok(())
    }

    fn unit_interval(field: &str, val: f64) -> Result<(), ConfigError> {
        if !(0.0..=1.0).contains(&val) {
            return Err(ConfigError::InvalidValue {
                field: field.into(),
                reason: "must be between 0 and 1".into(),
            });
        }
        Ok(())
    }

    // Embedder
    positive("embedder.batch_size", config.embedder.batch_size)?;
    positive("embedder.max_length", config.embedder.max_length)?;

    // Index
    positive("index.top_k", config.index.top_k)?;

    // Windows
    positive("windows.window_lines", config.windows.window_lines)?;
    positive("windows.stride_lines", config.windows.stride_lines)?;
    // windows.min_nonempty >= 0 is guaranteed by usize; Python checks >= 0

    // Thresholds (unit intervals)
    unit_interval("thresholds.func", config.thresholds.func)?;
    unit_interval("thresholds.win", config.thresholds.win)?;
    unit_interval("thresholds.exp", config.thresholds.exp)?;
    unit_interval(
        "thresholds.lexical_min_ratio",
        config.thresholds.lexical_min_ratio,
    )?;
    unit_interval(
        "thresholds.lexical_weight",
        config.thresholds.lexical_weight,
    )?;
    at_least(
        "thresholds.min_window_hits",
        config.thresholds.min_window_hits,
        1,
    )?;

    // Cluster
    at_least("cluster_min_size", config.cluster_min_size, 1)?;

    // Expansion
    // expansion.depth >= 0 is guaranteed by usize
    positive("expansion.max_chars", config.expansion.max_chars)?;

    Ok(())
}

/// Walk up from `start` to find the nearest directory containing `clonehunter.toml`.
/// Returns `None` if no config file is found (defaults apply).
pub(crate) fn find_config_root(start: &Path) -> Option<std::path::PathBuf> {
    let mut dir = if start.is_dir() {
        start.to_path_buf()
    } else {
        start.parent()?.to_path_buf()
    };
    // Canonicalize to resolve symlinks before walking
    dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    loop {
        if dir.join("clonehunter.toml").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::CODEBERT_REVISION;
    use tempfile::TempDir;

    #[test]
    fn load_config_from_toml_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("clonehunter.toml"),
            r#"
include_globs = ["src/**/*.py"]

[thresholds]
func = 0.8
min_window_hits = 3

[index]
name = "brute"
"#,
        )
        .unwrap();
        let config = load_config(dir.path(), None).unwrap();
        assert_eq!(config.include_globs, vec!["src/**/*.py"]);
        assert!((config.thresholds.func - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.thresholds.min_window_hits, 3);
        assert_eq!(config.index.name, IndexName::Brute);
    }

    #[test]
    fn default_device_is_auto() {
        let dir = TempDir::new().unwrap();
        let config = load_config(dir.path(), None).unwrap();
        assert_eq!(config.embedder.device, DeviceName::Auto);
    }

    #[test]
    fn embedder_preset_codebert_unchanged() {
        // When CLI re-states name=codebert (same as default), preset rebase must NOT occur.
        // Verify with TOML that customizes batch_size — it must survive.
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("clonehunter.toml"),
            "[embedder]\nbatch_size = 64\n",
        )
        .unwrap();
        let ov = ConfigOverride {
            embedder: Some(EmbedderOverride {
                name: Some(EmbedderName::Codebert),
                ..Default::default()
            }),
            ..Default::default()
        };
        let config = load_config(dir.path(), Some(&ov)).unwrap();
        assert_eq!(config.embedder.model_name, "microsoft/codebert-base");
        assert_eq!(config.embedder.max_length, 256);
        assert_eq!(config.embedder.batch_size, 64); // TOML value preserved — no preset clobber
    }

    #[test]
    fn cli_override_none_drop_does_not_clobber() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("clonehunter.toml"),
            "[thresholds]\nfunc = 0.85\n",
        )
        .unwrap();
        // CLI override with only win set; func should keep TOML value
        let ov = ConfigOverride {
            thresholds: Some(ThresholdsOverride {
                win: Some(0.88),
                ..Default::default()
            }),
            ..Default::default()
        };
        let config = load_config(dir.path(), Some(&ov)).unwrap();
        assert!((config.thresholds.func - 0.85).abs() < f64::EPSILON); // from TOML
        assert!((config.thresholds.win - 0.88).abs() < f64::EPSILON); // from CLI
        assert!((config.thresholds.exp - 0.90).abs() < f64::EPSILON); // from default
    }

    #[test]
    fn invalid_enum_in_toml_produces_config_error() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("clonehunter.toml"),
            "engine = \"unknown\"\n",
        )
        .unwrap();
        let err = load_config(dir.path(), None).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("clonehunter.toml"),
            "error should reference config file, got: {msg}"
        );
    }

    #[test]
    fn numeric_validation_rejections() {
        let dir = TempDir::new().unwrap();
        let cases: Vec<(ConfigOverride, &str)> = vec![
            (
                ConfigOverride {
                    embedder: Some(EmbedderOverride {
                        batch_size: Some(0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                "embedder.batch_size",
            ),
            (
                ConfigOverride {
                    index: Some(IndexOverride {
                        top_k: Some(0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                "index.top_k",
            ),
            (
                ConfigOverride {
                    thresholds: Some(ThresholdsOverride {
                        func: Some(1.1),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                "thresholds.func",
            ),
            (
                ConfigOverride {
                    thresholds: Some(ThresholdsOverride {
                        win: Some(-0.1),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                "thresholds.win",
            ),
            (
                ConfigOverride {
                    thresholds: Some(ThresholdsOverride {
                        exp: Some(1.1),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                "thresholds.exp",
            ),
            (
                ConfigOverride {
                    thresholds: Some(ThresholdsOverride {
                        lexical_min_ratio: Some(1.1),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                "thresholds.lexical_min_ratio",
            ),
            (
                ConfigOverride {
                    thresholds: Some(ThresholdsOverride {
                        lexical_weight: Some(-0.1),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                "thresholds.lexical_weight",
            ),
            (
                ConfigOverride {
                    embedder: Some(EmbedderOverride {
                        max_length: Some(0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                "embedder.max_length",
            ),
            (
                ConfigOverride {
                    thresholds: Some(ThresholdsOverride {
                        min_window_hits: Some(0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                "thresholds.min_window_hits",
            ),
            (
                ConfigOverride {
                    cluster_min_size: Some(0),
                    ..Default::default()
                },
                "cluster_min_size",
            ),
            (
                ConfigOverride {
                    expansion: Some(ExpansionOverride {
                        max_chars: Some(0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                "expansion.max_chars",
            ),
        ];
        for (ov, expected_field) in cases {
            let err = load_config(dir.path(), Some(&ov)).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains(expected_field),
                "expected error to mention {expected_field}, got: {msg}"
            );
        }
    }

    #[test]
    fn find_config_root_walks_up() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.path().join("clonehunter.toml"), "").unwrap();

        let found = find_config_root(&nested).unwrap();
        assert_eq!(found, std::fs::canonicalize(dir.path()).unwrap());
    }

    #[test]
    fn find_config_root_returns_none_when_missing() {
        let dir = TempDir::new().unwrap();
        assert!(find_config_root(dir.path()).is_none());
    }

    #[test]
    fn no_toml_file_uses_defaults() {
        let dir = TempDir::new().unwrap();
        let config = load_config(dir.path(), None).unwrap();
        assert_eq!(config.engine, EngineName::Semantic);
        assert_eq!(config.thresholds.func, 0.92);
    }

    #[test]
    fn toml_partial_override_preserves_other_defaults() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("clonehunter.toml"),
            "[thresholds]\nfunc = 0.8\n",
        )
        .unwrap();
        let config = load_config(dir.path(), None).unwrap();
        assert!((config.thresholds.func - 0.8).abs() < f64::EPSILON);
        assert!((config.thresholds.win - 0.90).abs() < f64::EPSILON); // default preserved
        assert!((config.thresholds.exp - 0.90).abs() < f64::EPSILON); // default preserved
    }

    #[test]
    fn three_layer_precedence() {
        // defaults: func=0.92, then TOML: func=0.85, then CLI: func=0.80
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("clonehunter.toml"),
            "[thresholds]\nfunc = 0.85\n",
        )
        .unwrap();
        let ov = ConfigOverride {
            thresholds: Some(ThresholdsOverride {
                func: Some(0.80),
                ..Default::default()
            }),
            ..Default::default()
        };
        let config = load_config(dir.path(), Some(&ov)).unwrap();
        assert!((config.thresholds.func - 0.80).abs() < f64::EPSILON); // CLI wins
    }

    #[test]
    fn scalar_glob_coerced_to_singleton_list() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("clonehunter.toml"),
            "include_globs = \"**/*.rs\"\n",
        )
        .unwrap();
        let config = load_config(dir.path(), None).unwrap();
        assert_eq!(config.include_globs, vec!["**/*.rs"]);
    }

    #[test]
    fn embedder_device_only_override_preserves_toml_model() {
        // Regression: a CLI override that only sets device should NOT rebase
        // from the preset and clobber TOML-customized model_name/batch_size.
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("clonehunter.toml"),
            "[embedder]\nname = \"codebert\"\nbatch_size = 64\n",
        )
        .unwrap();
        let ov = ConfigOverride {
            embedder: Some(EmbedderOverride {
                device: Some(DeviceName::Cpu),
                ..Default::default()
            }),
            ..Default::default()
        };
        let config = load_config(dir.path(), Some(&ov)).unwrap();
        assert_eq!(config.embedder.batch_size, 64); // TOML value preserved
        assert_eq!(config.embedder.device, DeviceName::Cpu); // CLI applied
    }

    #[test]
    fn codebert_revision_is_pinned_sha() {
        let dir = TempDir::new().unwrap();
        let config = load_config(dir.path(), None).unwrap();
        assert_eq!(config.embedder.revision, CODEBERT_REVISION);
        // Confirm it's a SHA (40 hex chars), not "main"
        assert_eq!(config.embedder.revision.len(), 40);
        assert!(
            config
                .embedder
                .revision
                .chars()
                .all(|c| c.is_ascii_hexdigit())
        );
    }
}
