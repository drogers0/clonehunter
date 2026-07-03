/// (name, includes, excludes) — matches Python's REPO_TYPE_PRESETS exactly.
pub(crate) const REPO_TYPE_PRESETS: &[(&str, &[&str], &[&str])] = &[
    (
        "cpp",
        &[
            "**/*.c", "**/*.cc", "**/*.cpp", "**/*.cxx", "**/*.h", "**/*.hh", "**/*.hpp",
            "**/*.hxx",
        ],
        &[
            "**/build/**",
            "**/out/**",
            "**/bin/**",
            "**/obj/**",
            "**/cmake-build-*/**",
        ],
    ),
    (
        "dotnet",
        &["**/*.cs", "**/*.vb", "**/*.fs"],
        &["**/bin/**", "**/obj/**", "**/packages/**", "**/.vs/**"],
    ),
    (
        "go",
        &["**/*.go"],
        &["**/vendor/**", "**/bin/**", "**/dist/**", "**/.git/**"],
    ),
    (
        "java",
        &["**/*.java"],
        &["**/target/**", "**/build/**", "**/.gradle/**", "**/out/**"],
    ),
    (
        "kotlin",
        &["**/*.kt", "**/*.kts"],
        &["**/build/**", "**/.gradle/**", "**/out/**"],
    ),
    ("monorepo", &[], &[]),
    ("none", &[], &[]),
    (
        "node",
        &["**/*.js", "**/*.mjs", "**/*.cjs", "**/*.ts"],
        &[
            "**/node_modules/**",
            "**/dist/**",
            "**/build/**",
            "**/.next/**",
            "**/.turbo/**",
            "**/coverage/**",
        ],
    ),
    (
        "php",
        &["**/*.php"],
        &[
            "**/vendor/**",
            "**/node_modules/**",
            "**/storage/**",
            "**/bootstrap/cache/**",
        ],
    ),
    (
        "python",
        &["**/*.py"],
        &[
            "**/.venv/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/site-packages/**",
        ],
    ),
    (
        "react",
        &["**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx"],
        &[
            "**/node_modules/**",
            "**/.next/**",
            "**/dist/**",
            "**/build/**",
            "**/coverage/**",
        ],
    ),
    (
        "ruby",
        &["**/*.rb", "**/*.rake"],
        &["**/vendor/**", "**/tmp/**", "**/log/**", "**/coverage/**"],
    ),
    ("rust", &["**/*.rs"], &["**/target/**"]),
    (
        "swift",
        &["**/*.swift"],
        &["**/.build/**", "**/DerivedData/**", "**/build/**"],
    ),
];

/// Validate a `--repotype` value against the preset list.
pub(crate) fn validate_repotype(
    s: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync + 'static>> {
    if REPO_TYPE_PRESETS.iter().any(|(k, _, _)| *k == s) {
        Ok(s.to_string())
    } else {
        let valid: Vec<&str> = REPO_TYPE_PRESETS.iter().map(|(k, _, _)| *k).collect();
        Err(format!("unknown repotype '{s}'. Valid values: {}", valid.join(", ")).into())
    }
}

/// Resolve the effective repotype list from the CLI argument.
///
/// - Empty input → `["monorepo"]` (bare scan defaults to all languages)
/// - Non-empty → filter out `"none"` entries and return (even if result is empty)
pub(crate) fn effective_repotypes(repotypes: &[String]) -> Vec<String> {
    if repotypes.is_empty() {
        return vec!["monorepo".to_string()];
    }
    repotypes
        .iter()
        .filter(|r| r.as_str() != "none")
        .cloned()
        .collect()
}

/// Expand a list of repotype names into their combined include/exclude glob lists.
///
/// "monorepo" expands to the union of ALL presets (except "monorepo" and "none" which are
/// empty and would contribute nothing; Python only skips "monorepo" itself but "none" is empty).
pub(crate) fn resolve_repotype_globs(repotypes: &[String]) -> (Vec<String>, Vec<String>) {
    let mut include: Vec<String> = Vec::new();
    let mut exclude: Vec<String> = Vec::new();
    for repotype in repotypes {
        if repotype == "monorepo" {
            for (key, preset_inc, preset_exc) in REPO_TYPE_PRESETS {
                if *key == "monorepo" {
                    continue;
                }
                include.extend(preset_inc.iter().map(|s| s.to_string()));
                exclude.extend(preset_exc.iter().map(|s| s.to_string()));
            }
        } else if let Some((_, preset_inc, preset_exc)) = REPO_TYPE_PRESETS
            .iter()
            .find(|(k, _, _)| *k == repotype.as_str())
        {
            include.extend(preset_inc.iter().map(|s| s.to_string()));
            exclude.extend(preset_exc.iter().map(|s| s.to_string()));
        }
    }
    (dedupe(&include), dedupe(&exclude))
}

/// Merge two glob lists (base + CLI), resolving conflicts in favour of the CLI layer.
///
/// - include = dedupe(base_inc + cli_inc)
/// - exclude = dedupe(base_exc + cli_exc)
/// - CLI includes remove matching entries from exclude
/// - CLI excludes remove matching entries from include
pub(crate) fn merge_globs(
    base_inc: &[String],
    base_exc: &[String],
    cli_inc: &[String],
    cli_exc: &[String],
) -> (Vec<String>, Vec<String>) {
    let mut include = dedupe(&[base_inc, cli_inc].concat());
    let mut exclude = dedupe(&[base_exc, cli_exc].concat());

    for pattern in cli_inc {
        exclude.retain(|e| e != pattern);
    }
    for pattern in cli_exc {
        include.retain(|i| i != pattern);
    }
    (include, exclude)
}

fn dedupe(values: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    values
        .iter()
        .filter(|v| seen.insert(v.as_str()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_effective_repotypes_empty_defaults_monorepo() {
        let result = effective_repotypes(&[]);
        assert_eq!(result, vec!["monorepo"]);
    }

    #[test]
    fn test_effective_repotypes_filters_none() {
        let result = effective_repotypes(&s(&["python", "none"]));
        assert_eq!(result, vec!["python"]);
    }

    #[test]
    fn test_effective_repotypes_all_none_returns_empty() {
        let result = effective_repotypes(&s(&["none"]));
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_repotype_python() {
        let (inc, exc) = resolve_repotype_globs(&s(&["python"]));
        assert!(inc.contains(&"**/*.py".to_string()));
        assert!(exc.iter().any(|e| e.contains(".venv")));
    }

    #[test]
    fn test_resolve_repotype_monorepo_is_union() {
        let (inc, _) = resolve_repotype_globs(&s(&["monorepo"]));
        // Monorepo union should contain globs from multiple presets
        assert!(
            inc.contains(&"**/*.py".to_string()),
            "should have python globs"
        );
        assert!(
            inc.contains(&"**/*.rs".to_string()),
            "should have rust globs"
        );
        assert!(inc.contains(&"**/*.go".to_string()), "should have go globs");
    }

    #[test]
    fn test_resolve_repotype_dedupe() {
        // Requesting python twice should not duplicate globs
        let (inc, _) = resolve_repotype_globs(&s(&["python", "python"]));
        let py_count = inc.iter().filter(|g| *g == "**/*.py").count();
        assert_eq!(py_count, 1, "duplicate globs must be collapsed");
    }

    #[test]
    fn test_merge_globs_additive() {
        let (inc, _) = merge_globs(&s(&["**/*.py"]), &s(&[]), &s(&["**/*.rs"]), &s(&[]));
        assert!(inc.contains(&"**/*.py".to_string()));
        assert!(inc.contains(&"**/*.rs".to_string()));
    }

    #[test]
    fn test_merge_globs_conflict_cli_wins() {
        // CLI includes "*.test.py" which is in base exclude → remove from exclude
        let (inc, exc) = merge_globs(
            &s(&["**/*.py"]),
            &s(&["**/*.test.py"]),
            &s(&["**/*.test.py"]),
            &s(&[]),
        );
        assert!(inc.contains(&"**/*.test.py".to_string()));
        assert!(
            !exc.contains(&"**/*.test.py".to_string()),
            "CLI include must remove from exclude"
        );
    }

    #[test]
    fn test_merge_globs_dedupe() {
        let (inc, _) = merge_globs(&s(&["**/*.py"]), &s(&[]), &s(&["**/*.py"]), &s(&[]));
        let py_count = inc.iter().filter(|g| *g == "**/*.py").count();
        assert_eq!(py_count, 1, "duplicate globs must be collapsed");
    }

    #[test]
    fn test_validate_repotype_valid() {
        assert!(validate_repotype("python").is_ok());
        assert!(validate_repotype("monorepo").is_ok());
        assert!(validate_repotype("none").is_ok());
    }

    #[test]
    fn test_validate_repotype_invalid() {
        assert!(validate_repotype("unknown_lang").is_err());
    }
}
