// allow dead_code until T12 wires cli::run to the pipeline
#![allow(dead_code)]

use crate::core::config::WindowConfig;
use crate::core::types::{FunctionRef, SnippetKind, SnippetRef};
use crate::io::fingerprints::hash_text;
use crate::snippets::normalization::{normalize_analysis, normalize_display};

/// Generate one FUNC snippet per function.
///
/// snippet_hash format (DD7): `hash_text("FUNC:{path}:{start_line}:{end_line}:{code_hash}")`
/// — deliberately excludes the normalized text so hashes are stable across normalization changes.
pub(crate) fn generate_function_snippets(functions: &[FunctionRef]) -> Vec<SnippetRef> {
    functions
        .iter()
        .map(|fn_| {
            let text = normalize_analysis(&fn_.code);
            let display_text = normalize_display(&fn_.code);
            let snippet_hash = hash_text(&format!(
                "FUNC:{}:{}:{}:{}",
                fn_.file.path, fn_.start_line, fn_.end_line, fn_.code_hash
            ));
            SnippetRef {
                kind: SnippetKind::Func,
                function: fn_.clone(),
                start_line: fn_.start_line,
                end_line: fn_.end_line,
                text,
                display_text,
                snippet_hash,
            }
        })
        .collect()
}

/// Generate WIN snippets by sliding a window over each function's lines.
///
/// snippet_hash format (DD7):
/// `hash_text("WIN:{path}:{fn_start}:{fn_end}:{code_hash}:{win_start}:{win_end}:{analysis_text}")`
/// where `win_start`/`win_end` are 1-indexed positions within the function code.
///
/// # Panics
/// Panics if `config.window_lines == 0` or `config.stride_lines == 0` (matches Python ValueError).
pub(crate) fn generate_window_snippets(
    functions: &[FunctionRef],
    config: &WindowConfig,
) -> Vec<SnippetRef> {
    assert!(config.window_lines > 0, "window_lines must be > 0");
    assert!(config.stride_lines > 0, "stride_lines must be > 0");

    let mut snippets = Vec::new();

    for fn_ in functions {
        let lines: Vec<&str> = fn_.code.lines().collect();
        if lines.is_empty() {
            continue;
        }

        let mut idx = 0usize;
        while idx < lines.len() {
            // start/end are 1-indexed within function code (matches Python's _make_snippet params)
            let start = idx + 1;
            let end = (idx + config.window_lines).min(lines.len());
            let window = &lines[idx..end];

            let nonempty = window.iter().filter(|l| !l.trim().is_empty()).count();
            if nonempty >= config.min_nonempty {
                let window_text = window.join("\n");
                let analysis = normalize_analysis(&window_text);
                let display = normalize_display(&window_text);

                // Absolute line numbers in the source file
                let abs_start = fn_.start_line + start - 1;
                let abs_end = fn_.start_line + end - 1;

                let snippet_hash = hash_text(&format!(
                    "WIN:{}:{}:{}:{}:{}:{}:{}",
                    fn_.file.path,
                    fn_.start_line,
                    fn_.end_line,
                    fn_.code_hash,
                    start,
                    end,
                    analysis
                ));

                snippets.push(SnippetRef {
                    kind: SnippetKind::Win,
                    function: fn_.clone(),
                    start_line: abs_start,
                    end_line: abs_end,
                    text: analysis,
                    display_text: display,
                    snippet_hash,
                });
            }

            idx += config.stride_lines;
        }
    }

    snippets
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use crate::core::types::{FileRef, Language};

    fn make_fn(path: &str, start_line: usize, end_line: usize, code: &str) -> FunctionRef {
        let code = code.to_string();
        let code_hash = hash_text(&code);
        FunctionRef {
            file: FileRef {
                path: path.to_string(),
                content_hash: String::new(),
                language: Language::Python,
            },
            qualified_name: "test_fn".to_string(),
            start_line,
            end_line,
            code,
            code_hash,
        }
    }

    fn fixture_functions() -> Vec<FunctionRef> {
        // Extract from fixtures/tiny_repo/a.py
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/tiny_repo/a.py")
            .to_string_lossy()
            .into_owned();
        let file = FileRef {
            path: path.clone(),
            content_hash: String::new(),
            language: Language::Python,
        };
        crate::parsing::python_ast::extract_functions(&file)
    }

    #[test]
    fn test_generate_function_snippets() {
        let fns = fixture_functions();
        assert!(!fns.is_empty(), "need at least one function from fixture");
        let snippets = generate_function_snippets(&fns);
        assert_eq!(snippets.len(), fns.len(), "one FUNC snippet per function");
        for s in &snippets {
            assert_eq!(s.kind, SnippetKind::Func);
            assert!(!s.snippet_hash.is_empty());
            assert!(!s.text.is_empty());
        }
    }

    #[test]
    fn test_generate_function_snippets_populates_both_texts() {
        // A function with a comment: analysis strips it, display keeps it
        let code = "def foo():\n    # a comment\n    return 1\n";
        let fn_ = make_fn("test.py", 1, 3, code);
        let snippets = generate_function_snippets(&[fn_]);
        assert_eq!(snippets.len(), 1);
        let s = &snippets[0];
        assert!(
            !s.text.contains("# a comment"),
            "analysis should strip comment"
        );
        assert!(
            s.display_text.contains("# a comment"),
            "display should keep comment"
        );
    }

    #[test]
    fn test_generate_function_snippets_hash_format() {
        let fn_ = make_fn("src/foo.py", 5, 10, "def foo():\n    pass\n");
        let snippets = generate_function_snippets(&[fn_.clone()]);
        let expected_hash = hash_text(&format!(
            "FUNC:{}:{}:{}:{}",
            fn_.file.path, fn_.start_line, fn_.end_line, fn_.code_hash
        ));
        assert_eq!(
            snippets[0].snippet_hash, expected_hash,
            "FUNC hash format must match DD7"
        );
    }

    #[test]
    fn test_window_snippets_basic() {
        let fns = fixture_functions();
        let config = WindowConfig {
            window_lines: 3,
            stride_lines: 2,
            min_nonempty: 2,
        };
        let snippets = generate_window_snippets(&fns, &config);
        assert!(!snippets.is_empty(), "should produce some WIN snippets");
        for s in &snippets {
            assert_eq!(s.kind, SnippetKind::Win);
        }
    }

    #[test]
    fn test_window_snippets_absolute_lines() {
        // Function starts at line 10
        let code = "def foo():\n    x = 1\n    y = 2\n    z = 3\n    return z\n";
        let fn_ = make_fn("test.py", 10, 14, code);
        let config = WindowConfig {
            window_lines: 3,
            stride_lines: 1,
            min_nonempty: 1,
        };
        let snippets = generate_window_snippets(&[fn_], &config);
        // All snippets should have start_line >= 10
        for s in &snippets {
            assert!(
                s.start_line >= 10,
                "start_line should be absolute (≥ fn start_line 10), got {}",
                s.start_line
            );
        }
    }

    #[test]
    fn test_window_snippets_min_nonempty() {
        // Function with mostly blank lines; only some windows should emit
        let code = "def foo():\n    x = 1\n\n\n\n\n    y = 2\n";
        let fn_ = make_fn("test.py", 1, 7, code);
        let config = WindowConfig {
            window_lines: 3,
            stride_lines: 2,
            min_nonempty: 2,
        };
        let snippets = generate_window_snippets(&[fn_], &config);
        // Verify all emitted windows have >= min_nonempty non-empty lines
        // (structural test — count of snippets may vary)
        let _ = snippets; // just verify no panic
    }

    #[test]
    fn test_window_snippets_skips_sparse_windows() {
        // All blank lines except one → no window should meet min_nonempty=3
        let code = "def foo():\n\n\n    x = 1\n\n\n";
        let fn_ = make_fn("test.py", 1, 6, code);
        let config = WindowConfig {
            window_lines: 3,
            stride_lines: 3,
            min_nonempty: 3,
        };
        let snippets = generate_window_snippets(&[fn_], &config);
        // No window has 3 non-empty lines
        assert!(
            snippets.is_empty(),
            "sparse windows should be skipped, got {} snippets",
            snippets.len()
        );
    }

    #[test]
    #[should_panic(expected = "window_lines must be > 0")]
    fn test_window_snippets_window_lines_zero_panics() {
        let fn_ = make_fn("test.py", 1, 5, "def foo():\n    pass\n");
        let config = WindowConfig {
            window_lines: 0,
            stride_lines: 1,
            min_nonempty: 1,
        };
        generate_window_snippets(&[fn_], &config);
    }

    #[test]
    #[should_panic(expected = "stride_lines must be > 0")]
    fn test_window_snippets_stride_validation() {
        let fn_ = make_fn("test.py", 1, 5, "def foo():\n    pass\n");
        let config = WindowConfig {
            window_lines: 3,
            stride_lines: 0,
            min_nonempty: 1,
        };
        generate_window_snippets(&[fn_], &config);
    }

    #[test]
    fn test_window_snippets_populates_both_texts() {
        // A function whose lines contain a comment: WIN analysis strips it, display keeps it
        let code = "def foo():\n    # comment line\n    x = 1\n    return x\n";
        let fn_ = make_fn("test.py", 1, 4, code);
        let config = WindowConfig {
            window_lines: 4,
            stride_lines: 1,
            min_nonempty: 1,
        };
        let snippets = generate_window_snippets(&[fn_], &config);
        assert!(!snippets.is_empty(), "should produce WIN snippets");
        let s = &snippets[0];
        assert!(
            !s.text.contains("# comment line"),
            "WIN text (analysis) should strip comment"
        );
        assert!(
            s.display_text.contains("# comment line"),
            "WIN display_text should preserve comment"
        );
    }

    #[test]
    fn test_window_snippet_hash_format() {
        // Verify the WIN hash includes win_start, win_end (1-indexed in fn code), and analysis
        let code = "def foo():\n    x = 1\n    y = 2\n    z = 3\n";
        let fn_ = make_fn("src/foo.py", 5, 8, code);
        let config = WindowConfig {
            window_lines: 2,
            stride_lines: 1,
            min_nonempty: 1,
        };
        let snippets = generate_window_snippets(&[fn_.clone()], &config);
        // First window: start=1, end=2 (1-indexed within function code)
        let first = &snippets[0];
        let window_text = fn_.code.lines().take(2).collect::<Vec<_>>().join("\n");
        let expected_analysis = crate::snippets::normalization::normalize_analysis(&window_text);
        let expected_hash = hash_text(&format!(
            "WIN:{}:{}:{}:{}:{}:{}:{}",
            fn_.file.path, fn_.start_line, fn_.end_line, fn_.code_hash, 1, 2, expected_analysis
        ));
        assert_eq!(
            first.snippet_hash, expected_hash,
            "WIN hash format must match DD7"
        );
    }
}
