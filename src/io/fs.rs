use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use globset::{Glob, GlobSet, GlobSetBuilder};
use thiserror::Error;
use walkdir::WalkDir;

use crate::core::types::{FileRef, Language};
use crate::io::fingerprints::hash_text;

/// Error type for file-collection failures (DD8).
///
/// Unlike Python's `collect_files` which silently returns `[]` on bad globs
/// (since its manual string matcher can't fail), we propagate parse errors.
/// A typo in glob patterns would otherwise silently produce a zero-file scan.
#[derive(Debug, Error)]
pub(crate) enum FsError {
    #[error("{0}")]
    InvalidGlob(String),
}

/// Detect language from file extension. `.py` → Python, everything else → Text.
fn detect_language(path: &Path) -> Language {
    match path.extension().and_then(|e| e.to_str()) {
        Some("py") => Language::Python,
        _ => Language::Text,
    }
}

/// Build a `GlobSet` from a list of glob patterns, returning an error if any pattern is invalid.
///
/// Globset divergence from Python's `PurePosixPath.match()`:
/// - Python matches from the right by default (anchored at filename); globset matches from the
///   left (anchored at root). Mitigated because all default globs use `**/` prefix patterns
///   (e.g., `**/*.py`, `**/.venv/**`) which work identically in both matchers.
/// - Patterns without a leading `**` may behave differently. T13 re-freeze absorbs any
///   file-collection differences into the new Rust baseline.
fn build_glob_set(patterns: &[String]) -> Result<GlobSet, String> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob = Glob::new(pattern).map_err(|e| format!("invalid glob {pattern:?}: {e}"))?;
        builder.add(glob);
    }
    builder
        .build()
        .map_err(|e| format!("glob set build error: {e}"))
}

/// Canonical key for deduplication: resolved absolute path as a string.
fn canonical_key(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Collect files matching include/exclude globs from the given paths.
///
/// Behavior:
/// - Directories are walked recursively; excluded directories are pruned early.
/// - Explicit file paths are filtered against include/exclude globs.
/// - Files are deduped by canonical (resolved) path.
/// - Results are sorted by path for deterministic ordering.
/// - Non-UTF-8 file bytes are read with replacement characters (matching Python's `errors="replace"`).
/// - Unreadable files are silently skipped (matching Python's `except OSError: continue`).
///
/// Returns `Err(FsError::InvalidGlob)` if any glob pattern fails to parse (DD8).
pub(crate) fn collect_files(
    paths: &[String],
    include_globs: &[String],
    exclude_globs: &[String],
) -> Result<Vec<FileRef>, FsError> {
    let include_set = build_glob_set(include_globs).map_err(FsError::InvalidGlob)?;
    let exclude_set = build_glob_set(exclude_globs).map_err(FsError::InvalidGlob)?;

    let mut gathered: Vec<PathBuf> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for raw in paths {
        let p = Path::new(raw);
        if p.is_dir() {
            walk_directory(p, &include_set, &exclude_set, &mut gathered, &mut seen);
        } else if p.is_file() {
            add_explicit_file(p, &include_set, &exclude_set, &mut gathered, &mut seen);
        }
    }

    // Sort for deterministic ordering across platforms
    gathered.sort();

    let mut results = Vec::new();
    for path in &gathered {
        let language = detect_language(path);
        let content = match std::fs::read(path) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(_) => continue, // silently skip unreadable files
        };
        results.push(FileRef {
            path: path.to_string_lossy().into_owned(),
            content_hash: hash_text(&content),
            language,
            content: Arc::from(content),
        });
    }
    Ok(results)
}

fn walk_directory(
    root: &Path,
    include: &GlobSet,
    exclude: &GlobSet,
    gathered: &mut Vec<PathBuf>,
    seen: &mut HashSet<String>,
) {
    for entry in WalkDir::new(root)
        .follow_links(false)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|e| {
            // Prune excluded directories early to avoid descending into them
            if e.file_type().is_dir() {
                let rel = match e.path().strip_prefix(root) {
                    Ok(r) => r,
                    Err(_) => return true,
                };
                // Never exclude the root itself
                if rel.as_os_str().is_empty() {
                    return true;
                }
                return !exclude.is_match(rel);
            }
            true
        })
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = match entry.path().strip_prefix(root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if !include.is_match(rel) {
            continue;
        }
        if exclude.is_match(rel) {
            continue;
        }
        append_unique(entry.path(), gathered, seen);
    }
}

fn add_explicit_file(
    path: &Path,
    include: &GlobSet,
    exclude: &GlobSet,
    gathered: &mut Vec<PathBuf>,
    seen: &mut HashSet<String>,
) {
    // For explicit file paths, match against the path relative to cwd when possible,
    // since globs like `**/*.py` are tested against a relative path component.
    let match_path = if path.is_absolute() {
        match std::env::current_dir() {
            Ok(cwd) => path
                .strip_prefix(&cwd)
                .map(|r| r.to_path_buf())
                .unwrap_or_else(|_| path.to_path_buf()),
            Err(_) => path.to_path_buf(),
        }
    } else {
        path.to_path_buf()
    };

    if include.is_match(&match_path) && !exclude.is_match(&match_path) {
        append_unique(path, gathered, seen);
    }
}

fn append_unique(path: &Path, gathered: &mut Vec<PathBuf>, seen: &mut HashSet<String>) {
    let key = canonical_key(path);
    if seen.insert(key) {
        gathered.push(path.to_path_buf());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// RAII guard that restores cwd on drop (including panics).
    struct CwdGuard(PathBuf);
    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.0);
        }
    }

    #[test]
    fn collect_files_excludes_venv() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("repo");
        let venv = root.join(".venv").join("lib");
        std::fs::create_dir_all(&venv).unwrap();
        std::fs::write(venv.join("skip.py"), "print('skip')").unwrap();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("keep.py"), "print('keep')").unwrap();

        let files = collect_files(
            &[root.to_string_lossy().into_owned()],
            &["**/*.py".into()],
            &["**/.venv/**".into(), "**/site-packages/**".into()],
        )
        .unwrap();
        let names: Vec<_> = files
            .iter()
            .map(|f| Path::new(&f.path).file_name().unwrap().to_str().unwrap())
            .collect();
        assert!(names.contains(&"keep.py"));
        assert!(!names.contains(&"skip.py"));
    }

    #[test]
    fn collect_files_handles_non_utf8() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("bad.py"), b"\xff\xfe\xfa").unwrap();
        let files = collect_files(
            &[dir.path().to_string_lossy().into_owned()],
            &["**/*.py".into()],
            &[],
        )
        .unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn non_python_files_are_text_language() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("sample.js"), "const x = 1;\n").unwrap();
        let files = collect_files(
            &[dir.path().to_string_lossy().into_owned()],
            &["**/*.js".into()],
            &[],
        )
        .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].language, Language::Text);
    }

    #[test]
    fn dedupes_overlapping_inputs() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("repo");
        let nested = root.join("pkg");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("a.py"), "print('x')\n").unwrap();

        let files = collect_files(
            &[
                root.to_string_lossy().into_owned(),
                nested.join("a.py").to_string_lossy().into_owned(),
            ],
            &["**/*.py".into()],
            &[],
        )
        .unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn deterministic_ordering() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("repo");
        std::fs::create_dir_all(&root).unwrap();
        for name in ["z.py", "a.py", "m.py"] {
            std::fs::write(root.join(name), "pass\n").unwrap();
        }
        let files = collect_files(
            &[root.to_string_lossy().into_owned()],
            &["**/*.py".into()],
            &[],
        )
        .unwrap();
        let names: Vec<_> = files
            .iter()
            .map(|f| {
                Path::new(&f.path)
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "files should be in sorted order");
    }

    #[test]
    fn explicit_path_respects_excludes() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("repo");
        let target = root.join("src").join("keep.py");
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(&target, "print('keep')").unwrap();

        let _guard = CwdGuard(std::env::current_dir().unwrap());
        std::env::set_current_dir(&root).unwrap();
        let files = collect_files(
            &["src/keep.py".into()],
            &["**/*.py".into()],
            &["src/**".into()],
        )
        .unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn external_absolute_path_respects_excludes() {
        // Note: even if parallel tests race on the process cwd, this test is robust:
        // if strip_prefix fails (wrong cwd), add_explicit_file falls back to the
        // full absolute path, which still contains "generated" and is correctly excluded.
        let dir = TempDir::new().unwrap();
        let cwd = dir.path().join("cwd");
        std::fs::create_dir(&cwd).unwrap();
        let external = dir
            .path()
            .join("external")
            .join("generated")
            .join("skip.py");
        std::fs::create_dir_all(external.parent().unwrap()).unwrap();
        std::fs::write(&external, "print('skip')\n").unwrap();

        let _guard = CwdGuard(std::env::current_dir().unwrap());
        std::env::set_current_dir(&cwd).unwrap();
        let files = collect_files(
            &[external.to_string_lossy().into_owned()],
            &["**/*.py".into()],
            &["**/generated/**".into()],
        )
        .unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn invalid_glob_returns_err() {
        let dir = TempDir::new().unwrap();
        let result = collect_files(
            &[dir.path().to_string_lossy().into_owned()],
            &["[invalid".into()], // unclosed bracket = invalid glob
            &[],
        );
        assert!(matches!(result, Err(FsError::InvalidGlob(_))));
    }
}
