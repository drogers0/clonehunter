// Items here are used starting T12 (diff command); allow dead_code until then.
#![allow(dead_code)]

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use thiserror::Error;

/// Error from a git subprocess call.
#[derive(Debug, Error)]
#[error("{0}")]
pub(crate) struct GitError(String);

/// Return files changed since `base` (union of tracked diffs + untracked files).
///
/// Matches Python's `io/git.py`:
/// - `git diff --name-only <base>` for tracked modifications
/// - `git ls-files --others --exclude-standard` for untracked files
///
/// `cwd`: working directory for the git commands. `None` inherits the process cwd
/// (same as Python's behavior). Pass `Some(repo_root)` to avoid depending on the
/// process cwd, which is useful in tests and CLI invocations that know the repo root.
///
/// NOTE: This adds a `cwd` parameter not in the plan's Python-derived signature.
/// Rationale: Rust tests run in parallel; process-cwd mutation causes races.
/// T12 must pass the resolved repo root explicitly (use `find_config_root` or `.`).
///
/// Results are deduplicated and sorted for deterministic ordering.
/// An optional `paths` slice restricts the query to specific paths (passed after `--`).
pub(crate) fn changed_files(
    base: &str,
    paths: Option<&[String]>,
    cwd: Option<&Path>,
) -> Result<Vec<String>, GitError> {
    let tracked = git_name_only(&with_paths(&["diff", "--name-only", base], paths), cwd)?;
    let untracked = git_name_only(
        &with_paths(&["ls-files", "--others", "--exclude-standard"], paths),
        cwd,
    )?;

    let mut seen = HashSet::new();
    let mut files = Vec::new();
    for raw in tracked.into_iter().chain(untracked) {
        let normalized = Path::new(&raw).to_string_lossy().into_owned();
        if seen.insert(normalized.clone()) {
            files.push(normalized);
        }
    }
    files.sort(); // deterministic ordering
    Ok(files)
}

/// Build a `Vec<String>` of git args, optionally appending `-- <paths>`.
fn with_paths(args: &[&str], paths: Option<&[String]>) -> Vec<String> {
    let mut result: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    if let Some(p) = paths {
        if !p.is_empty() {
            result.push("--".into());
            result.extend(p.iter().cloned());
        }
    }
    result
}

/// Run `git <args>` and return the output lines, or a `GitError` on non-zero exit.
fn git_name_only(args: &[String], cwd: Option<&Path>) -> Result<Vec<String>, GitError> {
    let mut cmd = Command::new("git");
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let output = cmd.output().map_err(|e| GitError(e.to_string()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let details = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            format!(
                "git {} failed with exit code {}",
                args.join(" "),
                output.status
            )
        };
        return Err(GitError(details));
    }

    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command as StdCommand;
    use tempfile::TempDir;

    fn git(dir: &Path, args: &[&str]) {
        let status = StdCommand::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .expect("git command failed to run");
        assert!(status.success(), "git {:?} failed", args);
    }

    #[test]
    fn changed_files_outside_git_raises() {
        let dir = TempDir::new().unwrap();
        let result = changed_files("HEAD", None, Some(dir.path()));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        // Git error message varies by version; just check it's not empty
        assert!(!msg.is_empty());
    }

    #[test]
    fn changed_files_includes_untracked() {
        let dir = TempDir::new().unwrap();
        let repo = dir.path().join("repo");
        std::fs::create_dir(&repo).unwrap();

        git(&repo, &["init"]);
        git(&repo, &["config", "user.email", "test@example.com"]);
        git(&repo, &["config", "user.name", "Test User"]);

        std::fs::write(repo.join("tracked.py"), "def keep():\n    return 1\n").unwrap();
        git(&repo, &["add", "tracked.py"]);
        git(&repo, &["commit", "-m", "init"]);

        std::fs::write(repo.join("new_file.py"), "def new_file():\n    return 2\n").unwrap();

        let files = changed_files("HEAD", None, Some(&repo)).unwrap();
        assert!(files.contains(&"new_file.py".to_string()));
    }

    #[test]
    fn changed_files_respects_paths() {
        let dir = TempDir::new().unwrap();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::create_dir_all(repo.join("tests")).unwrap();

        git(&repo, &["init"]);
        git(&repo, &["config", "user.email", "test@example.com"]);
        git(&repo, &["config", "user.name", "Test User"]);

        std::fs::write(repo.join("src/a.py"), "def a():\n    return 1\n").unwrap();
        std::fs::write(repo.join("tests/b.py"), "def b():\n    return 2\n").unwrap();
        git(&repo, &["add", "src/a.py", "tests/b.py"]);
        git(&repo, &["commit", "-m", "init"]);

        std::fs::write(repo.join("src/a.py"), "def a():\n    return 10\n").unwrap();
        std::fs::write(repo.join("tests/b.py"), "def b():\n    return 20\n").unwrap();

        let files = changed_files("HEAD", Some(&["src".to_string()]), Some(&repo)).unwrap();

        assert_eq!(files, vec!["src/a.py"]);
    }
}
