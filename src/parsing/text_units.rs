// Items here are used starting T10 (pipeline); allow dead_code until then.
#![allow(dead_code)]

use std::path::Path;

use crate::core::types::{FileRef, FunctionRef};
use crate::io::fingerprints::hash_text;

/// Convert a non-Python file into a single whole-file FunctionRef.
/// Returns empty Vec on IO error or empty/whitespace-only content.
pub(crate) fn extract_file_unit(file: &FileRef) -> Vec<FunctionRef> {
    let bytes = match std::fs::read(&file.path) {
        Ok(b) => b,
        Err(_) => return vec![],
    };
    let content = String::from_utf8_lossy(&bytes).into_owned();

    if content.trim().is_empty() {
        return vec![];
    }

    let end_line = content.lines().count().max(1);
    let qualified_name = Path::new(&file.path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
    let code_hash = hash_text(&content);

    vec![FunctionRef {
        file: file.clone(),
        qualified_name,
        start_line: 1,
        end_line,
        code: content,
        code_hash,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::NamedTempFile;

    use crate::core::types::Language;

    fn make_file_ref(path: &str) -> FileRef {
        FileRef {
            path: path.to_string(),
            content_hash: String::new(),
            language: Language::Text,
        }
    }

    #[test]
    fn test_extract_file_unit_basic() {
        let mut tmp = NamedTempFile::with_suffix(".js").unwrap();
        tmp.write_all(b"function hello() { return 1; }\n").unwrap();
        let path = tmp.path().to_string_lossy().into_owned();
        let file = make_file_ref(&path);
        let units = extract_file_unit(&file);
        assert_eq!(units.len(), 1);
        let unit = &units[0];
        // qualified_name is just the filename
        let file_name = tmp.path().file_name().unwrap().to_str().unwrap();
        assert_eq!(unit.qualified_name, file_name);
        assert_eq!(unit.start_line, 1);
    }

    #[test]
    fn test_extract_file_unit_empty_file() {
        let mut tmp = NamedTempFile::with_suffix(".js").unwrap();
        tmp.write_all(b"").unwrap();
        let path = tmp.path().to_string_lossy().into_owned();
        let file = make_file_ref(&path);
        let units = extract_file_unit(&file);
        assert!(units.is_empty(), "empty file should return no units");
    }

    #[test]
    fn test_extract_file_unit_whitespace_only() {
        let mut tmp = NamedTempFile::with_suffix(".js").unwrap();
        tmp.write_all(b"   \n  \n   ").unwrap();
        let path = tmp.path().to_string_lossy().into_owned();
        let file = make_file_ref(&path);
        let units = extract_file_unit(&file);
        assert!(
            units.is_empty(),
            "whitespace-only file should return no units"
        );
    }

    #[test]
    fn test_extract_file_unit_end_line() {
        let content = "line1\nline2\nline3\nline4\n";
        let mut tmp = NamedTempFile::with_suffix(".js").unwrap();
        tmp.write_all(content.as_bytes()).unwrap();
        let path = tmp.path().to_string_lossy().into_owned();
        let file = make_file_ref(&path);
        let units = extract_file_unit(&file);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].end_line, 4, "end_line should match line count");
    }

    #[test]
    fn test_extract_file_unit_code_matches_content() {
        let content = "const x = 1;\nconst y = 2;\n";
        let mut tmp = NamedTempFile::with_suffix(".js").unwrap();
        tmp.write_all(content.as_bytes()).unwrap();
        let path = tmp.path().to_string_lossy().into_owned();
        let file = make_file_ref(&path);
        let units = extract_file_unit(&file);
        assert_eq!(units[0].code, content);
        assert_eq!(units[0].code_hash, hash_text(content));
    }
}
