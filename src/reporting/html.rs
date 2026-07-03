use std::fs::File;
use std::io::{BufWriter, Write as IoWrite};

use similar::TextDiff;

use crate::core::types::{CandidateMatch, Finding, ScanResult};
use crate::reporting::ReportError;
use crate::reporting::compare::{CompareData, select_compare};
use crate::reporting::schema::SCHEMA_VERSION;
use crate::similarity::{SelfCloneOccurrences, best_match, is_self_clone};

pub(crate) fn write_html(result: &ScanResult, out_path: &str) -> Result<(), ReportError> {
    let rows: Vec<String> = result.findings.iter().map(render_finding).collect();
    let rows_html = rows.join("\n");
    let finding_count = result.findings.len();
    let doc = format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CloneHunter Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: sans-serif; padding: 24px; }}
    .list {{ display: flex; flex-direction: column; gap: 12px; }}
    details {{ border: 1px solid #ddd; border-radius: 6px; padding: 8px; width: 100%; }}
    details > summary {{ cursor: pointer; list-style: none; }}
    details > summary::-webkit-details-marker {{ display: none; }}
    .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr 90px 130px;
      gap: 12px; align-items: start; }}
    .summary-grid > div {{ min-width: 0; }}
    .path {{ color: #555; font-size: 0.9em; word-break: break-all; }}
    .meta {{ color: #444; font-size: 0.9em; }}
    .controls {{ display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }}
    .controls label {{ font-size: 0.9em; color: #333; }}
    .controls select {{ padding: 4px 6px; }}
    .compare-grid {{ margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr;
      gap: 12px; }}
    .diff-wrap {{ overflow-x: auto; max-width: 100%; }}
    table.diff {{ width: 100%; border-collapse: collapse; font-family: monospace;
      font-size: 12px; table-layout: fixed; }}
    table.diff th, table.diff td {{ padding: 4px 6px; vertical-align: top;
      border: 1px solid #e5e5e5; }}
    table.diff th {{ background: #f3f3f3; text-align: left; }}
    td.line-no {{ width: 3.5em; text-align: right; color: #666; }}
    td.code {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    .code-box {{ background: #fafafa; border: 1px solid #eee; padding: 8px;
      white-space: pre-wrap; overflow-x: auto; font-family: monospace; }}
    .diff_header {{ background: #f3f3f3; }}
    .diff_add {{ background: #e6ffed; }}
    .diff_chg {{ background: #fff5b1; }}
    .diff_sub {{ background: #ffeef0; }}
  </style>
</head>
<body>
  <h1>CloneHunter Report</h1>
  <p>Schema: {SCHEMA_VERSION}</p>
  <p>Findings: {finding_count}</p>
  <div class="controls">
    <label for="sort-findings">Sort findings:</label>
    <select id="sort-findings">
      <option value="lines_desc">Duplicated lines (high to low)</option>
      <option value="score_desc">Match score (high to low)</option>
      <option value="path_asc">File path (A/B, A to Z)</option>
    </select>
  </div>
  <div class="list">
    {rows_html}
  </div>
  <script>
    (() => {{
      const list = document.querySelector(".list");
      const sortSelect = document.getElementById("sort-findings");
      if (!list || !sortSelect) return;

      const collator = new Intl.Collator(undefined, {{ sensitivity: "base", numeric: true }});

      const sortFindings = () => {{
        const items = Array.from(list.querySelectorAll("details"));
        items.sort((a, b) => {{
          const mode = sortSelect.value;
          if (mode === "path_asc") {{
            const aPath = a.dataset.pathMin || "";
            const bPath = b.dataset.pathMin || "";
            return collator.compare(aPath, bPath);
          }}
          if (mode === "lines_desc") {{
            const aLines = Number(a.dataset.lines || "0");
            const bLines = Number(b.dataset.lines || "0");
            return bLines - aLines;
          }}
          const aScore = Number(a.dataset.score || "0");
          const bScore = Number(b.dataset.score || "0");
          return bScore - aScore;
        }});
        for (const item of items) {{
          list.appendChild(item);
        }}
      }};

      sortSelect.addEventListener("change", sortFindings);
      sortFindings();
    }})();
  </script>
</body>
</html>
"#
    );
    let mut file = BufWriter::new(File::create(out_path)?);
    file.write_all(doc.as_bytes())?;
    Ok(())
}

// ─── Per-finding rendering ────────────────────────────────────────────────────

fn render_finding(finding: &Finding) -> String {
    let func_a = &finding.function_a;
    let func_b = &finding.function_b;
    let matches = &finding.evidence;
    let occ = if is_self_clone(matches) {
        Some(SelfCloneOccurrences::new(matches))
    } else {
        None
    };
    let (span_a, span_b) = evidence_bounds(matches, occ.as_ref());
    let html_compare = build_html_compare(matches, occ.as_ref());
    let diff_html = render_diff(html_compare.as_ref());

    // path_min: case-insensitive minimum of the two paths
    let path_min = if func_a.file.path.to_lowercase() <= func_b.file.path.to_lowercase() {
        &func_a.file.path
    } else {
        &func_b.file.path
    };

    format!(
        r#"
<details
  data-path-min="{path_min_escaped}"
  data-score="{score}"
  data-lines="{lines}"
>
  <summary>
    <div class="summary-grid">
      <div>
        <div>{qname_a}</div>
        <div class="path">
          {path_a}:{span_a0}-{span_a1}
        </div>
      </div>
      <div>
        <div>{qname_b}</div>
        <div class="path">
          {path_b}:{span_b0}-{span_b1}
        </div>
      </div>
      <div>{score:.3}</div>
      <div>{lines} duplicated lines</div>
    </div>
  </summary>
  {diff_html}
</details>
"#,
        path_min_escaped = html_escape(path_min),
        score = finding.score,
        lines = finding.duplicated_lines,
        qname_a = html_escape(&func_a.qualified_name),
        path_a = html_escape(&func_a.file.path),
        span_a0 = span_a.0,
        span_a1 = span_a.1,
        qname_b = html_escape(&func_b.qualified_name),
        path_b = html_escape(&func_b.file.path),
        span_b0 = span_b.0,
        span_b1 = span_b.1,
    )
}

// ─── HtmlCompareData ─────────────────────────────────────────────────────────

struct HtmlCompareData {
    compare: CompareData,
    hidden_before_a: usize,
    hidden_before_b: usize,
    hidden_after_a: usize,
    hidden_after_b: usize,
}

fn build_html_compare(
    matches: &[CandidateMatch],
    occ: Option<&SelfCloneOccurrences>,
) -> Option<HtmlCompareData> {
    let compare = select_compare(matches)?;
    let (hidden_before_a, hidden_before_b, hidden_after_a, hidden_after_b) =
        hidden_duplicated_lines(matches, compare.span_a, compare.span_b, occ);
    Some(HtmlCompareData {
        compare,
        hidden_before_a,
        hidden_before_b,
        hidden_after_a,
        hidden_after_b,
    })
}

// ─── evidence_bounds ─────────────────────────────────────────────────────────

fn evidence_bounds(
    matches: &[CandidateMatch],
    occ: Option<&SelfCloneOccurrences>,
) -> ((usize, usize), (usize, usize)) {
    if matches.is_empty() {
        return ((1, 1), (1, 1));
    }
    if let Some(occ) = occ {
        // Self-clone: show the occurrences the diff is rendered for
        let best = best_match(matches).expect("non-empty matches");
        let span_a = occ.occurrence_for(best.snippet_a.start_line, best.snippet_a.end_line);
        let span_b = occ.occurrence_for(best.snippet_b.start_line, best.snippet_b.end_line);
        return (span_a, span_b);
    }
    let min_a = matches
        .iter()
        .map(|m| m.snippet_a.start_line)
        .min()
        .unwrap();
    let max_a = matches.iter().map(|m| m.snippet_a.end_line).max().unwrap();
    let min_b = matches
        .iter()
        .map(|m| m.snippet_b.start_line)
        .min()
        .unwrap();
    let max_b = matches.iter().map(|m| m.snippet_b.end_line).max().unwrap();
    ((min_a, max_a), (min_b, max_b))
}

// ─── hidden line counts ───────────────────────────────────────────────────────

fn hidden_duplicated_lines(
    matches: &[CandidateMatch],
    span_a: (usize, usize),
    span_b: (usize, usize),
    occ: Option<&SelfCloneOccurrences>,
) -> (usize, usize, usize, usize) {
    if let Some(occ) = occ {
        let occ_a = occ.occurrence_for(span_a.0, span_a.1);
        let occ_b = occ.occurrence_for(span_b.0, span_b.1);
        return (
            span_a.0.saturating_sub(occ_a.0),
            span_b.0.saturating_sub(occ_b.0),
            occ_a.1.saturating_sub(span_a.1),
            occ_b.1.saturating_sub(span_b.1),
        );
    }
    let spans_a: Vec<(usize, usize)> = matches
        .iter()
        .map(|m| (m.snippet_a.start_line, m.snippet_a.end_line))
        .collect();
    let spans_b: Vec<(usize, usize)> = matches
        .iter()
        .map(|m| (m.snippet_b.start_line, m.snippet_b.end_line))
        .collect();
    let before_a = covered_in_range(&spans_a, 1, span_a.0.saturating_sub(1));
    let before_b = covered_in_range(&spans_b, 1, span_b.0.saturating_sub(1));
    let after_a = covered_in_range(&spans_a, span_a.1 + 1, 1_000_000_000);
    let after_b = covered_in_range(&spans_b, span_b.1 + 1, 1_000_000_000);
    (before_a, before_b, after_a, after_b)
}

fn covered_in_range(spans: &[(usize, usize)], start: usize, end: usize) -> usize {
    if start > end {
        return 0;
    }
    let mut covered = 0;
    for &(span_start, span_end) in &merge_spans_adjacent(spans) {
        let overlap_start = start.max(span_start);
        let overlap_end = end.min(span_end);
        if overlap_start <= overlap_end {
            covered += overlap_end - overlap_start + 1;
        }
    }
    covered
}

/// Merge spans with ADJACENCY (`start <= prev_end + 1`), matching Python's `_merge_spans`.
/// This is intentionally different from `merge_overlapping` in occurrences.rs (strict overlap).
fn merge_spans_adjacent(spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if spans.is_empty() {
        return vec![];
    }
    let mut sorted = spans.to_vec();
    sorted.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in sorted {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 + 1 {
                if end > last.1 {
                    last.1 = end;
                }
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

// ─── Diff rendering ───────────────────────────────────────────────────────────

fn render_diff(compare: Option<&HtmlCompareData>) -> String {
    let Some(compare) = compare else {
        return r#"<div class="code-box">No diff available.</div>"#.into();
    };
    let lines_a = strip_blank_lines(&compare.compare.display_text_a, compare.compare.span_a.0);
    let lines_b = strip_blank_lines(&compare.compare.display_text_b, compare.compare.span_b.0);
    let table = render_side_by_side(
        &lines_a,
        &lines_b,
        compare.hidden_before_a,
        compare.hidden_before_b,
        compare.hidden_after_a,
        compare.hidden_after_b,
    );
    format!(r#"<div class="diff-wrap">{table}</div>"#)
}

/// Strip blank lines, preserving absolute line numbers from `start`.
fn strip_blank_lines(text: &str, start: usize) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| (start + i, line.to_string()))
        .collect()
}

fn render_side_by_side(
    lines_a: &[(usize, String)],
    lines_b: &[(usize, String)],
    hidden_before_a: usize,
    hidden_before_b: usize,
    hidden_after_a: usize,
    hidden_after_b: usize,
) -> String {
    let text_a: Vec<&str> = lines_a.iter().map(|(_, l)| l.as_str()).collect();
    let text_b: Vec<&str> = lines_b.iter().map(|(_, l)| l.as_str()).collect();
    let diff = TextDiff::from_slices(text_a.as_slice(), text_b.as_slice());

    let mut rows: Vec<String> = Vec::new();

    if let Some(top) = render_hidden_row(hidden_before_a, hidden_before_b) {
        rows.push(top);
    }

    for op in diff.ops() {
        let old_range = op.old_range();
        let new_range = op.new_range();
        match op {
            similar::DiffOp::Equal { .. } => {
                for (i, j) in old_range.zip(new_range) {
                    let (a_no, a_line) = &lines_a[i];
                    let (b_no, b_line) = &lines_b[j];
                    rows.push(render_row(Some(*a_no), a_line, Some(*b_no), b_line, ""));
                }
            }
            similar::DiffOp::Replace { .. } => {
                let count = old_range.len().max(new_range.len());
                for offset in 0..count {
                    let (a_no, a_line): (Option<usize>, &str) =
                        if old_range.start + offset < old_range.end {
                            let (no, line) = &lines_a[old_range.start + offset];
                            (Some(*no), line.as_str())
                        } else {
                            (None, "")
                        };
                    let (b_no, b_line): (Option<usize>, &str) =
                        if new_range.start + offset < new_range.end {
                            let (no, line) = &lines_b[new_range.start + offset];
                            (Some(*no), line.as_str())
                        } else {
                            (None, "")
                        };
                    rows.push(render_row(a_no, a_line, b_no, b_line, "diff_chg"));
                }
            }
            similar::DiffOp::Delete { .. } => {
                for i in old_range {
                    let (a_no, a_line) = &lines_a[i];
                    rows.push(render_row(Some(*a_no), a_line, None, "", "diff_sub"));
                }
            }
            similar::DiffOp::Insert { .. } => {
                for j in new_range {
                    let (b_no, b_line) = &lines_b[j];
                    rows.push(render_row(None, "", Some(*b_no), b_line, "diff_add"));
                }
            }
        }
    }

    if let Some(bottom) = render_hidden_row(hidden_after_a, hidden_after_b) {
        rows.push(bottom);
    }

    let header = concat!(
        r#"<table class="diff">"#,
        "<colgroup>",
        r#"<col style="width:3.5em" />"#,
        r#"<col style="width:calc((100% - 7em) / 2)" />"#,
        r#"<col style="width:3.5em" />"#,
        r#"<col style="width:calc((100% - 7em) / 2)" />"#,
        "</colgroup>",
        "<thead><tr>",
        r#"<th class="line-no"></th><th>Function A</th>"#,
        r#"<th class="line-no"></th><th>Function B</th>"#,
        "</tr></thead><tbody>",
    );
    let footer = "</tbody></table>";
    format!("{header}{}{footer}", rows.join(""))
}

fn render_row(
    a_no: Option<usize>,
    a_line: &str,
    b_no: Option<usize>,
    b_line: &str,
    cls: &str,
) -> String {
    let a_no_text = a_no.map(|n| n.to_string()).unwrap_or_default();
    let b_no_text = b_no.map(|n| n.to_string()).unwrap_or_default();
    let a_text = html_escape(a_line);
    let b_text = html_escape(b_line);
    let class_attr = if cls.is_empty() {
        String::new()
    } else {
        format!(" {cls}")
    };
    format!(
        "<tr>\
         <td class=\"line-no{class_attr}\">{a_no_text}</td>\
         <td class=\"code{class_attr}\">{a_text}</td>\
         <td class=\"line-no{class_attr}\">{b_no_text}</td>\
         <td class=\"code{class_attr}\">{b_text}</td>\
         </tr>"
    )
}

fn render_hidden_row(count_a: usize, count_b: usize) -> Option<String> {
    if count_a == 0 && count_b == 0 {
        return None;
    }
    let marker_a = if count_a > 0 {
        html_escape(&format!("<{count_a} lines not shown>"))
    } else {
        String::new()
    };
    let marker_b = if count_b > 0 {
        html_escape(&format!("<{count_b} lines not shown>"))
    } else {
        String::new()
    };
    Some(format!(
        "<tr>\
         <td class=\"line-no\"></td>\
         <td class=\"meta\">{marker_a}</td>\
         <td class=\"line-no\"></td>\
         <td class=\"meta\">{marker_b}</td>\
         </tr>"
    ))
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::core::types::{
        CandidateMatch, FileRef, Finding, FunctionRef, Language, ScanResult, ScanStats,
        SnippetKind, SnippetRef,
    };
    use tempfile::TempDir;

    fn make_result(findings: Vec<Finding>) -> ScanResult {
        ScanResult {
            findings,
            stats: ScanStats {
                file_count: 1,
                function_count: 2,
                snippet_count: 2,
                candidate_count: 1,
                finding_count: 0,
                cache_hits: 0,
                cache_misses: 0,
            },
            config_snapshot: serde_json::json!({}),
            timing: BTreeMap::new(),
            degradations: vec![],
        }
    }

    fn make_finding_with_code(code_a: &str, code_b: &str) -> Finding {
        let file = FileRef {
            path: "a.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
        };
        let func_a = FunctionRef {
            file: file.clone(),
            qualified_name: "foo".into(),
            start_line: 1,
            end_line: 10,
            code: code_a.into(),
            code_hash: "ca".into(),
        };
        let func_b = FunctionRef {
            file,
            qualified_name: "bar".into(),
            start_line: 20,
            end_line: 30,
            code: code_b.into(),
            code_hash: "cb".into(),
        };
        let snip_a = SnippetRef {
            kind: SnippetKind::Func,
            function: func_a.clone(),
            start_line: 1,
            end_line: 10,
            text: code_a.into(),
            display_text: code_a.into(),
            snippet_hash: "sha".into(),
        };
        let snip_b = SnippetRef {
            kind: SnippetKind::Func,
            function: func_b.clone(),
            start_line: 20,
            end_line: 30,
            text: code_b.into(),
            display_text: code_b.into(),
            snippet_hash: "shb".into(),
        };
        Finding {
            function_a: func_a,
            function_b: func_b,
            score: 0.95,
            duplicated_lines: 10,
            evidence: vec![CandidateMatch {
                snippet_a: snip_a,
                snippet_b: snip_b,
                similarity: 0.95,
                evidence: "".into(),
            }],
            reasons: vec!["test".into()],
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn html_contains_report_header() {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.html").to_string_lossy().into_owned();
        write_html(&make_result(vec![]), &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        assert!(content.contains("CloneHunter Report"));
    }

    #[test]
    fn html_contains_schema_version() {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.html").to_string_lossy().into_owned();
        write_html(&make_result(vec![]), &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        assert!(content.contains(SCHEMA_VERSION));
    }

    #[test]
    fn html_escape_special_chars() {
        let escaped = html_escape("<script>alert('xss')</script>");
        assert!(!escaped.contains('<'));
        assert!(!escaped.contains('>'));
        assert!(escaped.contains("&lt;script&gt;"));
    }

    #[test]
    fn html_uses_display_text() {
        // The display_text contains a comment; verify it appears in rendered HTML
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.html").to_string_lossy().into_owned();
        let finding = make_finding_with_code(
            "def foo():\n    # a comment\n    return 1",
            "def bar():\n    # a comment\n    return 1",
        );
        write_html(&make_result(vec![finding]), &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        // The comment should appear (display_text preserves comments)
        assert!(content.contains("# a comment"));
    }

    #[test]
    fn html_self_clone_hidden_lines() {
        // Two overlapping windows within the same function → they merge into one occurrence
        // (1-10) ∪ (5-15) = (1-15).  Showing span_a=(1-10) vs span_b=(5-15) means:
        //   hidden_after_a  = 15-10 = 5  ("5 lines not shown" after left side)
        //   hidden_before_b = 5-1   = 4  ("4 lines not shown" before right side)
        let file = FileRef {
            path: "a.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
        };
        let func = FunctionRef {
            file,
            qualified_name: "f".into(),
            start_line: 1,
            end_line: 50,
            code: "pass".into(),
            code_hash: "c".into(),
        };
        let snip_a = SnippetRef {
            kind: SnippetKind::Win,
            function: func.clone(),
            start_line: 1,
            end_line: 10,
            text: "block1\nblock2\nblock3".into(),
            display_text: "block1\nblock2\nblock3".into(),
            snippet_hash: "a".into(),
        };
        let snip_b = SnippetRef {
            kind: SnippetKind::Win,
            function: func.clone(),
            start_line: 5, // overlaps with snip_a (1-10) → merges into (1-15)
            end_line: 15,
            text: "block1\nblock2\nblock3".into(),
            display_text: "block1\nblock2\nblock3".into(),
            snippet_hash: "b".into(),
        };
        let finding = Finding {
            function_a: func.clone(),
            function_b: func,
            score: 0.95,
            duplicated_lines: 10,
            evidence: vec![CandidateMatch {
                snippet_a: snip_a,
                snippet_b: snip_b,
                similarity: 0.95,
                evidence: "".into(),
            }],
            reasons: vec!["self_clone".into()],
            metadata: BTreeMap::new(),
        };
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.html").to_string_lossy().into_owned();
        write_html(&make_result(vec![finding]), &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        // Overlapping spans merge into (1-15); displaying (1-10) vs (5-15) leaves
        // 5 hidden-after lines on side A and 4 hidden-before lines on side B.
        assert!(content.contains("lines not shown"));
    }

    #[test]
    fn merge_spans_adjacent_combines_touching() {
        // Adjacent: (1,5) and (6,10) → merged because 6 <= 5+1
        let merged = merge_spans_adjacent(&[(1, 5), (6, 10)]);
        assert_eq!(merged.len(), 1, "adjacent spans must be merged");
        assert_eq!(merged[0], (1, 10));
    }

    #[test]
    fn merge_spans_adjacent_separates_gap() {
        // Gap: (1,5) and (7,10) → NOT merged because 7 > 5+1
        let merged = merge_spans_adjacent(&[(1, 5), (7, 10)]);
        assert_eq!(merged.len(), 2, "spans with gap must not be merged");
    }
}
