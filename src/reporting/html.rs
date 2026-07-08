use std::fs::File;
use std::io::{BufWriter, Write as IoWrite};

use similar::TextDiff;

use crate::core::types::{CandidateMatch, Degradation, Finding, ScanResult};
use crate::reporting::ReportError;
use crate::reporting::compare::{CompareData, select_compare};
use crate::reporting::schema::SCHEMA_VERSION;
use crate::similarity::{
    CloneGroup, SelfCloneOccurrences, best_match, build_groups, is_self_clone,
};

pub(crate) fn write_html(result: &ScanResult, out_path: &str) -> Result<(), ReportError> {
    // Show group chrome only when clustering ran (DD6); otherwise the flat per-finding list,
    // unchanged. A clustered run gets a summary line and one outer <details> per multi-finding
    // group; 1-finding groups render as plain cards.
    let clustered = result
        .findings
        .iter()
        .any(|f| f.metadata.contains_key("cluster_id"));
    let (rows, summary_html) = if clustered {
        let groups = build_groups(&result.findings);
        // Both counts come from stats (populated once via group_stats) so the summary can't drift
        // between the two numbers; group_count is guaranteed equal to groups.len() here.
        let summary = format!(
            r#"  <p>{} clone groups across {} functions</p>"#,
            result.stats.group_count, result.stats.grouped_function_count
        );
        let rows: Vec<String> = groups
            .iter()
            .map(|g| render_group(g, &result.findings))
            .collect();
        (rows, summary)
    } else {
        let rows: Vec<String> = result.findings.iter().map(render_finding).collect();
        (rows, String::new())
    };
    let rows_html = rows.join("\n");
    let finding_count = result.findings.len();
    let degradations_html = render_degradations(&result.degradations);
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
    .degradations {{ background: #fff5b1; border: 1px solid #e0c000; border-radius: 6px;
      padding: 8px 12px; margin-bottom: 12px; }}
    .degradations ul {{ margin: 4px 0 0; padding-left: 20px; }}
    .group-locations {{ margin: 8px 0; padding-left: 20px; color: #444; font-size: 0.9em;
      word-break: break-all; }}
  </style>
</head>
<body>
  <h1>CloneHunter Report</h1>
  <p>Schema: {SCHEMA_VERSION}</p>
  <p>Findings: {finding_count}</p>
{summary_html}
  {degradations_html}
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
        // Reorder only TOP-LEVEL items (group in clustered mode, finding in flat mode); a
        // descendant selector would rip nested member findings out of their group on first paint.
        const items = Array.from(list.children).filter((el) => el.tagName === "DETAILS");
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

/// Render a warning banner listing graceful-degradation events. Empty string when there are none.
fn render_degradations(degradations: &[Degradation]) -> String {
    if degradations.is_empty() {
        return String::new();
    }
    let mut items = String::new();
    for d in degradations {
        use std::fmt::Write as _;
        let _ = write!(items, "<li>{:?}: {}</li>", d.kind, html_escape(&d.message));
    }
    format!(r#"<div class="degradations"><strong>⚠ Degradations</strong><ul>{items}</ul></div>"#)
}

/// Render one clone group (DD6). A 1-finding group renders as a plain card (no wrapper); a
/// multi-finding group is an outer `<details>` listing every location, with member findings
/// rendered inside via the existing `render_finding`.
fn render_group(group: &CloneGroup, findings: &[Finding]) -> String {
    if group.finding_indices.len() == 1 {
        return render_finding(&findings[group.finding_indices[0]]);
    }
    // Case-insensitive minimum of the group's location paths — the same key `render_finding`
    // uses per finding, so grouped/flat "File path A→Z" sorting stay consistent.
    let path_min = group
        .locations
        .iter()
        .map(|f| &f.file.path)
        .min_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()))
        .expect("group has at least one location");
    let mut locations_html = String::new();
    for f in &group.locations {
        use std::fmt::Write as _;
        let _ = write!(
            locations_html,
            "<li>{}:{}-{}</li>",
            html_escape(&f.file.path),
            f.start_line,
            f.end_line
        );
    }
    let members_html: String = group
        .finding_indices
        .iter()
        .map(|&i| render_finding(&findings[i]))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"
<details
  data-path-min="{path_min_escaped}"
  data-score="{score}"
  data-lines="{lines}"
>
  <summary>Clone group #{id} — {n} locations</summary>
  <ul class="group-locations">{locations_html}</ul>
  {members_html}
</details>
"#,
        path_min_escaped = html_escape(path_min),
        score = group.max_score,
        lines = group.max_duplicated_lines,
        id = group.id,
        n = group.locations.len(),
    )
}

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
        .replace('\'', "&#x27;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::core::types::{
        CandidateMatch, FileRef, Finding, FunctionRef, Language, SnippetKind, SnippetRef,
    };
    use crate::test_support::{
        make_finding, make_function, make_match, make_scan_result, make_snippet_for,
    };
    use tempfile::TempDir;

    fn make_result(findings: Vec<Finding>) -> ScanResult {
        make_scan_result(findings)
    }

    #[test]
    fn render_degradations_empty_is_blank() {
        assert_eq!(render_degradations(&[]), "");
    }

    #[test]
    fn render_degradations_lists_events_escaped() {
        use crate::core::types::{Degradation, DegradationKind};
        let html = render_degradations(&[Degradation {
            kind: DegradationKind::DeviceFallback,
            message: "CUDA <unavailable> & slow".into(),
        }]);
        assert!(html.contains("class=\"degradations\""));
        assert!(html.contains("DeviceFallback"));
        // message is HTML-escaped
        assert!(html.contains("CUDA &lt;unavailable&gt; &amp; slow"));
    }

    fn make_finding_with_code(code_a: &str, code_b: &str) -> Finding {
        let func_a = make_function("a.py", "foo", 1, 10, code_a);
        let func_b = make_function("a.py", "bar", 20, 30, code_b);
        let m = make_match(
            make_snippet_for(&func_a, code_a),
            make_snippet_for(&func_b, code_b),
            0.95,
        );
        make_finding(func_a, func_b, 0.95, 10, vec![m], &["test"])
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
        assert!(!escaped.contains('\''), "apostrophe must be escaped");
        assert!(escaped.contains("&lt;script&gt;"));
        assert!(escaped.contains("&#x27;"));
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
            content: "".into(),
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
    fn html_clustered_shows_group_chrome() {
        // Two findings sharing "foo" → one clustered group of 3 locations rendered with chrome.
        use crate::similarity::{cluster_findings, group_stats};
        let foo = make_function("a.py", "foo", 1, 10, "def foo(): pass");
        let bar = make_function("b.py", "bar", 1, 10, "def bar(): pass");
        let baz = make_function("c.py", "baz", 1, 10, "def baz(): pass");
        let mk = |x: &FunctionRef, y: &FunctionRef| {
            make_finding(
                x.clone(),
                y.clone(),
                0.95,
                10,
                vec![make_match(
                    make_snippet_for(x, "def f(): pass"),
                    make_snippet_for(y, "def f(): pass"),
                    0.95,
                )],
                &["func"],
            )
        };
        let clustered = cluster_findings(&[mk(&foo, &bar), mk(&foo, &baz)]);
        // Populate stats exactly as the pipeline would, so the summary renders real numbers.
        let (group_count, grouped_function_count) = group_stats(&clustered);
        let mut result = make_result(clustered);
        result.stats.group_count = group_count;
        result.stats.grouped_function_count = grouped_function_count;

        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.html").to_string_lossy().into_owned();
        write_html(&result, &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        // Exact summary numbers: 1 group across 3 unique functions (not just substring presence).
        assert!(
            content.contains("1 clone groups across 3 functions"),
            "summary must report the real group/function counts"
        );
        assert!(content.contains("Clone group #1"), "group header present");
        assert!(content.contains("3 locations"), "lists all 3 locations");
        // All three member paths appear in the locations list.
        assert!(content.contains("a.py:1-10"));
        assert!(content.contains("b.py:1-10"));
        assert!(content.contains("c.py:1-10"));
        // Member diffs still render (each finding is a nested <details>).
        assert!(content.contains("class=\"diff"));
    }

    #[test]
    fn html_unclustered_has_no_group_chrome() {
        let finding =
            make_finding_with_code("def foo():\n    return 1", "def bar():\n    return 1");
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.html").to_string_lossy().into_owned();
        write_html(&make_result(vec![finding]), &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        assert!(
            !content.contains("clone groups across"),
            "no summary line without clustering"
        );
        assert!(
            !content.contains("Clone group #"),
            "no group wrapper without clustering"
        );
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

    // ── DD9 edge-case ports ──────────────────────────────────────────────────

    /// Port of Python test_html_reporter_marks_hidden_duplicated_lines.
    /// Three WIN pairs over a 60-line function: before(1-10), mid(20-30), after(40-50).
    /// Evidence union spans 1-50 on each side.  best_match picks mid (highest sim=0.99).
    /// hidden_before=10 (lines 1-10 before the rendered mid window) → "<10 lines not shown>".
    /// hidden_after=11 (lines 40-50 after mid window) → "<11 lines not shown>".
    /// Each marker appears exactly twice (once per column in the hidden-line row).
    #[test]
    fn marks_hidden_duplicated_lines() {
        use crate::core::config::Thresholds;
        use crate::similarity::rollup_findings;

        let file_a = FileRef {
            path: "a.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
            content: "".into(),
        };
        let file_b = FileRef {
            path: "b.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
            content: "".into(),
        };
        let code_a: String = (1usize..=60)
            .map(|i| format!("a{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let code_b: String = (1usize..=60)
            .map(|i| format!("b{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let fn_a = FunctionRef {
            file: file_a,
            qualified_name: "a".into(),
            start_line: 1,
            end_line: 60,
            code: code_a,
            code_hash: "ca".into(),
        };
        let fn_b = FunctionRef {
            file: file_b,
            qualified_name: "b".into(),
            start_line: 1,
            end_line: 60,
            code: code_b,
            code_hash: "cb".into(),
        };

        let make_win =
            |func: &FunctionRef, start: usize, end: usize, hash: &str, text: &str| SnippetRef {
                kind: SnippetKind::Win,
                function: func.clone(),
                start_line: start,
                end_line: end,
                text: text.to_string(),
                display_text: text.to_string(),
                snippet_hash: hash.to_string(),
            };

        let before_a = make_win(&fn_a, 1, 10, "ba", "pre");
        let before_b = make_win(&fn_b, 1, 10, "bb", "pre");
        let mid_a = make_win(&fn_a, 20, 30, "ma", "mid");
        let mid_b = make_win(&fn_b, 20, 30, "mb", "mid");
        let after_a = make_win(&fn_a, 40, 50, "aa", "post");
        let after_b = make_win(&fn_b, 40, 50, "ab", "post");

        let thresholds = Thresholds {
            func: 0.9,
            win: 0.9,
            exp: 0.9,
            min_window_hits: 3, // 3 WIN pairs → "min_window_hits" reason
            lexical_min_ratio: 0.0,
            lexical_weight: 0.3,
        };
        let matches = vec![
            CandidateMatch {
                snippet_a: before_a,
                snippet_b: before_b,
                similarity: 0.91,
                evidence: "".into(),
            },
            CandidateMatch {
                snippet_a: mid_a,
                snippet_b: mid_b,
                similarity: 0.99,
                evidence: "".into(),
            },
            CandidateMatch {
                snippet_a: after_a,
                snippet_b: after_b,
                similarity: 0.92,
                evidence: "".into(),
            },
        ];
        let findings = rollup_findings(matches, &thresholds);
        assert_eq!(findings.len(), 1, "expected 1 finding");

        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.html").to_string_lossy().into_owned();
        write_html(&make_result(findings), &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();

        // Evidence bounds = union of all WIN spans → 1-50 for both sides.
        assert!(content.contains("a.py:1-50"), "header must show a.py:1-50");
        assert!(content.contains("b.py:1-50"), "header must show b.py:1-50");
        // Before the rendered mid window (1-10 = 10 lines); after (40-50 = 11 lines).
        assert_eq!(
            content.matches("&lt;10 lines not shown&gt;").count(),
            2,
            "<10 lines not shown> must appear exactly twice"
        );
        assert_eq!(
            content.matches("&lt;11 lines not shown&gt;").count(),
            2,
            "<11 lines not shown> must appear exactly twice"
        );
    }

    /// Port of Python test_html_reporter_diff_numbers_survive_blank_line_stripping.
    /// Code "line1\n\nline3\nline4" starting at line 10 (A) / 50 (B).
    /// After blank-line stripping, "line3" retains its true offset (i=2 from start):
    /// absolute line 12 for A, 52 for B — not the running post-strip index (11/51).
    #[test]
    fn diff_numbers_survive_blank_line_stripping() {
        let code = "line1\n\nline3\nline4";
        let file_a = FileRef {
            path: "a.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
            content: "".into(),
        };
        let file_b = FileRef {
            path: "b.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
            content: "".into(),
        };
        let fn_a = FunctionRef {
            file: file_a,
            qualified_name: "a".into(),
            start_line: 10,
            end_line: 13,
            code: code.into(),
            code_hash: "a".into(),
        };
        let fn_b = FunctionRef {
            file: file_b,
            qualified_name: "b".into(),
            start_line: 50,
            end_line: 53,
            code: code.into(),
            code_hash: "b".into(),
        };
        let snip_a = SnippetRef {
            kind: SnippetKind::Func,
            function: fn_a.clone(),
            start_line: 10,
            end_line: 13,
            text: code.into(),
            display_text: code.into(),
            snippet_hash: "sa".into(),
        };
        let snip_b = SnippetRef {
            kind: SnippetKind::Func,
            function: fn_b.clone(),
            start_line: 50,
            end_line: 53,
            text: code.into(),
            display_text: code.into(),
            snippet_hash: "sb".into(),
        };
        let finding = Finding {
            function_a: fn_a,
            function_b: fn_b,
            score: 1.0,
            duplicated_lines: 4,
            evidence: vec![CandidateMatch {
                snippet_a: snip_a,
                snippet_b: snip_b,
                similarity: 1.0,
                evidence: "".into(),
            }],
            reasons: vec!["func".into()],
            metadata: BTreeMap::new(),
        };

        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.html").to_string_lossy().into_owned();
        write_html(&make_result(vec![finding]), &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();

        // "line3" is at enumerate offset 2 from start=10 → absolute line 12 (A), 52 (B).
        assert!(
            content.contains(">12</td>"),
            "line3 must carry line number 12"
        );
        assert!(
            content.contains(">52</td>"),
            "line3 must carry line number 52"
        );
        assert!(
            !content.contains(">11</td>"),
            "running-offset 11 must not appear"
        );
        assert!(
            !content.contains(">51</td>"),
            "running-offset 51 must not appear"
        );
    }

    /// Port of Python test_html_reporter_self_clone_evidence_bounds_disjoint.
    /// Two overlapping WIN pairs within the same function.
    /// After rollup+SelfCloneOccurrences, span_a=(1800,1825) and span_b=(2100,2125).
    #[test]
    fn self_clone_evidence_bounds_disjoint() {
        use crate::core::config::Thresholds;
        use crate::similarity::rollup_findings;

        let file = FileRef {
            path: "app/service.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
            content: "".into(),
        };
        let func = FunctionRef {
            file,
            qualified_name: "f".into(),
            start_line: 1,
            end_line: 3000,
            code: "pass".into(),
            code_hash: "c".into(),
        };

        let make_win = |start: usize, end: usize, hash: &str| SnippetRef {
            kind: SnippetKind::Win,
            function: func.clone(),
            start_line: start,
            end_line: end,
            text: "s".into(),
            display_text: "s".into(),
            snippet_hash: hash.to_string(),
        };

        let early_1 = make_win(1800, 1820, "w1");
        let late_1 = make_win(2100, 2120, "w2");
        let late_2 = make_win(2105, 2125, "w3");
        let early_2 = make_win(1805, 1825, "w4");

        let thresholds = Thresholds {
            func: 0.9,
            win: 0.9,
            exp: 0.9,
            min_window_hits: 2,
            lexical_min_ratio: 0.0,
            lexical_weight: 0.3,
        };
        let matches = vec![
            CandidateMatch {
                snippet_a: early_1,
                snippet_b: late_1,
                similarity: 0.9,
                evidence: "".into(),
            },
            CandidateMatch {
                snippet_a: late_2,
                snippet_b: early_2,
                similarity: 0.9,
                evidence: "".into(),
            },
        ];
        let findings = rollup_findings(matches, &thresholds);
        assert!(!findings.is_empty(), "expected at least one finding");

        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.html").to_string_lossy().into_owned();
        write_html(&make_result(findings), &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();

        // SelfCloneOccurrences merges (1800-1820)+(1805-1825)→(1800,1825)
        // and (2100-2120)+(2105-2125)→(2100,2125).
        assert!(
            content.contains("app/service.py:1800-1825"),
            "span A must be 1800-1825"
        );
        assert!(
            content.contains("app/service.py:2100-2125"),
            "span B must be 2100-2125"
        );
    }

    /// Port of Python test_html_reporter_nway_chained_self_clone_bounds_disjoint.
    /// Three chained occurrences: r1(1000-1020)↔r2(2000-2020) and r3(2900-2920)↔r2b(2005-2025).
    /// best_match selects r1↔r2 (sim=0.95 > 0.90).
    /// Header must show (1000-1020) and (2000-2025) — NOT the min/max envelope (1000-2025).
    #[test]
    fn nway_chained_self_clone_bounds_disjoint() {
        use crate::core::config::Thresholds;
        use crate::similarity::rollup_findings;

        let file = FileRef {
            path: "app/nway.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
            content: "".into(),
        };
        let func = FunctionRef {
            file,
            qualified_name: "f".into(),
            start_line: 1,
            end_line: 3000,
            code: "pass".into(),
            code_hash: "c".into(),
        };

        let make_win = |start: usize, end: usize, hash: &str| SnippetRef {
            kind: SnippetKind::Win,
            function: func.clone(),
            start_line: start,
            end_line: end,
            text: "s".into(),
            display_text: "s".into(),
            snippet_hash: hash.to_string(),
        };

        let r1 = make_win(1000, 1020, "r1");
        let r2 = make_win(2000, 2020, "r2");
        let r3 = make_win(2900, 2920, "r3");
        let r2b = make_win(2005, 2025, "r2b");

        let thresholds = Thresholds {
            func: 0.9,
            win: 0.9,
            exp: 0.9,
            min_window_hits: 2,
            lexical_min_ratio: 0.0,
            lexical_weight: 0.3,
        };
        let matches = vec![
            CandidateMatch {
                snippet_a: r1,
                snippet_b: r2,
                similarity: 0.95,
                evidence: "".into(),
            },
            CandidateMatch {
                snippet_a: r3,
                snippet_b: r2b,
                similarity: 0.90,
                evidence: "".into(),
            },
        ];
        let findings = rollup_findings(matches, &thresholds);
        assert_eq!(findings.len(), 1, "expected exactly one finding");

        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.html").to_string_lossy().into_owned();
        write_html(&make_result(findings), &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();

        // best_match = r1↔r2 (sim=0.95). occurrence_for(1000,1020)→(1000,1020);
        // occurrence_for(2000,2020)→(2000,2025) (merged with r2b 2005-2025).
        assert!(
            content.contains("app/nway.py:1000-1020"),
            "span A must be 1000-1020"
        );
        assert!(
            content.contains("app/nway.py:2000-2025"),
            "span B must be 2000-2025 (r2 merged with r2b)"
        );
        assert!(
            !content.contains("app/nway.py:1000-2025"),
            "must NOT show min/max envelope 1000-2025"
        );
    }
}
