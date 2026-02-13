from __future__ import annotations

import difflib
import html
from typing import Any, cast

from clonehunter.core.types import CandidateMatch, Finding, ScanResult
from clonehunter.reporting.compare import CompareData, select_compare
from clonehunter.reporting.schema import SCHEMA_VERSION


class HtmlReporter:
    def write(self, result: ScanResult, out_path: str) -> None:
        rows = "\n".join(_render_finding(finding) for finding in result.findings)
        html = f"""
<!doctype html>
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
  <p>Findings: {len(result.findings)}</p>
  <div class="controls">
    <label for="sort-findings">Sort findings:</label>
    <select id="sort-findings">
      <option value="lines_desc">Duplicated lines (high to low)</option>
      <option value="score_desc">Match score (high to low)</option>
      <option value="path_asc">File path (A/B, A to Z)</option>
    </select>
  </div>
  <div class="list">
    {rows}
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
"""
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(html)


def _render_finding(finding: Finding) -> str:
    func_a = finding.function_a
    func_b = finding.function_b
    span_a, span_b = _evidence_bounds(finding.evidence)
    compare = _select_compare(finding.evidence)
    diff_html = _render_diff(compare)
    path_min = min(func_a.file.path, func_b.file.path, key=str.casefold)
    return f"""
<details
  data-path-min="{html.escape(path_min)}"
  data-score="{finding.score}"
  data-lines="{finding.duplicated_lines}"
>
  <summary>
    <div class="summary-grid">
      <div>
        <div>{html.escape(func_a.qualified_name)}</div>
        <div class="path">
          {html.escape(func_a.file.path)}:{span_a[0]}-{span_a[1]}
        </div>
      </div>
      <div>
        <div>{html.escape(func_b.qualified_name)}</div>
        <div class="path">
          {html.escape(func_b.file.path)}:{span_b[0]}-{span_b[1]}
        </div>
      </div>
      <div>{finding.score:.3f}</div>
      <div>{finding.duplicated_lines} duplicated lines</div>
    </div>
  </summary>
  {diff_html}
</details>
"""


def _select_compare(matches: list[CandidateMatch]) -> dict[str, object] | None:
    compare = select_compare(matches)
    if compare is None:
        return None
    return _compare_payload(compare, matches)


def _compare_payload(compare: CompareData, matches: list[CandidateMatch]) -> dict[str, object]:
    hidden_before_a, hidden_before_b, hidden_after_a, hidden_after_b = _hidden_duplicated_lines(
        matches=matches, span_a=compare.span_a, span_b=compare.span_b
    )
    return {
        "kind_a": compare.kind_a,
        "kind_b": compare.kind_b,
        "span_a": compare.span_a,
        "span_b": compare.span_b,
        "similarity": compare.similarity,
        "text_a": compare.text_a,
        "text_b": compare.text_b,
        "hidden_before_a": hidden_before_a,
        "hidden_before_b": hidden_before_b,
        "hidden_after_a": hidden_after_a,
        "hidden_after_b": hidden_after_b,
    }


def _render_diff(compare: dict[str, object] | None) -> str:
    if not compare:
        return '<div class="code-box">No diff available.</div>'
    text_a = str(compare.get("text_a", ""))
    text_b = str(compare.get("text_b", ""))
    span_a = _as_span(compare.get("span_a"))
    span_b = _as_span(compare.get("span_b"))
    lines_a = _strip_blank_lines(text_a.splitlines())
    lines_b = _strip_blank_lines(text_b.splitlines())
    table = _render_side_by_side(
        lines_a=lines_a,
        lines_b=lines_b,
        start_a=span_a[0],
        start_b=span_b[0],
        hidden_before_a=_as_int(compare.get("hidden_before_a", 0)),
        hidden_before_b=_as_int(compare.get("hidden_before_b", 0)),
        hidden_after_a=_as_int(compare.get("hidden_after_a", 0)),
        hidden_after_b=_as_int(compare.get("hidden_after_b", 0)),
    )
    return f'<div class="diff-wrap">{table}</div>'


def _render_side_by_side(
    lines_a: list[str],
    lines_b: list[str],
    start_a: int,
    start_b: int,
    hidden_before_a: int,
    hidden_before_b: int,
    hidden_after_a: int,
    hidden_after_b: int,
) -> str:
    matcher = difflib.SequenceMatcher(a=lines_a, b=lines_b)
    rows: list[str] = []
    top_row = _render_hidden_row(hidden_before_a, hidden_before_b)
    if top_row:
        rows.append(top_row)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(max(i2 - i1, j2 - j1)):
                a_line = lines_a[i1 + offset]
                b_line = lines_b[j1 + offset]
                rows.append(
                    _render_row(start_a + i1 + offset, a_line, start_b + j1 + offset, b_line, "")
                )
        elif tag == "replace":
            count = max(i2 - i1, j2 - j1)
            for offset in range(count):
                a_line = lines_a[i1 + offset] if i1 + offset < i2 else ""
                b_line = lines_b[j1 + offset] if j1 + offset < j2 else ""
                a_no = start_a + i1 + offset if i1 + offset < i2 else ""
                b_no = start_b + j1 + offset if j1 + offset < j2 else ""
                rows.append(_render_row(a_no, a_line, b_no, b_line, "diff_chg"))
        elif tag == "delete":
            for offset in range(i1, i2):
                rows.append(_render_row(start_a + offset, lines_a[offset], "", "", "diff_sub"))
        elif tag == "insert":
            for offset in range(j1, j2):
                rows.append(_render_row("", "", start_b + offset, lines_b[offset], "diff_add"))
    bottom_row = _render_hidden_row(hidden_after_a, hidden_after_b)
    if bottom_row:
        rows.append(bottom_row)
    header = (
        '<table class="diff">'
        "<colgroup>"
        '<col style="width:3.5em" />'
        '<col style="width:calc((100% - 7em) / 2)" />'
        '<col style="width:3.5em" />'
        '<col style="width:calc((100% - 7em) / 2)" />'
        "</colgroup>"
        "<thead><tr>"
        '<th class="line-no"></th><th>Function A</th>'
        '<th class="line-no"></th><th>Function B</th>'
        "</tr></thead><tbody>"
    )
    footer = "</tbody></table>"
    return header + "".join(rows) + footer


def _render_row(a_no: int | str, a_line: str, b_no: int | str, b_line: str, cls: str) -> str:
    a_no_text = html.escape(str(a_no)) if a_no != "" else ""
    b_no_text = html.escape(str(b_no)) if b_no != "" else ""
    a_text = html.escape(a_line)
    b_text = html.escape(b_line)
    class_attr = f" {cls}" if cls else ""
    return (
        f"<tr>"
        f'<td class="line-no{class_attr}">{a_no_text}</td>'
        f'<td class="code{class_attr}">{a_text}</td>'
        f'<td class="line-no{class_attr}">{b_no_text}</td>'
        f'<td class="code{class_attr}">{b_text}</td>'
        f"</tr>"
    )


def _strip_blank_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.strip()]


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _as_span(value: object) -> tuple[int, int]:
    if not isinstance(value, dict):
        return (1, 1)
    span = cast(dict[str, Any], value)
    start = _as_int(span.get("start_line"))
    end = _as_int(span.get("end_line"))
    if start <= 0:
        start = 1
    if end < start:
        end = start
    return (start, end)


def _render_hidden_row(line_count_a: int, line_count_b: int) -> str:
    if line_count_a <= 0 and line_count_b <= 0:
        return ""
    marker_a = html.escape(f"<{line_count_a} lines not shown>") if line_count_a > 0 else ""
    marker_b = html.escape(f"<{line_count_b} lines not shown>") if line_count_b > 0 else ""
    return (
        "<tr>"
        '<td class="line-no"></td>'
        f'<td class="meta">{marker_a}</td>'
        '<td class="line-no"></td>'
        f'<td class="meta">{marker_b}</td>'
        "</tr>"
    )


def _hidden_duplicated_lines(
    matches: list[CandidateMatch], span_a: dict[str, int], span_b: dict[str, int]
) -> tuple[int, int, int, int]:
    spans_a = [(m.snippet_a.start_line, m.snippet_a.end_line) for m in matches]
    spans_b = [(m.snippet_b.start_line, m.snippet_b.end_line) for m in matches]
    before_a = _covered_in_range(spans_a, 1, span_a["start_line"] - 1)
    before_b = _covered_in_range(spans_b, 1, span_b["start_line"] - 1)
    after_a = _covered_in_range(spans_a, span_a["end_line"] + 1, 1_000_000_000)
    after_b = _covered_in_range(spans_b, span_b["end_line"] + 1, 1_000_000_000)
    return before_a, before_b, after_a, after_b


def _covered_in_range(spans: list[tuple[int, int]], start: int, end: int) -> int:
    if start > end:
        return 0
    covered = 0
    for span_start, span_end in _merge_spans(spans):
        overlap_start = max(start, span_start)
        overlap_end = min(end, span_end)
        if overlap_start <= overlap_end:
            covered += overlap_end - overlap_start + 1
    return covered


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if end > prev_end:
            merged[-1] = (prev_start, end)
    return merged


def _evidence_bounds(matches: list[CandidateMatch]) -> tuple[tuple[int, int], tuple[int, int]]:
    if not matches:
        return (1, 1), (1, 1)
    starts_a = [m.snippet_a.start_line for m in matches]
    ends_a = [m.snippet_a.end_line for m in matches]
    starts_b = [m.snippet_b.start_line for m in matches]
    ends_b = [m.snippet_b.end_line for m in matches]
    return (min(starts_a), max(ends_a)), (min(starts_b), max(ends_b))
