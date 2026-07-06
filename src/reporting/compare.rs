use crate::core::types::{CandidateMatch, SnippetKind};
use crate::similarity::best_match;

/// Evidence data for the best-match snippet pair, carrying both text forms.
///
/// `text_a`/`text_b` — analysis text (docstrings+comments stripped): used for unified diff
/// in the JSON reporter (schema parity with Python).
///
/// `display_text_a`/`display_text_b` — display text (comments preserved): used for
/// side-by-side diff in the HTML reporter (human readability).
pub(crate) struct CompareData {
    pub kind_a: SnippetKind,
    pub kind_b: SnippetKind,
    /// (start_line, end_line) — serialized as `{"start_line": N, "end_line": N}` in JSON.
    pub span_a: (usize, usize),
    pub span_b: (usize, usize),
    pub similarity: f64,
    pub text_a: String,
    pub text_b: String,
    pub display_text_a: String,
    pub display_text_b: String,
}

/// Select the best-match evidence pair from a finding's evidence list.
/// Returns `None` if `matches` is empty.
pub(crate) fn select_compare(matches: &[CandidateMatch]) -> Option<CompareData> {
    let best = best_match(matches)?;
    Some(CompareData {
        kind_a: best.snippet_a.kind,
        kind_b: best.snippet_b.kind,
        span_a: (best.snippet_a.start_line, best.snippet_a.end_line),
        span_b: (best.snippet_b.start_line, best.snippet_b.end_line),
        similarity: best.similarity,
        text_a: best.snippet_a.text.clone(),
        text_b: best.snippet_b.text.clone(),
        display_text_a: best.snippet_a.display_text.clone(),
        display_text_b: best.snippet_b.display_text.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::SnippetRef;
    use crate::test_support::{make_function, make_match, make_snippet_kind};

    fn make_snip(
        kind: SnippetKind,
        start: usize,
        end: usize,
        text: &str,
        disp: &str,
    ) -> SnippetRef {
        make_snippet_kind(
            kind,
            make_function("a.py", "f", start, end, "pass"),
            start,
            end,
            text,
            disp,
        )
    }

    #[test]
    fn select_compare_empty_returns_none() {
        assert!(select_compare(&[]).is_none());
    }

    #[test]
    fn select_compare_picks_best_match() {
        let sa = make_snip(SnippetKind::Func, 1, 10, "text_a", "disp_a");
        let sb = make_snip(SnippetKind::Win, 20, 30, "text_b", "disp_b");
        let m = make_match(sa, sb, 0.95);
        let cd = select_compare(&[m]).unwrap();
        assert_eq!(cd.kind_a, SnippetKind::Func);
        assert_eq!(cd.kind_b, SnippetKind::Win);
        assert_eq!(cd.span_a, (1, 10));
        assert_eq!(cd.span_b, (20, 30));
        assert!((cd.similarity - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn select_compare_carries_both_text_forms() {
        let sa = make_snip(SnippetKind::Func, 1, 5, "analysis_a", "display_a");
        let sb = make_snip(SnippetKind::Func, 10, 15, "analysis_b", "display_b");
        let m = make_match(sa, sb, 0.9);
        let cd = select_compare(&[m]).unwrap();
        assert_eq!(cd.text_a, "analysis_a");
        assert_eq!(cd.display_text_a, "display_a");
        assert_eq!(cd.text_b, "analysis_b");
        assert_eq!(cd.display_text_b, "display_b");
    }
}
