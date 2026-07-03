mod candidates;
mod clustering;
mod lexical;
mod occurrences;
mod ranking;
mod rollup;
mod scoring;

pub(crate) use candidates::retrieve_candidates;
pub(crate) use clustering::{cluster_findings, filter_clusters};
pub(crate) use rollup::rollup_findings;
// Re-exports below are consumed by T11 (reporters) and T14 (tests).
// allow(unused_imports) until T11 wires them from outside this module.
#[allow(unused_imports)]
pub(crate) use lexical::lexical_similarity;
#[allow(unused_imports)]
pub(crate) use occurrences::{SelfCloneOccurrences, is_self_clone};
#[allow(unused_imports)]
pub(crate) use ranking::{best_match, kind_rank};
#[allow(unused_imports)]
pub(crate) use scoring::best_score;
