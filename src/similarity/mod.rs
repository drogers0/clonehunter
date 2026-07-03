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
// Consumed by T11 reporters
pub(crate) use occurrences::{SelfCloneOccurrences, is_self_clone};
pub(crate) use ranking::best_match;
// Consumed by T14 (test port); keep targeted allow until then
#[allow(unused_imports)]
pub(crate) use lexical::lexical_similarity;
#[allow(unused_imports)]
pub(crate) use ranking::kind_rank;
#[allow(unused_imports)]
pub(crate) use scoring::best_score;
