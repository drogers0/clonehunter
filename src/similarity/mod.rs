mod candidates;
mod clustering;
mod lexical;
mod occurrences;
mod ranking;
mod rollup;
mod scoring;

pub(crate) use candidates::retrieve_candidates;
pub(crate) use clustering::{cluster_findings, filter_clusters};
pub(crate) use occurrences::{SelfCloneOccurrences, is_self_clone};
pub(crate) use ranking::best_match;
pub(crate) use rollup::rollup_findings;
