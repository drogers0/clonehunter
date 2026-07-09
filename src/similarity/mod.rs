mod candidates;
mod clustering;
mod lexical;
mod occurrences;
mod ranking;
mod rollup;
mod scoring;

pub(crate) use candidates::retrieve_candidates;
pub(crate) use clustering::{CloneGroup, build_groups, group_stats};
pub(crate) use occurrences::{SelfCloneOccurrences, is_self_clone};
pub(crate) use ranking::best_match;
pub(crate) use rollup::rollup_findings;
