#![allow(dead_code, unused_imports)] // T10 (pipeline) will wire these up
mod candidates;
mod clustering;
mod lexical;
mod occurrences;
mod ranking;
mod rollup;
mod scoring;

pub(crate) use candidates::retrieve_candidates;
pub(crate) use clustering::{cluster_findings, filter_clusters};
pub(crate) use lexical::lexical_similarity;
pub(crate) use occurrences::{SelfCloneOccurrences, is_self_clone};
pub(crate) use ranking::{best_match, kind_rank};
pub(crate) use rollup::rollup_findings;
pub(crate) use scoring::best_score;
