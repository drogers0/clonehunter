pub(crate) mod cli;
pub(crate) mod core;
pub(crate) mod embedding;
pub(crate) mod engines;
pub(crate) mod index;
pub(crate) mod io;
pub(crate) mod parsing;
pub(crate) mod reporting;
pub(crate) mod similarity;
pub(crate) mod snippets;

// Re-export only the public entry point needed by main.rs
pub use cli::run;
