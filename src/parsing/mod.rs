use tree_sitter::{Language as TsLanguage, Parser};

pub(crate) mod python_ast;
pub(crate) mod text_units;

/// Construct a tree-sitter parser configured for the Python grammar. Returns `None` only if
/// the bundled grammar fails to load (a build-time invariant — effectively unreachable).
pub(crate) fn make_python_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let language: TsLanguage = tree_sitter_python::LANGUAGE.into();
    parser.set_language(&language).ok()?;
    Some(parser)
}
