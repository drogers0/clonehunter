# 1 · Concepts & Glossary

Before the mechanics, the mental model. This page defines the handful of ideas that
the rest of the manual assumes.

## What "clone" means here

A **clone** is not a byte-for-byte copy. CloneHunter looks for code that is
*semantically* similar — code that does the same thing, even if variables are
renamed, lines are reordered, or comments differ. Two functions that compute a
running average with different variable names are clones; two functions that merely
share a few keywords are not.

To judge that, CloneHunter combines two independent signals:

- **Semantic similarity** — a transformer model (CodeBERT) turns each piece of code
  into a 768-number vector ("embedding"). Code that means the same thing lands close
  together in that vector space, regardless of surface wording. Closeness is measured
  by cosine similarity.
- **Lexical similarity** — a plain overlap of identifier tokens (Jaccard: shared
  tokens ÷ total distinct tokens). This is a sanity check: the model can be fooled
  into thinking two unrelated snippets are close, and requiring real token overlap
  keeps those false matches out.

The two are blended into one **composite score**. A pair must clear both a lexical
floor *and* a composite threshold to survive. See
[Detection internals](03-detection.md) for the exact arithmetic.

## Why a pipeline, not a matcher

You cannot embed a whole file and compare files — a duplicated *helper* buried in a
1000-line module would be drowned out. So CloneHunter first chops code into small,
comparable **snippets**, embeds each one, and finds duplicates at the snippet level.
It then **rolls up** all the snippet-level matches between the same two functions
into a single **finding**. One finding = one pair of functions worth looking at,
carrying all the evidence that linked them.

## The three kinds of snippet

Every snippet has a `kind`. The kind determines which threshold applies and how much
weight it carries as evidence.

| Kind | Stands for | Produced from | Purpose |
|------|-----------|---------------|---------|
| **FUNC** | Function | One per Python function | Catches whole-function duplication. The strongest evidence. |
| **WIN** | Window | Sliding windows over *every* file (Python functions **and** whole non-Python files) | Catches partial duplication and is the *only* signal for non-Python code. |
| **EXP** | Expansion | Python functions with their called helpers inlined (opt-in) | Catches duplication that only appears once you follow the call graph. Off by default. |

Windows are how CloneHunter handles languages it cannot parse into functions: a
`.js` or `.go` file becomes one big unit that is sliced into overlapping windows.
Python additionally gets FUNC snippets because it is parsed into an AST.

## Analysis text vs display text

Each snippet carries **two** versions of its code, and confusing them causes bugs:

- **Analysis text** — comments stripped out. This is what gets embedded, hashed,
  tokenized, and diffed. Detection operates *entirely* on analysis text, so a
  comment change never affects whether something is flagged.
- **Display text** — comments preserved. This is what the HTML report shows a human,
  so the rendered diff looks like the real source.

## Glossary

Terms used throughout the manual and the code:

- **Snippet** (`SnippetRef`) — a chunk of analysis text with its kind, source
  span, and a `snippet_hash`. The unit of embedding and matching.
- **Embedding** (`Embedding`) — the 768-float vector for one snippet.
- **Candidate** (`CandidateMatch`) — a surviving snippet-to-snippet pair: its two
  snippets, the composite similarity, and an evidence string.
- **Finding** (`Finding`) — the rolled-up result for one pair of functions: the two
  functions, a headline score, the count of duplicated lines, the list of candidate
  matches that back it, and the *reasons* it qualified.
- **Reason** — *why* a finding was emitted: `func_threshold`, `exp_threshold`, or
  `min_window_hits`. A finding needs at least one.
- **Composite score** — the blend of embedding and lexical similarity that a threshold
  is compared against (exact formula in [chapter 3](03-detection.md#the-composite-score)).
- **Degradation** — a recorded fallback (GPU→CPU, cache self-heal, skipped file)
  that is surfaced in the report instead of crashing the scan.
- **Baseline** — `benchmark/baseline.json`, a checked-in record of the expected
  findings and scores for a set of sample repos; used as a regression fixture to catch
  accidental changes in detection output.

Next: [the pipeline](02-pipeline.md) that ties all of this together.
