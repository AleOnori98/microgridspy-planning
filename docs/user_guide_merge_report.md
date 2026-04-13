# User Guide Merge Report

## Scope

This pass merged the existing Typical-Year and Multi-Year documentation tracks into a single LaTeX master guide:

- `docs/user_guide_full.tex`

The merge was done with Multi-Year as the reference workflow and Typical-Year rewritten as a reduced special case.

## Sections Merged

Shared workflow content now appears once in:

- Part I - Shared Platform Workflow
  - Introduction
  - General Workflow
  - Streamlit Interface

Full Multi-Year reference content now appears in:

- Part II - Multi-Year Planning Mode
  - Multi-Year Overview
  - Multi-Year Initialization Logic
  - Multi-Year Project Directory
  - Multi-Year Input Files
  - Multi-Year Time-Series Structure
  - Multi-Year Conditional Features
  - Multi-Year Consistency Checks
  - Multi-Year Minimal Examples

Reduced Typical-Year content now appears in:

- Part III - Typical-Year Planning Mode
  - Typical-Year Overview
  - Differences from Multi-Year

Appendix material now appears in:

- Part IV - Appendices
  - Glossary
  - UI Mapping
  - Source Basis

## Sections Removed or De-duplicated

The following content was intentionally defined only once in the unified guide:

- Streamlit page descriptions
- project directory structure
- primary YAML schema definitions
- primary CSV schema definitions
- conditional feature explanations

The Typical-Year section no longer repeats those shared definitions. It now references the Multi-Year chapters and lists only the reduced-formulation differences.

## Sections Rewritten

The main rewrites were:

- Typical-Year content refactored from a standalone guide into a "differences from Multi-Year" section
- Multi-Year markdown content converted into LaTeX sections, longtables, itemized rules, and `lstlisting` examples
- Shared workflow prose rewritten so it reads naturally before either formulation-specific part

## Remaining Ambiguities

These were not resolved in this merge pass and may still deserve human review before final PDF publication:

- some implementation details are still documented as legacy-compatible rather than fully removed, especially around battery degradation aliases and advanced curve semantics;
- the repo currently provides compiled PDFs but not the original LaTeX source used to generate the previous Typical-Year user guide, so the new master document was reconstructed in LaTeX rather than edited from the original source;
- `grid_availability.csv` remains a derived file stored under `inputs/`, which is practical but still somewhat ambiguous from a user-workflow perspective.

## Recommended Future Improvements

- compile and visually review `docs/user_guide_full.tex` to tune spacing, page breaks, and table widths;
- decide whether to keep the legacy-compatibility notes in the user-facing guide or move some of them to an internal appendix;
- add a short results-section appendix if a more formal external reporting glossary is desired;
- if the original LaTeX source for the prior guide is recovered later, compare typographic details and merge any project-specific macros or title-page conventions that should be preserved.
