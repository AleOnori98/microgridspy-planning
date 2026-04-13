# Multi-Year Guide Open Issues

This file tracks the main Multi-Year issues that still remain after the focused consistency cleanup.

## 1. Calendar-fade time increment still carries legacy `per_step` aliases

- **Where**: `core/data_pipeline/battery_degradation_model.py`, `core/data_pipeline/battery_calendar_fade_model.py`, legacy example inputs
- **Type**: legacy compatibility / naming drift
- **Severity**: medium
- **Problem**: the current UI writes `battery_calendar_time_increment_mode = "constant_per_year"` and `battery_calendar_time_increment_per_year`, but the loader still accepts `constant_per_step` and `battery_calendar_time_increment_per_step`, mapping them to the newer yearly semantics.
- **Suggested direction**: keep the alias only if backward compatibility is intentionally required; otherwise phase it out and update legacy example projects.

## 2. Dynamic battery exogenous degradation is only suppressed when calendar fade is enabled

- **Where**: `core/data_pipeline/battery_degradation_model.py::suppress_exogenous_battery_capacity_degradation_when_endogenous`
- **Type**: unclear modeling rule
- **Severity**: medium
- **Problem**: the code suppresses `battery_capacity_degradation_rate_per_year` when calendar fade is enabled, but not when cycle fade is enabled. This is documented in code comments, but it is still a non-obvious modeling choice and can surprise users who expect all endogenous degradation to disable the exogenous term.
- **Suggested direction**: confirm this is intentional and surface the rule more explicitly in UI help or cleanup notes.

## 3. `grid_availability.csv` lives in `inputs/` even though it is derived

- **Where**: `core/multi_year_model/data.py`, `pages/1_Data_Audit_and_Visualization.py`
- **Type**: workflow ambiguity
- **Severity**: medium
- **Problem**: the derived grid-availability artifact is written into the same folder as primary user-maintained inputs. This is practical, but it can make the project structure look as if the user should edit the file manually.
- **Suggested direction**: either move it to a generated-artifacts subfolder or label it more clearly in the UI and README as backend-generated only.

## 4. Vintage-label plumbing exists but is not surfaced as a primary user workflow

- **Where**: `pages/0_Project_Setup.py`, `core/io/templates.py`, `core/io/vintage_labels.py`
- **Type**: stale or partially implemented feature path
- **Severity**: low to medium
- **Problem**: the template settings and results code include vintage-label support, but the generated YAML files do not currently expose obvious user-facing `meta.labels` sections for those mappings in the standard template path.
- **Suggested direction**: either complete the template-writing path for vintage labels or remove unused template settings fields.

## 5. Result-export naming is implementation-faithful but may still confuse external users

- **Where**: `core/export/multi_year_results.py`, `core/visualization/multi_year_results_page.py`
- **Type**: naming / UX issue
- **Severity**: low
- **Problem**: exported files use names such as `cashflows_discounted.csv`, `scenario_costs_yearly.csv`, and `design_by_step.csv`, which are accurate but not yet tied to a formal external reporting glossary.
- **Suggested direction**: define a stable results naming convention before final PDF publication.

## Resolved in this cleanup pass

- user-facing Multi-Year investment-step keys now match the internal step labels (`"1"`, `"2"`, ...);
- Multi-Year year labels are enforced as integer-like calendar years in the UI and template generation;
- battery degradation ownership is clearer:
  - `formulation.json` stores activation flags,
  - `battery.yaml` stores battery-specific degradation parameters such as `initial_soh`;
- stale legacy Multi-Year schema branches such as top-level `by_step`, `operation` blocks, `technical.by_step`, and legacy `fuel.by_scenario` are now rejected explicitly by the parser.
