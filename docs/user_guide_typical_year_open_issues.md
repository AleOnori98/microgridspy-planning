# Typical Year User Guide Open Issues

## Remaining follow-up items

1. Typical Year `formulation.json` still stores fields such as `start_year_label`, `time_horizon_years`, and battery-degradation structure even though they are not operationally relevant in the steady-state branch.
2. The Project Setup page still exposes several battery-degradation controls in a shared expander even though Typical Year only uses the optional battery loss-curve path. This is functionally correct but could be simplified later.
3. Land-constraint usability still depends on users providing finite `renewables.technical.specific_area_m2_per_kw`; there is no template-side coupling between that field and the land-limit toggle.
4. Resource labels are derived from `renewables.yaml` during Typical Year set initialization. If a user edits `formulation.json.system_configuration.n_sources` manually without regenerating templates, consistency can still drift.
5. `grid_availability.csv` is backend-generated and re-generated in Data Audit, but manual edits are still technically possible.
6. Typical Year `unit_commitment` semantics are integer sizing only. The label may still suggest chronological commitment to some users.

## Suggested follow-up checks for the Multi-Year guide

1. Trace dynamic year and investment-step label generation end to end.
2. Document battery degradation inputs and endogenous degradation outputs in detail.
3. Verify all Multi-Year year-dependent CSV conventions against the current loader.
4. Reconcile dynamic grid first-year-connection semantics with year labels and parser expectations.
5. Document result-export structure for `design_by_step.csv`, `kpis_yearly.csv`, and discounted cash-flow outputs.
