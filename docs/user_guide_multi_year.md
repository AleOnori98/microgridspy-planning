# MicroGridsPy Planning User Guide: Multi-Year

## 1. Introduction

MicroGridsPy Planning is the planning module of the MicroGridsPy ecosystem. It is a Streamlit application backed by Linopy optimization and a project-folder workflow based on JSON, YAML, and CSV files.

The app currently supports two complementary planning formulations:

- **Typical Year**: a steady-state, representative-year formulation. It uses one hourly year as the operational profile repeated over time, is computationally lighter, and minimizes an expected equivalent annual cost.
- **Multi-Year**: an explicit dynamic formulation with year-by-year evolution, investment-step logic, and cohort-style capacity tracking over the planning horizon.

This guide covers only the **Multi-Year** formulation, which is implemented in the codebase as `core_formulation = "dynamic"`. Compared with the Typical-Year workflow, Multi-Year requires more structure because the user must define a model horizon, yearly time series, and optionally investment steps for capacity expansion.

## 2. Quickstart Workflow

1. Launch the app and open **Project Setup**.
2. Create a project name. The app creates `projects/<project_name>/`.
3. Choose **Multi-year formulation**.
4. Configure the project:
   - start year and horizon length,
   - off-grid or on-grid,
   - export allowed or not,
   - continuous or discrete sizing,
   - single-scenario or multi-scenario,
   - renewable resources and labels,
   - battery loss model and optional degradation settings,
   - generator efficiency model,
   - optional capacity expansion and investment-step durations,
   - global optimization constraints.
5. Click **Initialize project and generate templates**.
6. Edit the generated files under `projects/<project_name>/inputs/`.
7. Open **Data Audit and Visualization** to verify that the dataset loads and that the year/scenario/resource dimensions match your intent.
8. Open **Model Optimization** and click **Build and solve**.
9. Open **Results** to inspect design by investment step, yearly KPIs, dispatch, discounted cash flows, and exported CSV/Excel results.

## 3. How the Streamlit Interface Handles Multi-Year

### Home

`Home.py` is the landing page. It links to:

- `pages/0_Project_Setup.py`
- `pages/1_Data_Audit_and_Visualization.py`
- `pages/3_Optimization.py`
- `pages/4_Results.py`

The landing page describes the two formulations at a high level but does not itself configure the model.

### Project Setup

This page is where a project is created or loaded.

- **Create** tab:
  - project name and description,
  - formulation selection,
  - horizon and start-year inputs for Multi-Year,
  - optional capacity expansion setup,
  - externality settings,
  - system configuration,
  - uncertainty settings,
  - optimization constraints,
  - template generation.
- **Load** tab:
  - marks an existing project as active without rewriting files.

Multi-Year-specific behavior:

- selecting Multi-Year writes `core_formulation = "dynamic"` to `inputs/formulation.json`;
- the UI requires:
  - `start_year_label`,
  - `time_horizon_years`,
  - `social_discount_rate`;
- if **capacity expansion** is enabled, the page asks for an ordered list of investment-step durations; these durations must sum exactly to the horizon length;
- battery degradation controls appear only for Multi-Year and only when the battery loss model is `convex_loss_epigraph`.

### Data Audit and Visualization

This page reads `inputs/formulation.json`, initializes the Multi-Year sets, loads the project dataset, and shows:

- required and optional files,
- dataset-loading status,
- coordinate summaries,
- parameter summaries,
- optimization-constraint summary,
- optional curve diagnostics,
- grid-availability controls for on-grid projects,
- time-series plots with scenario, year, and resource selectors.

Multi-Year-specific behavior:

- it uses `core.multi_year_model.sets.initialize_sets`;
- it loads the canonical dataset through the Multi-Year loader;
- for on-grid projects it regenerates `grid_availability.csv` from `grid.yaml`;
- plots expose explicit year selection rather than a single representative-year view.

### Model Optimization

This page:

- requires an active project,
- reads `inputs/formulation.json`,
- builds a `MultiYearModel` when `core_formulation = "dynamic"`,
- lets the user choose `highs` or `gurobi`,
- optionally writes a problem file,
- stores a solver log under `projects/<project_name>/logs/`,
- solves the single-objective model.

### Results

For Multi-Year projects, the page delegates to the dedicated Multi-Year results renderer. The outputs are organized around:

- design by investment step,
- yearly and scenario-based KPIs,
- discounted yearly cash flows,
- dispatch windows for a selected year and scenario or expected view,
- result export to CSV and Excel.

## 4. Project Directory Anatomy

Representative structure:

```text
projects/
  <project_name>/
    inputs/
      formulation.json
      README_inputs.md
      load_demand.csv
      resource_availability.csv
      renewables.yaml
      battery.yaml
      generator.yaml
      grid.yaml                       # only if on-grid
      grid_import_price.csv           # only if on-grid
      grid_export_price.csv           # only if on-grid and export enabled
      grid_availability.csv           # derived, backend-generated
      battery_efficiency_curve.csv    # only if battery loss model = convex_loss_epigraph
      battery_calendar_fade_curve.csv # only if calendar fade is enabled
      generator_efficiency_curve.csv  # only if generator efficiency model = efficiency_curve
    logs/
    results/
      multi_year/
        dispatch_timeseries.csv
        energy_balance.csv
        design_by_step.csv
        kpis_yearly.csv
        cashflows_discounted.csv
        scenario_costs_yearly.csv
        results_multi_year.xlsx
```

### Auto-generated files

- `formulation.json`
- `README_inputs.md`
- all primary input templates
- `grid_availability.csv` when the dataset is loaded or regenerated
- exported result CSVs and Excel workbook
- solver logs

### Files edited manually

- all primary input templates in `inputs/`, except `grid_availability.csv`

### Files reused from Typical Year vs Multi-Year-specific behavior

Shared filenames:

- `load_demand.csv`
- `resource_availability.csv`
- `renewables.yaml`
- `battery.yaml`
- `generator.yaml`
- grid files when on-grid

Multi-Year difference:

- the same filenames carry richer structure:
  - explicit year headers in CSVs,
  - `investment.by_step` blocks in YAMLs,
  - yearly fuel-price lists,
  - optional degradation inputs,
  - optional `first_year_connection` for the grid.

## 5. Multi-Year Initialization Logic

### Formulation switch

The app selects Multi-Year by writing:

- `core_formulation = "dynamic"`

to `inputs/formulation.json`.

The Multi-Year set initializer then derives:

- `period = 0..8759`
- `year = [start_year_label, start_year_label + 1, ...]`
- `scenario = scenario labels`
- `resource = renewable resource labels`
- `inv_step = [1]` if capacity expansion is disabled
- `inv_step = [1, 2, ..., N]` if capacity expansion is enabled

### Horizon and year labels

The implementation expects:

- `start_year_label` to be integer-like,
- `time_horizon_years` to be a positive integer.

If the user enters `start_year_label = 2026` and `time_horizon_years = 10`, the model years are:

- `2026, 2027, ..., 2035`

These year labels are then used consistently in:

- CSV headers,
- yearly fuel-price lists,
- grid-connection timing,
- yearly results.

### Capacity expansion and investment steps

If `capacity_expansion = false`:

- there is only one investment step in the model (`inv_step = [1]`);
- templates use a single `"1"` block in `investment.by_step`.

If `capacity_expansion = true`:

- the UI writes `investment_steps_years`, for example `[5, 5]`;
- the set initializer creates:
  - `inv_step = [1, 2]`
  - `inv_step_start_year = [2026, 2031]`
  - `inv_step_end_year = [2030, 2035]`
  - `inv_step_len_years = [5, 5]`
  - `year_inv_step[year]`
  - `inv_active_in_year[inv_step, year]`
- templates generate `investment.by_step.1`, `investment.by_step.2`, and so on.

Backward-compatibility detail: the loader still accepts legacy aliases such as `step_1` and single-step `base`, but the current generated Multi-Year templates now use the canonical numeric step labels directly.

### Shared-technology interpretation

In the current Multi-Year implementation:

- **investment-side parameters may vary by investment step**;
- **technical parameters are treated as shared across steps**.

This applies to:

- renewables,
- battery,
- generator,
- fuel physical properties.

The model therefore behaves as a shared-technology capacity-expansion model with cohort-specific economics and commissioning dates, not as a step-specific technology redesign model.

### Scenarios and scenario weights

If multi-scenario is disabled:

- the app uses a single scenario label, usually `scenario_1`.

If multi-scenario is enabled:

- `multi_scenario.scenario_labels` defines the ordered labels;
- `multi_scenario.scenario_weights` defines the weights;
- time-series CSVs and scenario-based YAML mappings must use the same labels.

The loader normalizes scenario weights if needed.

## 6. Multi-Year Input Files Overview

Support tables:

- `docs/tables/multi_year_input_reference.csv`
- `docs/tables/multi_year_timeseries_reference.csv`

### Master file table

| File | Format | Mandatory | When used | Edited by user? | Dimensional meaning |
| --- | --- | --- | --- | --- | --- |
| `formulation.json` | JSON | Yes | always | Normally no | project-level flags, horizon, scenarios, constraints |
| `load_demand.csv` | CSV | Yes | always | Yes | hourly demand by scenario and year |
| `resource_availability.csv` | CSV | Yes | always | Yes | hourly renewable availability by scenario, year, and resource |
| `renewables.yaml` | YAML | Yes | always | Yes | renewable investment-by-step and shared technical data |
| `battery.yaml` | YAML | Yes | always | Yes | battery investment-by-step and shared technical data |
| `generator.yaml` | YAML | Yes | always | Yes | generator investment-by-step, shared technical data, and yearly fuel prices |
| `grid.yaml` | YAML | Conditional | on-grid only | Yes | grid line, outages, renewable share, emissions, first-year connection |
| `grid_import_price.csv` | CSV | Conditional | on-grid only | Yes | hourly import tariff by scenario and year |
| `grid_export_price.csv` | CSV | Conditional | export enabled only | Yes | hourly export tariff by scenario and year |
| `grid_availability.csv` | CSV | Derived | on-grid only | No | hourly backend-generated availability by scenario and year |
| `battery_efficiency_curve.csv` | CSV | Conditional | battery loss model = `convex_loss_epigraph` | Yes | battery power-efficiency curve |
| `battery_calendar_fade_curve.csv` | CSV | Conditional | calendar fade enabled | Yes | yearly-average-SoC calendar-fade curve |
| `generator_efficiency_curve.csv` | CSV | Conditional | generator efficiency model = `efficiency_curve` | Yes | generator partial-load efficiency curve |
| `README_inputs.md` | Markdown | No | always | No | human-readable input summary only |

## 7. Detailed Parameter Reference

### 7.1 `inputs/formulation.json`

This file is written by Project Setup and read by the pages, set initializer, and Multi-Year data loader.

| Path | Meaning | Type | Unit | Required | Notes |
| --- | --- | --- | --- | --- | --- |
| `core_formulation` | formulation switch | string | - | Yes | must be `"dynamic"` for Multi-Year |
| `system_type` | UI-level system type | string | - | Yes | `"off_grid"` or `"on_grid"` |
| `on_grid` | grid activation flag | bool | - | Yes | used directly by the loader |
| `grid_allow_export` | export activation flag | bool | - | Yes | controls export variables and file generation |
| `unit_commitment` | integer sizing toggle | bool | - | Yes | in current code this means integer sizing, not chronological unit-commitment binaries |
| `start_year_label` | first modeled year label | int-like string or int | year | Yes | must be parseable as a positive integer |
| `time_horizon_years` | number of modeled years | int | years | Yes | positive integer |
| `social_discount_rate` | social discount rate | number | share | Yes | used in objective accounting and reporting |
| `capacity_expansion` | capacity-expansion toggle | bool | - | Yes | activates multiple investment steps |
| `investment_steps_years` | ordered list of step durations | list[int] or `null` | years | Conditional | required when `capacity_expansion = true`; sum must equal `time_horizon_years` |
| `multi_scenario.enabled` | scenario replication flag | bool | - | Yes | `false` means a single scenario |
| `multi_scenario.n_scenarios` | number of scenarios | int | - | Yes | should match labels and weights |
| `multi_scenario.scenario_labels` | scenario labels | list[str] | - | Yes | must match CSV headers and scenario-keyed YAML sections |
| `multi_scenario.scenario_weights` | scenario weights | list[number] | share | Yes if multi-scenario | normalized by the loader |
| `optimization_constraints.enforcement` | scenario-enforcement mode | string | - | Yes | `"scenario_wise"` or `"expected"` |
| `optimization_constraints.min_renewable_penetration` | renewable-share floor | number | share | Yes | typically between 0 and 1 |
| `optimization_constraints.max_lost_load_fraction` | unmet-load cap | number | share | Yes | typically between 0 and 1 |
| `optimization_constraints.lost_load_cost_per_kwh` | unmet-load penalty | number | currency/kWh | Yes | zero disables economic penalty |
| `optimization_constraints.land_availability_m2` | land cap | number or `null` | m2 | Yes | `null` disables land constraint |
| `optimization_constraints.emission_cost_per_kgco2e` | emissions cost | number | currency/kgCO2e | Yes | applies to scope 1, 2, and 3 accounting in objective/reporting |
| `system_configuration.n_sources` | renewable-resource count | int | - | Yes | should match the actual generated renewable entries |
| `battery_model.loss_model` | battery loss formulation | string | - | Yes | `"constant_efficiency"` or `"convex_loss_epigraph"` |
| `battery_model.degradation_model.cycle_fade_enabled` | cycle-fade toggle | bool | - | Yes | only valid in Multi-Year and only with `convex_loss_epigraph` |
| `battery_model.degradation_model.calendar_fade_enabled` | calendar-fade toggle | bool | - | Yes | only valid in Multi-Year and only with `convex_loss_epigraph` |
| `battery_model.degradation_model.cycle_lifetime_to_eol_cycles` | legacy detailed cycle-fade input | number | cycles | Legacy only | current UI does not write this here |
| `battery_model.degradation_model.end_of_life_soh` | legacy detailed end-of-life input | number | share | Legacy only | current UI does not write this here |
| `battery_model.degradation_model.initial_soh` | legacy initial state of health input | number | share | Legacy only | current UI now owns this in `battery.yaml` |
| `battery_model.degradation_model.battery_calendar_fade_curve_csv` | legacy calendar-fade curve filename | string | path | Legacy only | current UI now owns this in `battery.yaml` |
| `battery_model.degradation_model.battery_calendar_time_increment_mode` | calendar time-increment mode | string | - | Conditional | current code supports only `"constant_per_year"` |
| `battery_model.degradation_model.battery_calendar_time_increment_per_year` | legacy yearly calendar-ageing increment | number | increment/year | Legacy only | current UI now owns this in `battery.yaml` |
| `generator_model.efficiency_model` | generator part-load formulation | string | - | Yes | `"constant_efficiency"` or `"efficiency_curve"` |

### 7.2 `inputs/renewables.yaml`

Generated schema:

```yaml
renewables:
  - id: res_1
    conversion_technology: Solar_PV
    resource: Solar
    investment:
      by_step:
        "1":
          nominal_capacity_kw: 1.0
          specific_investment_cost_per_kw: 0.0
          wacc: 0.0
          grant_share_of_capex: 0.0
          lifetime_years: 25
          embedded_emissions_kgco2e_per_kw: 0.0
          fixed_om_share_per_year: 0.0
          production_subsidy_per_kwh: 0.0
    technical:
      inverter_efficiency: 1.0
      specific_area_m2_per_kw: null
      max_installable_capacity_kw: null
      capacity_degradation_rate_per_year: 0.0
```

One entry is required for each renewable resource. The `resource` field must match the third header level of `resource_availability.csv`.

#### Renewable fields

| Path | Meaning | Type | Unit | Required | Step/year/scenario behavior | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `renewables[].id` | internal identifier | string | - | No practical effect | shared | informational only |
| `renewables[].conversion_technology` | technology label | string | - | Yes | shared | used mainly for labeling and reporting |
| `renewables[].resource` | resource label | string | - | Yes | shared | must match `resource_availability.csv` resource header and the `sets.resource` coordinate |
| `renewables[].investment.by_step.<step>.nominal_capacity_kw` | capacity per installed renewable unit | number | kW | Yes | step-dependent | used with installed units to compute capacity |
| `renewables[].investment.by_step.<step>.specific_investment_cost_per_kw` | CAPEX intensity | number | currency/kW | Yes | step-dependent | annualized using `wacc` and `lifetime_years` |
| `renewables[].investment.by_step.<step>.wacc` | technology-specific WACC | number | share | Yes | step-dependent | usually in `[0,1]` |
| `renewables[].investment.by_step.<step>.grant_share_of_capex` | CAPEX grant share | number | share | Yes | step-dependent | usually in `[0,1]` |
| `renewables[].investment.by_step.<step>.lifetime_years` | lifetime | number | years | Yes | step-dependent | replacement logic uses this value |
| `renewables[].investment.by_step.<step>.embedded_emissions_kgco2e_per_kw` | embodied emissions | number | kgCO2e/kW | Yes | step-dependent | used in scope 3 emissions |
| `renewables[].investment.by_step.<step>.fixed_om_share_per_year` | fixed O&M share | number | share/year | Optional | step-dependent | defaults to `0.0` if omitted |
| `renewables[].investment.by_step.<step>.production_subsidy_per_kwh` | operating subsidy | number | currency/kWh | Optional | step-dependent | defaults to `0.0` if omitted |
| `renewables[].technical.inverter_efficiency` | conversion efficiency | number | - | Yes | shared across steps and scenarios | usually in `(0,1]` |
| `renewables[].technical.specific_area_m2_per_kw` | land coefficient | number or `null` | m2/kW | Yes | shared | required in practice if land constraint is active |
| `renewables[].technical.max_installable_capacity_kw` | max installable renewable capacity | number or `null` | kW | Yes | shared | `null` means no explicit cap |
| `renewables[].technical.capacity_degradation_rate_per_year` | exogenous yearly capacity fade | number | 1/year | Optional | shared | defaults to `0.0` |

Implementation notes:

- the current generated schema is `investment.by_step + shared technical`;
- legacy `operation` and `technical.by_step` branches are no longer accepted in the Multi-Year parser;
- step-varying technical parameters are not supported in the current Multi-Year workflow.

### 7.3 `inputs/battery.yaml`

Generated schema:

```yaml
battery:
  label: Battery
  investment:
    by_step:
      "1":
        nominal_capacity_kwh: 1.0
        specific_investment_cost_per_kwh: 0.0
        wacc: 0.0
        calendar_lifetime_years: 10
        embedded_emissions_kgco2e_per_kwh: 0.0
        fixed_om_share_per_year: 0.0
  technical:
    charge_efficiency: 0.95
    discharge_efficiency: 0.96
    initial_soc: 0.5
    depth_of_discharge: 0.8
    max_discharge_time_hours: 5.0
    max_charge_time_hours: 5.0
    max_installable_capacity_kwh: null
    efficiency_curve_csv: null or battery_efficiency_curve.csv
    capacity_degradation_rate_per_year: 0.0
    initial_soh: 1.0
    end_of_life_soh: 0.8
    cycle_lifetime_to_eol_cycles: 6000.0
    calendar_fade_curve_csv: battery_calendar_fade_curve.csv
    calendar_time_increment_per_year: 1.0
```

#### Battery fields

| Path | Meaning | Type | Unit | Required | Step/year/scenario behavior | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `battery.label` | battery label | string | - | No practical effect | shared | used in reporting |
| `battery.investment.by_step.<step>.nominal_capacity_kwh` | energy capacity per installed battery unit | number | kWh | Yes | step-dependent | used with installed units to compute nominal capacity |
| `battery.investment.by_step.<step>.specific_investment_cost_per_kwh` | CAPEX intensity | number | currency/kWh | Yes | step-dependent | annualized using `wacc` and lifetime |
| `battery.investment.by_step.<step>.wacc` | battery-specific WACC | number | share | Yes | step-dependent | usually in `[0,1]` |
| `battery.investment.by_step.<step>.calendar_lifetime_years` | replacement lifetime | number | years | Yes | step-dependent | economic/replacement lifetime |
| `battery.investment.by_step.<step>.embedded_emissions_kgco2e_per_kwh` | embodied emissions | number | kgCO2e/kWh | Yes | step-dependent | used in scope 3 emissions |
| `battery.investment.by_step.<step>.fixed_om_share_per_year` | fixed O&M share | number | share/year | Optional | step-dependent | defaults to `0.0` if omitted |
| `battery.technical.charge_efficiency` | one-way charge efficiency | number | - | Yes | shared | used directly in constant-efficiency mode and as base efficiency in curve mode |
| `battery.technical.discharge_efficiency` | one-way discharge efficiency | number | - | Yes | shared | same interpretation as above |
| `battery.technical.initial_soc` | initial state of charge | number | share | Yes | shared | fraction of usable capacity |
| `battery.technical.initial_soh` | initial state of health | number | share | Conditional | shared | required by the current UI-generated schema when endogenous degradation is enabled |
| `battery.technical.depth_of_discharge` | usable fraction of nominal capacity | number | share | Yes | shared | must be consistent with degradation assumptions |
| `battery.technical.max_discharge_time_hours` | minimum discharge time at nominal power | number | h | Yes | shared | power limit proxy |
| `battery.technical.max_charge_time_hours` | minimum charge time at nominal power | number | h | Yes | shared | power limit proxy |
| `battery.technical.max_installable_capacity_kwh` | total battery capacity cap | number or `null` | kWh | Yes | shared | `null` means no explicit cap |
| `battery.technical.efficiency_curve_csv` | battery efficiency-curve filename | string or `null` | path | Conditional | shared | required when `battery_model.loss_model = "convex_loss_epigraph"` |
| `battery.technical.capacity_degradation_rate_per_year` | exogenous yearly capacity fade | number | 1/year | Conditional | shared | used unless calendar fade is enabled; then it is zeroed internally to avoid double counting |
| `battery.technical.end_of_life_soh` | end-of-life state of health | number | share | Conditional | shared | generated when endogenous degradation is enabled |
| `battery.technical.cycle_lifetime_to_eol_cycles` | cycle life to end of life | number | cycles | Conditional | shared | generated when cycle fade is enabled |
| `battery.technical.calendar_fade_curve_csv` | calendar-fade curve filename | string | path | Conditional | shared | generated when calendar fade is enabled |
| `battery.technical.calendar_time_increment_per_year` | yearly calendar-ageing increment | number | increment/year | Conditional | shared | generated when calendar fade is enabled |

Current implementation behavior:

- battery technical parameters are shared across all investment steps;
- battery degradation tracking is supported only in Multi-Year and only with `battery_model.loss_model = "convex_loss_epigraph"`;
- cycle fade is derived from:
  - `initial_soh`,
  - `end_of_life_soh`,
  - `cycle_lifetime_to_eol_cycles`,
  - `depth_of_discharge`;
- the current UI-generated workflow stores battery-specific degradation parameters in `battery.yaml`, while `formulation.json` carries only the high-level activation flags;
- legacy direct input `cycle_fade_coefficient_per_kwh_throughput` is still accepted by the parser but not generated by the UI.

### 7.4 `inputs/generator.yaml`

Generated schema:

```yaml
generator:
  label: Generator
  investment:
    by_step:
      "1":
        nominal_capacity_kw: 1.0
        lifetime_years: 10
        specific_investment_cost_per_kw: 0.0
        wacc: 0.0
        embedded_emissions_kgco2e_per_kw: 0.0
        fixed_om_share_per_year: 0.0
  technical:
    nominal_efficiency_full_load: 0.30
    efficiency_curve_csv: null or generator_efficiency_curve.csv
    max_installable_capacity_kw: null
    capacity_degradation_rate_per_year: 0.0
fuel:
  label: Fuel
  technical:
    lhv_kwh_per_unit_fuel: 0.0
    direct_emissions_kgco2e_per_unit_fuel: 0.0
  cost:
    by_scenario:
      scenario_1:
        by_year_cost_per_unit_fuel: [0.0, ...]
```

#### Generator and fuel fields

| Path | Meaning | Type | Unit | Required | Step/year/scenario behavior | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `generator.label` | generator label | string | - | No practical effect | shared | reporting only |
| `generator.investment.by_step.<step>.nominal_capacity_kw` | power capacity per generator unit | number | kW | Yes | step-dependent | used with installed units |
| `generator.investment.by_step.<step>.lifetime_years` | lifetime | number | years | Yes | step-dependent | replacement logic uses this value |
| `generator.investment.by_step.<step>.specific_investment_cost_per_kw` | CAPEX intensity | number | currency/kW | Yes | step-dependent | annualized using `wacc` and lifetime |
| `generator.investment.by_step.<step>.wacc` | generator-specific WACC | number | share | Yes | step-dependent | usually in `[0,1]` |
| `generator.investment.by_step.<step>.embedded_emissions_kgco2e_per_kw` | embodied emissions | number | kgCO2e/kW | Yes | step-dependent | used in scope 3 emissions |
| `generator.investment.by_step.<step>.fixed_om_share_per_year` | fixed O&M share | number | share/year | Optional | step-dependent | defaults to `0.0` if omitted |
| `generator.technical.nominal_efficiency_full_load` | full-load efficiency | number | - | Yes | shared | baseline efficiency for constant and curve modes |
| `generator.technical.efficiency_curve_csv` | partial-load curve filename | string or `null` | path | Conditional | shared | used only when generator efficiency model is `efficiency_curve` |
| `generator.technical.max_installable_capacity_kw` | max installable generator capacity | number or `null` | kW | Yes | shared | `null` means no explicit cap |
| `generator.technical.capacity_degradation_rate_per_year` | exogenous yearly capacity fade | number | 1/year | Optional | shared | defaults to `0.0` |
| `fuel.label` | fuel label | string | - | No practical effect | shared | reporting only |
| `fuel.technical.lhv_kwh_per_unit_fuel` | lower heating value | number | kWh/unit fuel | Yes | shared | required for efficiency/fuel-consumption accounting |
| `fuel.technical.direct_emissions_kgco2e_per_unit_fuel` | direct combustion emissions factor | number | kgCO2e/unit fuel | Yes | shared | used in scope 1 emissions |
| `fuel.cost.by_scenario.<scenario>.by_year_cost_per_unit_fuel` | yearly fuel-price trajectory | list[number] | currency/unit fuel | Yes | scenario- and year-dependent | length must equal number of modeled years |

Implementation notes:

- generator technical parameters are shared across steps;
- fuel physical properties are shared across steps and scenarios;
- only yearly fuel prices are scenario-dependent in the generated schema;
- legacy `fuel.by_scenario`, top-level `by_step`, and `operation` branches are no longer accepted in the current Multi-Year parser.

### 7.5 `inputs/grid.yaml`

This file exists only if `on_grid = true`.

Generated schema:

```yaml
grid:
  by_scenario:
    scenario_1:
      line:
        capacity_kw: 0.0
        transmission_efficiency: 1.0
        renewable_share: 0.0
        emissions_factor_kgco2e_per_kwh: 0.0
      outages:
        average_outages_per_year: 0.0
        average_outage_duration_minutes: 0.0
        outage_scale_od_hours: 0.6
        outage_shape_od: 0.56
        outage_seed: 0
      first_year_connection: 2026
```

#### Grid fields

| Path | Meaning | Type | Unit | Required | Step/year/scenario behavior | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `grid.by_scenario.<scenario>.line.capacity_kw` | interconnection capacity | number | kW | Yes | scenario-dependent | same limit used for import and export |
| `grid.by_scenario.<scenario>.line.transmission_efficiency` | delivered-energy efficiency | number | - | Yes | scenario-dependent | must be in `[0,1]` |
| `grid.by_scenario.<scenario>.line.renewable_share` | renewable share of imported power | number | share | Yes | scenario-dependent | must be in `[0,1]` |
| `grid.by_scenario.<scenario>.line.emissions_factor_kgco2e_per_kwh` | grid import emissions factor | number | kgCO2e/kWh | Yes | scenario-dependent | must be `>= 0` |
| `grid.by_scenario.<scenario>.outages.average_outages_per_year` | outage frequency | number | events/year | Yes | scenario-dependent | used by outage simulator |
| `grid.by_scenario.<scenario>.outages.average_outage_duration_minutes` | outage duration | number | minutes | Yes | scenario-dependent | used by outage simulator |
| `grid.by_scenario.<scenario>.outages.outage_scale_od_hours` | Weibull scale parameter | number | h | Yes | scenario-dependent | must be `> 0` |
| `grid.by_scenario.<scenario>.outages.outage_shape_od` | Weibull shape parameter | number | - | Yes | scenario-dependent | must be `> 0` |
| `grid.by_scenario.<scenario>.outages.outage_seed` | RNG seed for outage simulation | int-like number | seed | Yes | scenario-dependent | used for deterministic regeneration |
| `grid.by_scenario.<scenario>.first_year_connection` | first connected model year | year label, int, `null`, or blank | year | Optional | scenario-dependent | `null` or blank means connected from horizon start |

Connection timing rules:

- preferred input: an exact year label from `sets.year`, for example `2031`;
- integer fallback is accepted only because model years are currently integer-like;
- if `first_year_connection` is after the last modeled year, grid availability is zero for the whole horizon.

### 7.6 Informational files not parsed as model inputs

`README_inputs.md` is generated for human guidance only. It is not a primary model input.

`meta` blocks inside YAML files are mainly descriptive. The main loader logic reads the substantive user-editable sections, not the descriptive metadata.

## 8. Detailed Time-Series Reference

### 8.1 `load_demand.csv`

Header structure:

- row 1: scenario
- row 2: year

Required first column:

- `meta/hour`

Example:

```csv
meta,scenario_1,scenario_1
hour,2026,2027
0,15.2,16.0
1,14.9,15.8
```

Rules:

- exactly 8760 rows are required;
- `meta/hour` must run from `0` to `8759`;
- every scenario-year combination must exist;
- units are `kWh` during each hour.

Loaded dataset shape:

- `(year, period, scenario)`

### 8.2 `resource_availability.csv`

Header structure:

- row 1: scenario
- row 2: year
- row 3: resource

Required first column:

- `meta/hour` represented as `("meta","hour","")`

Example:

```csv
meta,scenario_1,scenario_1
hour,2026,2026
,Solar,Wind
0,0.00,0.45
1,0.00,0.43
```

Rules:

- exactly 8760 rows are required;
- every scenario-year-resource combination must exist;
- values must be numeric;
- the loader enforces a broad bound of approximately `[0, 1.5]`, though intended values are capacity factors near `[0,1]`.

Loaded dataset shape:

- `(year, period, scenario, resource)`

### 8.3 `grid_import_price.csv`

Header structure matches `load_demand.csv`:

- row 1: scenario
- row 2: year

Rules:

- required only in on-grid mode;
- exactly 8760 rows;
- all scenario-year combinations required;
- units are `currency/kWh`.

Loaded dataset shape:

- `(period, scenario, year)`

### 8.4 `grid_export_price.csv`

Same structure as `grid_import_price.csv`.

Rules:

- required only when both `on_grid = true` and `grid_allow_export = true`.

### 8.5 `generator_efficiency_curve.csv`

Required columns:

- `Relative Power Output [-]`
- `Efficiency [-]`

Current accepted behavior:

- the parser accepts an explicit `0.0` row but does not require it;
- at least one positive-load point is required;
- the last relative-power point must be `1.0`;
- positive-load points must be strictly increasing;
- efficiency values for positive-load points must be strictly positive.

Preferred interpretation:

- `Efficiency [-]` is a normalized multiplier relative to `generator.technical.nominal_efficiency_full_load`;
- the full-load point should therefore be `1.0`.

Legacy behavior still accepted:

- if all values are within a valid absolute-efficiency range and the last point is not `1.0`, the curve is interpreted as an absolute-efficiency curve.

The loader converts this user-facing curve into an internal convex relative fuel-use surrogate for the LP formulation.

### 8.6 `battery_efficiency_curve.csv`

Required columns:

- `relative_power_pu`
- `charge_efficiency`
- `discharge_efficiency`

Rules:

- at least 2 rows are required;
- `relative_power_pu` must lie in `(0,1]`;
- it must be strictly increasing;
- the last point must be `1.0`.

Preferred interpretation:

- `charge_efficiency` and `discharge_efficiency` are normalized multipliers relative to the scalar efficiencies in `battery.yaml`;
- the full-load point should therefore be `1.0`.

Legacy behavior still accepted:

- absolute-efficiency curves are accepted if values remain in the valid efficiency range.

The loader derives convex charge-loss and discharge-loss epigraphs from this file.

### 8.7 `battery_calendar_fade_curve.csv`

Required columns:

- `soc_pu`
- preferred: `calendar_fade_coefficient_per_year`
- legacy alias: `calendar_fade_coefficient_per_step`

Rules:

- at least 2 rows are required;
- `soc_pu` must lie in `[0,1]`;
- if the first point is above `0`, the loader prepends a `0` point automatically using the first coefficient value;
- the last `soc_pu` point must be `1.0`;
- the resulting curve must be convex in the sense required by the piecewise-linear surrogate.

Interpretation:

- `soc_pu` is yearly average SoC;
- the coefficient is a yearly calendar-ageing coefficient used together with `battery_calendar_time_increment_per_year`.

### 8.8 `grid_availability.csv`

This file is derived, not user-authored.

It is regenerated from:

- `grid.yaml`
- the current Multi-Year sets

Header structure:

- row 1: scenario
- row 2: year

Required first column:

- `meta/hour`

Loaded dataset shape:

- `(period, scenario, year)`

## 9. Conditional Inputs and Advanced Options

### On-grid mode

Activated by:

- `on_grid = true`

Additional required inputs:

- `grid.yaml`
- `grid_import_price.csv`
- `grid_export_price.csv` if export is enabled

Derived file:

- `grid_availability.csv`

### Export mode

Activated by:

- `grid_allow_export = true`

Additional required input:

- `grid_export_price.csv`

### Capacity expansion

Activated by:

- `capacity_expansion = true`

Effects:

- the UI requires `investment_steps_years`;
- renewable, battery, and generator templates generate one `investment.by_step` block per step;
- the model tracks each commissioned cohort by investment step;
- technical parameters remain shared across steps.

### Multi-scenario mode

Activated by:

- `multi_scenario.enabled = true`

Effects:

- all hourly time-series CSVs must contain every scenario-year combination;
- `fuel.cost.by_scenario` and `grid.by_scenario` must contain every scenario label;
- results pages expose expected and scenario-specific views.

### Battery convex loss model

Activated by:

- `battery_model.loss_model = "convex_loss_epigraph"`

Additional required input:

- `battery_efficiency_curve.csv`

Effect:

- battery charge/discharge losses are modeled with a convex piecewise-linear surrogate based on DC-side relative power.

### Battery endogenous degradation

Activated by:

- `battery_model.loss_model = "convex_loss_epigraph"`
- and either `cycle_fade_enabled` or `calendar_fade_enabled`

Possible additional inputs:

- `end_of_life_soh`
- `cycle_lifetime_to_eol_cycles`
- `battery_calendar_fade_curve.csv`
- `battery_calendar_time_increment_per_year`

Behavior:

- cycle fade and calendar fade are tracked as yearly degradation state variables;
- exogenous `capacity_degradation_rate_per_year` is suppressed only when calendar fade is enabled;
- replacement timing still follows the calendar-lifetime input.

### Generator efficiency curve

Activated by:

- `generator_model.efficiency_model = "efficiency_curve"`

Additional required input:

- `generator_efficiency_curve.csv`

Effect:

- the user curve is converted into an internal convex fuel-use surrogate.

## 10. Consistency Checks and Common Mistakes

Check the following before solving:

1. `start_year_label` must be an integer-like calendar year. The current UI now enforces this directly.
2. If capacity expansion is enabled, `investment_steps_years` must sum exactly to `time_horizon_years`.
3. Scenario labels must match everywhere:
   - `formulation.json`
   - CSV headers
   - `fuel.cost.by_scenario`
   - `grid.by_scenario`
4. Resource labels must match everywhere:
   - `renewables[].resource`
   - `resource_availability.csv` third header row
5. `fuel.cost.by_scenario.<scenario>.by_year_cost_per_unit_fuel` must have one value for every modeled year.
6. `load_demand.csv`, `resource_availability.csv`, and grid price CSVs must contain exactly 8760 rows and a correct `meta/hour` sequence from `0` to `8759`.
7. In on-grid projects, `first_year_connection` should use an exact modeled year label when possible.
8. If land constraints are active, every renewable entry should provide a meaningful `specific_area_m2_per_kw`.
9. If `battery_model.loss_model = "convex_loss_epigraph"`, `battery.technical.efficiency_curve_csv` must point to an existing curve file.
10. If battery calendar fade is enabled, `battery_calendar_fade_curve.csv` is mandatory and the current code only supports `battery_calendar_time_increment_mode = "constant_per_year"`.
11. If generator curve mode is active, the generator curve must end at relative power `1.0`.
12. Do not try to make technology parameters vary by investment step. The current Multi-Year backend still assumes shared technical parameters across steps.

## 11. Minimal Working Examples

### Example A: Simplest off-grid Multi-Year case

Use this when you want the smallest dynamic project that still exercises the Multi-Year machinery.

- `core_formulation = "dynamic"`
- `system_type = "off_grid"`
- `on_grid = false`
- `capacity_expansion = false`
- `start_year_label = 2026`
- `time_horizon_years = 10`
- `multi_scenario.enabled = false`
- one renewable resource
- battery loss model = `constant_efficiency`
- generator efficiency model = `constant_efficiency`

Required files:

- `formulation.json`
- `load_demand.csv` with one scenario and ten year columns
- `resource_availability.csv` with one scenario, ten year columns, and one resource
- `renewables.yaml` with a single `"1"` investment block
- `battery.yaml` with a single `"1"` investment block
- `generator.yaml` with:
  - a single `"1"` generator investment block,
  - shared fuel properties,
  - one yearly fuel-price list with ten entries

### Example B: Richer on-grid Multi-Year case

Use this when you want to exercise most dynamic features.

- `core_formulation = "dynamic"`
- `system_type = "on_grid"`
- `on_grid = true`
- `grid_allow_export = true`
- `capacity_expansion = true`
- `investment_steps_years = [5, 5]`
- `start_year_label = 2026`
- `time_horizon_years = 10`
- optionally multiple scenarios
- optional battery convex loss model and degradation
- optional generator efficiency curve

Additional required files:

- `grid.yaml`
- `grid_import_price.csv`
- `grid_export_price.csv`
- `battery_efficiency_curve.csv` if convex battery model is enabled
- `battery_calendar_fade_curve.csv` if calendar fade is enabled
- `generator_efficiency_curve.csv` if generator curve mode is enabled

Recommended sanity checks:

- use exact year labels such as `2026` and `2031` for `first_year_connection`;
- keep `fuel.cost.by_scenario` list lengths equal to the horizon length;
- confirm that step durations sum to the horizon length.

## 12. Appendix

### Glossary

| Term | Meaning in current implementation |
| --- | --- |
| `year` | explicit modeled year label, usually a calendar year such as `2026` |
| `period` | hourly index within the representative operating year, `0..8759` |
| `scenario` | uncertainty branch with shared investment decisions and different operating conditions |
| `inv_step` | investment step / commissioning cohort index |
| `year_inv_step` | mapping from each year to the corresponding investment step |
| `inv_active_in_year` | mask showing whether a cohort commissioned in a step is available in a given year |
| cohort / vintage | capacity commissioned in an investment step and then tracked through time |
| shared-technology model | current Multi-Year interpretation in which investment terms may vary by step but technical parameters remain shared |

### Mapping between UI choices and generated files

| UI choice | Effect |
| --- | --- |
| Multi-Year formulation | writes `core_formulation = "dynamic"` |
| Start year + horizon | sets the `year` coordinate and all year headers |
| Capacity expansion enabled | generates multiple `investment.by_step` blocks |
| On-grid | generates `grid.yaml` and `grid_import_price.csv` |
| Export enabled | also generates `grid_export_price.csv` |
| Multi-scenario | replicates scenario labels across CSV headers and scenario-keyed YAML sections |
| Battery loss model = `convex_loss_epigraph` | generates battery efficiency-curve template and activates curve-based loader logic |
| Battery calendar fade enabled | generates calendar-fade curve template and writes calendar-fade settings |
| Generator efficiency model = `efficiency_curve` | generates generator efficiency-curve template and writes its filename into `generator.yaml` |

### Source basis used for this guide

This guide was reconstructed primarily from the repository implementation, especially:

- `pages/0_Project_Setup.py`
- `pages/1_Data_Audit_and_Visualization.py`
- `pages/3_Optimization.py`
- `pages/4_Results.py`
- `core/io/templates.py`
- `core/multi_year_model/sets.py`
- `core/multi_year_model/data.py`
- `core/data_pipeline/battery_loss_model.py`
- `core/data_pipeline/battery_degradation_model.py`
- `core/data_pipeline/battery_calendar_fade_model.py`
- `core/data_pipeline/generator_partial_load_model.py`
- `core/export/multi_year_results.py`

When code and generated template comments diverged, this guide follows the current loader and set-initialization behavior and records mismatches in `docs/user_guide_multi_year_open_issues.md`.
