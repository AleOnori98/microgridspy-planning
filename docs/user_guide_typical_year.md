# MicroGridsPy Planning User Guide: Typical Year

## 1. Introduction

MicroGridsPy Planning is the planning module of the MicroGridsPy ecosystem. It is a Streamlit application backed by Linopy optimization and a project-folder workflow based on JSON, YAML, and CSV files.

The app currently supports two complementary planning formulations:

- **Typical Year**: a steady-state, representative-year formulation. It uses one hourly year as the operational profile repeated over time, is computationally lighter, and minimizes an expected equivalent annual cost.
- **Multi-Year**: an explicit dynamic formulation with year-by-year evolution, capacity expansion logic, and time-dependent inputs.

This guide covers only the **Typical Year** formulation, which is implemented in the codebase as `core_formulation = "steady_state"`.

## 2. Quickstart Workflow

1. Launch the app and open **Project Setup**.
2. Create a project name. The app creates `projects/<project_name>/`.
3. Choose **Typical-year formulation**.
4. Configure the project:
   - off-grid or on-grid,
   - export allowed or not,
   - continuous or discrete sizing,
   - single-scenario or multi-scenario,
   - renewable resources and labels,
   - optional battery loss curve,
   - optional generator efficiency curve,
   - system constraints.
5. Click **Initialize project and generate templates**.
6. Edit the generated files under `projects/<project_name>/inputs/`.
7. Open **Data Audit and Visualization** to confirm that all required files are present and the canonical dataset loads.
8. Open **Model Optimization** and click **Build and solve**.
9. Open **Results** to inspect sizing, KPIs, dispatch, and export CSV results.

## 3. How the Streamlit Interface Is Organized

### Home

`Home.py` is the landing page. It links to:

- `pages/0_Project_Setup.py`
- `pages/1_Data_Audit_and_Visualization.py`
- `pages/3_Optimization.py`
- `pages/4_Results.py`

### Project Setup

This page is where a project is created or loaded.

- **Create** tab:
  - project name and description,
  - formulation selection,
  - externality settings,
  - system configuration,
  - uncertainty settings,
  - optimization constraints,
  - template generation.
- **Load** tab:
  - marks an existing project as active without rewriting files.

Important behavior: creating a project with an already existing name overwrites the generated files in `inputs/` using the current page settings.

### Data Audit and Visualization

This page reads `inputs/formulation.json`, initializes the sets, loads the project dataset, and shows:

- required and optional files,
- dataset summary,
- optimization-constraint summary,
- optional curve diagnostics,
- grid-availability controls for on-grid projects,
- time-series plots,
- scenario-comparison views.

For on-grid projects, `grid_availability.csv` is treated as a derived artifact and can be regenerated from `grid.yaml`.

### Model Optimization

This page:

- requires an active project,
- reads `inputs/formulation.json`,
- builds the Typical Year or Multi-Year model,
- lets the user choose `highs` or `gurobi`,
- optionally writes a problem file,
- stores a solver log under `projects/<project_name>/logs/`,
- solves the single-objective model.

### Results

For Typical Year projects, this page displays:

- sizing summary,
- KPIs,
- dispatch plot,
- cost and annuity breakdown,
- scenario-specific operational costs and emissions,
- CSV export.

## 4. Project Directory Anatomy

Typical structure:

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
      battery_efficiency_curve.csv    # only if advanced battery loss mode is enabled
      generator_efficiency_curve.csv  # only if generator curve mode is enabled
    results/
      typical_year/
        dispatch_timeseries.csv
        energy_balance.csv
        design_summary.csv
        kpis.csv
    logs/
```

### Auto-generated files

- `formulation.json`
- `README_inputs.md`
- all primary input templates
- `grid_availability.csv` when the dataset is loaded or regenerated
- exported result CSVs
- solver logs

### Files edited manually

- all primary input templates in `inputs/`, except `grid_availability.csv`

## 5. Typical Year Initialization Logic

When **Typical-year formulation** is selected, the app writes:

- `core_formulation = "steady_state"`
- `start_year_label = "typical_year"`
- no dynamic horizon or investment-step logic for the optimization path

The following UI choices control which files exist:

| UI choice | Effect |
| --- | --- |
| Off-grid | no grid files generated |
| On-grid | generates `grid.yaml` and `grid_import_price.csv` |
| Export enabled | also generates `grid_export_price.csv` |
| Single scenario | uses `scenario_1` |
| Multi-scenario | uses configured scenario labels and weights in headers and YAML mappings |
| Battery loss model = `convex_loss_epigraph` | generates battery efficiency-curve CSV and writes its filename into `battery.yaml` |
| Generator efficiency model = `efficiency_curve` | generates generator curve CSV and writes its filename into `generator.yaml` |
| Renewable count and labels | changes `renewables.yaml` entries and `resource_availability.csv` headers |

## 6. Typical Year Input Files Overview

Support tables:

- `docs/tables/typical_year_input_reference.csv`
- `docs/tables/typical_year_timeseries_reference.csv`

### Master file table

| File | Format | Mandatory | When used | Edited by user? |
| --- | --- | --- | --- | --- |
| `formulation.json` | JSON | Yes | always | Normally no |
| `load_demand.csv` | CSV | Yes | always | Yes |
| `resource_availability.csv` | CSV | Yes | always | Yes |
| `renewables.yaml` | YAML | Yes | always | Yes |
| `battery.yaml` | YAML | Yes | always | Yes |
| `generator.yaml` | YAML | Yes | always | Yes |
| `grid.yaml` | YAML | Conditional | on-grid only | Yes |
| `grid_import_price.csv` | CSV | Conditional | on-grid only | Yes |
| `grid_export_price.csv` | CSV | Conditional | export enabled only | Yes |
| `grid_availability.csv` | CSV | Derived | on-grid only | No |
| `battery_efficiency_curve.csv` | CSV | Conditional | advanced battery loss mode | Yes |
| `generator_efficiency_curve.csv` | CSV | Conditional | generator curve mode | Yes |

## 7. Detailed Parameter Reference

### 7.1 `inputs/formulation.json`

This file is written by Project Setup and read by the loaders and pages.

| Key | Meaning in Typical Year | Type | Required | Notes |
| --- | --- | --- | --- | --- |
| `project_name` | project identifier | string | Yes | usually matches folder name |
| `description` | free-text description | string | No | informational |
| `core_formulation` | formulation switch | string | Yes | must be `"steady_state"` |
| `system_type` | UI-level system type | string | Yes | `"off_grid"` or `"on_grid"` |
| `on_grid` | grid activation flag | bool | Yes | used by the loader |
| `grid_allow_export` | export activation flag | bool | Yes | controls export files and variables |
| `unit_commitment` | integer sizing toggle | bool | Yes | in Typical Year it does not add chronological commitment binaries |
| `multi_scenario.enabled` | scenario replication flag | bool | Yes | `false` means single scenario |
| `multi_scenario.n_scenarios` | number of scenarios | int | Yes | must align with labels and weights |
| `multi_scenario.scenario_labels` | ordered scenario labels | list[str] | Yes | must match CSV headers and scenario YAML mappings |
| `multi_scenario.scenario_weights` | scenario probabilities | list[number] | Yes if multi-scenario | normalized by the loader |
| `optimization_constraints.enforcement` | scenario enforcement mode | string | Yes | `"scenario_wise"` or `"expected"` |
| `optimization_constraints.min_renewable_penetration` | renewable share floor | number | Yes | share, usually 0 to 1 |
| `optimization_constraints.max_lost_load_fraction` | unmet-load cap | number | Yes | share, usually 0 to 1 |
| `optimization_constraints.lost_load_cost_per_kwh` | unmet-load penalty | number | Yes | cost per kWh |
| `optimization_constraints.land_availability_m2` | optional land cap | number or `null` | Yes | `null` disables land constraint |
| `optimization_constraints.emission_cost_per_kgco2e` | carbon externality cost | number | Yes | cost per kgCO2e |
| `system_configuration.n_sources` | renewable-resource count | int | Yes | should match the actual templates |
| `battery_model.loss_model` | battery loss formulation | string | Yes | `"constant_efficiency"` or `"convex_loss_epigraph"` |
| `generator_model.efficiency_model` | generator part-load formulation | string | Yes | `"constant_efficiency"` or `"efficiency_curve"` |

### 7.2 `inputs/renewables.yaml`

Typical structure:

```yaml
renewables:
  - id: res_1
    conversion_technology: Solar_PV
    resource: Solar
    investment:
      by_step:
        base:
          nominal_capacity_kw: 1.0
          specific_investment_cost_per_kw: 800
          wacc: 0.05
          grant_share_of_capex: 0.0
          lifetime_years: 25
          embedded_emissions_kgco2e_per_kw: 0.0
          fixed_om_share_per_year: 0.02
          production_subsidy_per_kwh: 0.0
    technical:
      inverter_efficiency: 1.0
      specific_area_m2_per_kw: null
      max_installable_capacity_kw: null
```

Typical Year uses one renewable entry per resource and reads one investment block from `investment.by_step`.

| Path | Meaning | Type | Unit | Required | Notes |
| --- | --- | --- | --- | --- | --- |
| `resource` | resource label | string | - | Yes | must match the resource header in `resource_availability.csv` |
| `investment.by_step.<step>.nominal_capacity_kw` | capacity per installed unit | number | kW | Yes | Typical Year accepts `base` or one single step block |
| `investment.by_step.<step>.specific_investment_cost_per_kw` | CAPEX intensity | number | currency/kW | Yes | annualized through WACC and lifetime |
| `investment.by_step.<step>.wacc` | renewable-specific WACC | number | - | Yes | typically 0 to 1 |
| `investment.by_step.<step>.grant_share_of_capex` | CAPEX grant fraction | number | share | Yes | usually 0 to 1 |
| `investment.by_step.<step>.lifetime_years` | lifetime | number | years | Yes | economic lifetime |
| `investment.by_step.<step>.embedded_emissions_kgco2e_per_kw` | embodied emissions | number | kgCO2e/kW | Yes | used in embodied emissions accounting |
| `investment.by_step.<step>.fixed_om_share_per_year` | fixed O&M share | number | share/year | Optional | defaults to `0.0` if omitted |
| `investment.by_step.<step>.production_subsidy_per_kwh` | production subsidy | number | currency/kWh | Optional | defaults to `0.0` if omitted |
| `technical.inverter_efficiency` | conversion efficiency | number | - | Yes | generally 0 to 1 |
| `technical.specific_area_m2_per_kw` | land intensity | number or `null` | m2/kW | Yes | use a finite value if land constraint is active |
| `technical.max_installable_capacity_kw` | max installable renewable capacity | number or `null` | kW | Yes | `null` means no explicit cap |

Typical Year no longer accepts the old `operation.by_scenario` renewable block. Fixed O&M and production subsidy must be declared in `investment.by_step.<step>`.

### 7.3 `inputs/battery.yaml`

Typical structure:

```yaml
battery:
  label: Battery
  investment:
    by_step:
      base:
        nominal_capacity_kwh: 1.0
        specific_investment_cost_per_kwh: 350
        wacc: 0.05
        calendar_lifetime_years: 10
        embedded_emissions_kgco2e_per_kwh: 0.0
        fixed_om_share_per_year: 0.02
  technical:
    charge_efficiency: 0.95
    discharge_efficiency: 0.96
    initial_soc: 0.5
    depth_of_discharge: 0.8
    max_discharge_time_hours: 5.0
    max_charge_time_hours: 5.0
    max_installable_capacity_kwh: null
    efficiency_curve_csv: null
```

| Path | Meaning | Type | Unit | Required | Notes |
| --- | --- | --- | --- | --- | --- |
| `battery.label` | component label | string | - | No | informational |
| `battery.investment.by_step.<step>.nominal_capacity_kwh` | capacity per battery unit | number | kWh | Yes | Typical Year accepts `base` or one single step block |
| `battery.investment.by_step.<step>.specific_investment_cost_per_kwh` | CAPEX intensity | number | currency/kWh | Yes | annualized |
| `battery.investment.by_step.<step>.wacc` | battery-specific WACC | number | - | Yes | typically 0 to 1 |
| `battery.investment.by_step.<step>.calendar_lifetime_years` | calendar lifetime | number | years | Yes | economic lifetime |
| `battery.investment.by_step.<step>.embedded_emissions_kgco2e_per_kwh` | embodied emissions | number | kgCO2e/kWh | Yes | embodied externality input |
| `battery.investment.by_step.<step>.fixed_om_share_per_year` | fixed O&M share | number | share/year | Optional | defaults to `0.0` if omitted |
| `battery.technical.charge_efficiency` | charge efficiency | number | - | Yes | full-load baseline |
| `battery.technical.discharge_efficiency` | discharge efficiency | number | - | Yes | full-load baseline |
| `battery.technical.initial_soc` | initial state of charge | number | share | Yes | fraction of usable capacity |
| `battery.technical.depth_of_discharge` | usable fraction of nominal capacity | number | share | Yes | usually 0 to 1 |
| `battery.technical.max_discharge_time_hours` | full discharge time at nominal power | number | hours | Yes | power proxy |
| `battery.technical.max_charge_time_hours` | full charge time at nominal power | number | hours | Yes | power proxy |
| `battery.technical.max_installable_capacity_kwh` | planning upper bound | number or `null` | kWh | Yes | `null` means no explicit cap |
| `battery.technical.efficiency_curve_csv` | advanced battery curve filename | string or `null` | path | Conditional | required only for `convex_loss_epigraph` |

Typical Year no longer accepts the old `battery.operation` block. Fixed O&M must be declared in `battery.investment.by_step.<step>`.

### 7.4 `inputs/generator.yaml`

Typical structure:

```yaml
generator:
  label: Generator
  investment:
    by_step:
      base:
        nominal_capacity_kw: 1.0
        lifetime_years: 10
        specific_investment_cost_per_kw: 400
        wacc: 0.07
        embedded_emissions_kgco2e_per_kw: 0.0
        fixed_om_share_per_year: 0.03
  technical:
    nominal_efficiency_full_load: 0.3
    efficiency_curve_csv: null
    max_installable_capacity_kw: null
fuel:
  label: Diesel
  by_scenario:
    scenario_1:
      lhv_kwh_per_unit_fuel: 10
      direct_emissions_kgco2e_per_unit_fuel: 0.0
      fuel_cost_per_unit_fuel: 1.2
```

| Path | Meaning | Type | Unit | Required | Notes |
| --- | --- | --- | --- | --- | --- |
| `generator.investment.by_step.<step>.nominal_capacity_kw` | unit size | number | kW | Yes | Typical Year accepts `base` or one single step block |
| `generator.investment.by_step.<step>.lifetime_years` | generator lifetime | number | years | Yes | economic lifetime |
| `generator.investment.by_step.<step>.specific_investment_cost_per_kw` | CAPEX intensity | number | currency/kW | Yes | annualized |
| `generator.investment.by_step.<step>.wacc` | generator-specific WACC | number | - | Yes | typically 0 to 1 |
| `generator.investment.by_step.<step>.embedded_emissions_kgco2e_per_kw` | embodied emissions | number | kgCO2e/kW | Optional | defaults to `0.0` |
| `generator.investment.by_step.<step>.fixed_om_share_per_year` | fixed O&M share | number | share/year | Optional | defaults to `0.0` |
| `generator.technical.nominal_efficiency_full_load` | full-load efficiency | number | - | Yes | used directly in constant-efficiency mode |
| `generator.technical.efficiency_curve_csv` | shared curve file | string or `null` | path | Optional | activates generator curve mode when non-empty |
| `generator.technical.max_installable_capacity_kw` | planning upper bound | number or `null` | kW | Optional | `null` means no explicit cap |
| `fuel.by_scenario.<scenario>.lhv_kwh_per_unit_fuel` | lower heating value | number | kWh per unit fuel | Yes | scenario-specific block required for every scenario |
| `fuel.by_scenario.<scenario>.direct_emissions_kgco2e_per_unit_fuel` | direct emissions | number | kgCO2e per unit fuel | Yes | scope 1 emissions |
| `fuel.by_scenario.<scenario>.fuel_cost_per_unit_fuel` | fuel price | number | currency per unit fuel | Yes | can differ by scenario |

Typical Year no longer accepts the old `generator.operation` block. Use the shared `generator.technical.efficiency_curve_csv` path and keep fixed O&M in `generator.investment.by_step.<step>`.

### 7.5 `inputs/grid.yaml` (on-grid only)

Typical structure:

```yaml
grid:
  by_scenario:
    scenario_1:
      line:
        capacity_kw: 500
        transmission_efficiency: 1.0
        renewable_share: 0.0
        emissions_factor_kgco2e_per_kwh: 0.0
      outages:
        average_outages_per_year: 60
        average_outage_duration_minutes: 180
        outage_scale_od_hours: 0.6
        outage_shape_od: 0.56
        outage_seed: 0
```

| Path | Meaning | Type | Unit | Required | Notes |
| --- | --- | --- | --- | --- | --- |
| `line.capacity_kw` | interconnection limit | number | kW | Yes | used for import and export limit |
| `line.transmission_efficiency` | grid delivery efficiency | number | - | Yes | must stay in `[0,1]` |
| `line.renewable_share` | renewable share of imports | number | share | Optional | defaults to `0.0`; must stay in `[0,1]` |
| `line.emissions_factor_kgco2e_per_kwh` | scope 2 emissions factor | number | kgCO2e/kWh | Optional | defaults to `0.0`; must be non-negative |
| `outages.average_outages_per_year` | outage frequency | number | events/year | Yes | `<= 0` effectively gives always-available grid |
| `outages.average_outage_duration_minutes` | average outage duration | number | minutes/event | Yes | used by simulator |
| `outages.outage_scale_od_hours` | Weibull outage-duration scale | number | hours | Yes | must be `> 0` |
| `outages.outage_shape_od` | Weibull outage-duration shape | number | - | Yes | must be `> 0` |
| `outages.outage_seed` | deterministic random seed | integer-like number | - | Optional | defaults to `0` |

`grid_availability.csv` is generated from this file, not maintained as a primary input.

## 8. Detailed Time-Series Reference

### `load_demand.csv`

- two-row header,
- first column must be `meta/hour`,
- one column per `(scenario, typical_year)`,
- exactly 8760 rows,
- hour index must be `0..8759`,
- units are `kWh` per hour.

Example:

```csv
meta,scenario_1,scenario_2
hour,typical_year,typical_year
0,12.4,11.8
1,11.9,11.2
```

### `resource_availability.csv`

- three-row header,
- first column must be `meta/hour`,
- one column per `(scenario, typical_year, resource)`,
- exactly 8760 rows,
- values must be numeric,
- units are dimensionless capacity factor.

Example:

```csv
meta,scenario_1,scenario_1
hour,typical_year,typical_year
,Solar,Wind
0,0.00,0.45
1,0.00,0.44
```

### `grid_import_price.csv` and `grid_export_price.csv`

- same two-row structure as `load_demand.csv`,
- units are currency per kWh.

### `generator_efficiency_curve.csv`

Required columns:

- `Relative Power Output [-]`
- `Efficiency [-]`

Rules enforced by the parser:

- numeric values only,
- at least 2 points,
- relative power strictly increasing,
- relative power inside `[0,1]`,
- last relative-power point must be exactly `1.0`.

Zero-load convention:

- the parser always normalizes the internal curve to a single zero-output anchor,
- users may include an explicit `0.0` row or provide only positive-load points,
- the generated helper CSV includes the `0.0` row for clarity.

### `battery_efficiency_curve.csv`

Required columns:

- `relative_power_pu`
- `charge_efficiency`
- `discharge_efficiency`

Current template semantics:

- `relative_power_pu` is relative DC-side power,
- charge/discharge columns are normalized multipliers around the scalar full-load efficiencies in `battery.yaml`,
- the full-load row should be `1.0`.

## 9. Conditional Inputs and Advanced Options

### On-grid projects

Additional required files:

- `grid.yaml`
- `grid_import_price.csv`
- `grid_export_price.csv` if export is enabled

Derived file:

- `grid_availability.csv`

### Battery advanced loss model

Activated by:

- `formulation.json -> battery_model.loss_model = "convex_loss_epigraph"`

Then:

- `battery.technical.efficiency_curve_csv` must point to a valid file,
- that file must exist under `inputs/` unless an absolute path is used.

### Generator efficiency-curve mode

Activated by:

- `formulation.json -> generator_model.efficiency_model = "efficiency_curve"`

Then:

- `generator.technical.efficiency_curve_csv` must point to a valid file,
- the same shared curve applies across all scenarios.

## 10. Consistency Checks and Common Mistakes

Before solving, verify:

1. Scenario labels are identical in `formulation.json`, all scenario CSV headers, `fuel.by_scenario`, and `grid.by_scenario` if on-grid.
2. Resource labels are identical in `renewables.yaml` and row 3 of `resource_availability.csv`.
3. Typical Year time-series files use `typical_year` as the year header.
4. All hourly files have exactly 8760 rows and hour index `0..8759`.
5. Mandatory numeric cells are not blank.
6. If on-grid is enabled, `grid_import_price.csv` exists.
7. If export is enabled, `grid_export_price.csv` exists.
8. If battery advanced loss mode is enabled, `battery.technical.efficiency_curve_csv` is not empty.
9. If generator curve mode is enabled, the referenced generator curve file exists and ends at relative power `1.0`.
10. If land limit is active, provide finite `specific_area_m2_per_kw` values in `renewables.yaml`.
11. If all supply options are disabled or capped too tightly while demand is positive, the model can become infeasible unless lost load is allowed.
12. If `max_lost_load_fraction = 0`, `lost_load_cost_per_kwh` does not make an infeasible model feasible.

## 11. Minimal Working Examples

### Example A: simplest off-grid Typical Year case

Settings:

- Typical-year formulation
- Off-grid
- Single scenario
- One renewable resource
- Constant battery efficiency
- Constant generator efficiency

Required files:

- `formulation.json`
- `load_demand.csv`
- `resource_availability.csv`
- `renewables.yaml`
- `battery.yaml`
- `generator.yaml`

### Example B: weak-grid Typical Year case

Settings:

- Typical-year formulation
- On-grid
- Optional export on or off
- Single scenario or multi-scenario

Additional required files:

- `grid.yaml`
- `grid_import_price.csv`
- `grid_export_price.csv` if export is enabled

## 12. Appendix

### Glossary

| Term | Meaning in Typical Year |
| --- | --- |
| `period` | hourly index `0..8759` |
| `scenario` | uncertainty case label |
| `resource` | renewable resource label |
| `typical_year` | fixed year-header string used in Typical Year templates |

### Mapping between UI choices and generated files

| UI choice | File impact |
| --- | --- |
| renewable count and labels | changes `renewables.yaml` and `resource_availability.csv` |
| multi-scenario toggle and labels | changes scenario headers and YAML scenario mappings |
| on-grid toggle | adds grid YAML and tariff CSVs |
| export toggle | adds export tariff CSV |
| battery loss model | may add battery efficiency-curve CSV |
| generator efficiency model | may add generator efficiency-curve CSV |
| land-limit toggle | writes finite or null `land_availability_m2` in `formulation.json` |
| lost-load cap | writes `max_lost_load_fraction` and optionally `lost_load_cost_per_kwh` |

### Source basis

This guide was reconstructed from the current implementation, mainly:

- `pages/0_Project_Setup.py`
- `pages/1_Data_Audit_and_Visualization.py`
- `pages/3_Optimization.py`
- `pages/4_Results.py`
- `core/io/templates.py`
- `core/typical_year_model/sets.py`
- `core/data_pipeline/typical_year_loader.py`
- `core/data_pipeline/typical_year_parsing.py`

Where behavior was only implicit in parser logic, the guide states or assumes the implementation-visible behavior rather than older documentation.
