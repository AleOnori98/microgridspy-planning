# Multi-Year Mapping Notes

Internal map used to reconstruct the Multi-Year guide.

## UI and workflow

- `Home.py`: shared landing page
- `pages/0_Project_Setup.py`: writes `formulation.json` and generates templates
- `pages/1_Data_Audit_and_Visualization.py`: initializes Multi-Year sets, loads dataset, regenerates dynamic grid availability
- `pages/3_Optimization.py`: builds `MultiYearModel` when `core_formulation = "dynamic"`
- `pages/4_Results.py`: delegates to Multi-Year results rendering/export

## Template generation

- `core/io/templates.py`
  - `TemplateSettings`
  - `_safe_year_labels`
  - `_safe_step_keys`
  - `_write_load_demand_csv`
  - `_write_resource_availability_csv`
  - `_write_renewables_yaml`
  - `_write_battery_yaml`
  - `_write_generator_yaml`
  - `_write_grid_yaml`
  - `_write_battery_efficiency_curve_csv`
  - `_write_battery_calendar_fade_curve_csv`
  - `_write_generator_efficiency_curve_csv`

## Set and dimension logic

- `core/multi_year_model/sets.py`
  - `initialize_sets`
  - builds `year`, `inv_step`, `year_inv_step`, `inv_active_in_year`
- `core/multi_year_model/lifecycle.py`
  - cohort activity, replacement timing, repeated degradation factors

## Data loading

- `core/data_pipeline/multi_year_loader.py`
- `core/multi_year_model/data.py`
  - `_load_load_demand_csv`
  - `_load_resource_availability_csv`
  - `_load_renewables_yaml`
  - `_load_battery_yaml`
  - `_load_generator_and_fuel_yaml`
  - `_load_price_csv_dynamic`
  - `_load_grid_yaml_dynamic`
  - `regenerate_grid_availability_dynamic`
  - `_initialize_data_legacy`

## Optional advanced input parsers

- `core/data_pipeline/battery_loss_model.py`
- `core/data_pipeline/battery_degradation_model.py`
- `core/data_pipeline/battery_calendar_fade_model.py`
- `core/data_pipeline/generator_partial_load_model.py`

## Results and exports

- `core/export/multi_year_results.py`
- `core/visualization/multi_year_results_page.py`

## Key implementation facts

- `start_year_label` is enforced as an integer-like calendar year in the current Multi-Year workflow
- Multi-Year templates now use canonical numeric step labels such as `"1"`, `"2"`, ... and the loader keeps only lightweight alias support for backward compatibility
- Multi-Year uses shared technical parameters across steps and step-dependent investment data
- yearly fuel costs are stored as scenario-keyed lists aligned with the `year` coordinate
- battery degradation ownership is split intentionally as:
  - activation flags in `formulation.json`
  - battery-specific degradation parameters in `battery.yaml`
- grid availability is generated from outage parameters and first-year connection timing
