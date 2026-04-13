# Inputs folder

This folder contains user-editable input templates.

## load_demand.csv
- Hourly **load demand** template (8760 rows).
- Units: **kWh per hour** (energy during each hour).
- **Two-row header**:
  - Row 1: scenario labels
  - Row 2: year labels
- A meta column `meta/hour` provides the hour index (0..8759).

Scenarios: scenario_1

Years: 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035

## resource_availability.csv
- Hourly **resource availability** template (8760 rows).
- Units: **capacity factor** (per unit of nominal capacity, typically 0..1).
- **Three-row header**:
  - Row 1: scenario labels
  - Row 2: year labels
  - Row 3: resource labels
- A meta column `meta/hour` provides the hour index (0..8759).

Scenarios: scenario_1

Years: 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035

Resources: Solar

## renewables.yaml
- Renewable techno-economic parameters.
- Parameters can vary by resource and, for investment-side data, by investment step.


Renewables Technologies: Solar_PV


## battery.yaml
- Battery techno-economic parameters.
- In the multi-year formulation, battery investment data are written by step while technical parameters remain shared.
- Battery-owned degradation controls are written in `battery.technical` only for the enabled endogenous degradation modes.

## battery_efficiency_curve.csv
- Optional battery conversion-efficiency curve used only when `formulation.json -> battery_model.loss_model = convex_loss_epigraph`.
- Preferred semantics: the CSV stores normalized efficiency multipliers relative to the scalar efficiencies in `battery.yaml`, with the full-load row equal to `1.0`.
- Legacy absolute-efficiency curves are still accepted for backward compatibility.
- Columns:
  - `relative_power_pu`: relative DC-side battery power in (0,1]
  - `charge_efficiency`: normalized charge-efficiency multiplier (actual eta = `battery.technical.charge_efficiency * charge_efficiency`)
  - `discharge_efficiency`: normalized discharge-efficiency multiplier (actual eta = `battery.technical.discharge_efficiency * discharge_efficiency`)

## battery_calendar_fade_curve.csv
- Optional SoC-dependent calendar-fade coefficient curve used when calendar fade is enabled.
- Columns:
  - `soc_pu`: state of charge normalized by the cohort nominal available energy reference used by the LP surrogate
  - `calendar_fade_coefficient_per_step`: non-negative coefficient applied to the configured time increment per modeled step
- For v1, the curve is evaluated against a fixed cohort reference to preserve LP compatibility.
- In the current multi-year implementation, endogenous degradation directly reduces usable battery energy capacity and the associated power limits remain proportional to that degraded effective capacity. Replacement timing still follows calendar lifetime.

## generator.yaml
- Generator and fuel techno-economic parameters.
- In the multi-year formulation, generator investment data are written by step while technical and fuel-physics data remain shared. Yearly fuel prices remain scenario-based.
- `generator.technical.efficiency_curve_csv` is automatically set from the Project Setup choice:
  - `null` for constant generator efficiency in partial load
  - `generator_efficiency_curve.csv` for efficiency-curve mode
