# MicroGridsPy Planning - Streamlit Optimization Tool

MicroGridsPy Planning is a bottom-up, open-source optimization tool for the techno-economic planning of mini-grid energy systems in remote and underserved areas. The model is implemented in Python using Linopy and provides a transparent framework for system sizing, dispatch optimization, and long-term planning under uncertainty.

The tool is designed as an interactive Streamlit application guiding the user through a structured workflow from problem definition to optimization results.

The reference energy system includes renewable generation, battery storage, backup generators, and optional grid connection, enabling analysis of both off-grid and weak-grid configurations. :contentReference[oaicite:0]{index=0}

---

## Intended Workflow and User Experience

The application is organized as a pipeline of Streamlit pages that progressively define, validate, and solve a planning problem.

### 1. Project Setup — Global Settings

The initial page defines the overall characteristics of the planning problem:

- System configuration (off-grid or grid-connected)
- Technologies included (renewables, storage, generators, grid)
- Planning mode (Typical-Year or Multi-Year)
- Temporal resolution and problem dimensions
- Scenario structure (deterministic or multi-scenario)
- Economic parameters and modelling options

These settings determine the structure of the optimization problem and are saved as a JSON configuration file.

---

### 2. Input Template Generation

Based on the selected configuration, the tool generates a set of structured input templates (CSV/YAML/JSON) tailored to the problem dimensions and enabled components.

The user fills these templates with:

- Time-series data (load, renewable resources, grid availability)
- Technology parameters (costs, efficiencies, lifetimes)
- Economic assumptions
- Scenario definitions

This ensures consistency between user inputs and the mathematical model.

---

### 3. Input Visualization

A dedicated page allows inspection and validation of input data before optimization:

- Time-series plots (load, resources, prices, outages)
- Scenario comparison
- Data consistency checks

This step reduces modelling errors and improves transparency.

---

### 4. Optimization Execution

The optimization page builds and solves the model using:

- HiGHS (open-source solver)
- Gurobi (commercial solver, optional)

The model performs optimal system sizing and dispatch to minimize system cost subject to technical and economic constraints.

---

### 5. Results Exploration

Results include:

- Optimal capacities of all technologies
- Dispatch profiles
- Cost breakdown
- Energy balances
- Reliability indicators
- Scenario statistics

---

## Planning Modes

MicroGridsPy supports two complementary modelling formulations, representing different levels of temporal realism and computational complexity:

- Typical-Year Planning (steady-state)
- Multi-Year Planning (dynamic capacity expansion)

Both formulations are investment-oriented and minimize expected system cost across scenarios. :contentReference[oaicite:1]{index=1}

---

## Typical-Year Planning Mode (Steady-State)

The Typical-Year model represents the system using a single representative year that repeats indefinitely.

Key assumptions:

- Stationary demand and resource conditions
- No explicit capacity expansion
- No degradation or learning effects
- System operates in long-term equilibrium

Investment costs are annualized using the Capital Recovery Factor (CRF), converting capital expenditure into an equivalent annual payment.

The objective is to minimize the Expected Equivalent Annual Cost (EAC):

- Annualized investment costs
- Expected operational costs across scenarios
- Externalities (e.g., emissions, unserved energy)

This formulation is computationally efficient and suitable for:

- Feasibility studies
- Technology comparison
- Systems with stable long-term conditions
- Large stochastic scenario sets

In this steady-state interpretation, assets are implicitly replaced indefinitely, producing a perpetual replacement logic equivalent to infinite discounted reinvestments. :contentReference[oaicite:2]{index=2}

---

## Multi-Year Planning Mode (Dynamic Formulation)

The Multi-Year model explicitly simulates a planning horizon composed of multiple years.

It is formulated as a stochastic capacity-expansion problem where investments and operations evolve over time.

Key features include:

### Capacity Expansion

- Investments can occur at multiple stages
- Installed capacity is non-decreasing
- Technology roll-out and replacement cycles are modeled

### Intertemporal Economics

All costs are evaluated in present value terms using a dual-rate logic:

- Technology-specific WACC governs capital recovery
- A social discount rate is used for system-level valuation

This separates financial costs from societal time preferences.

### Residual Value (Salvage)

Assets that outlive the planning horizon retain residual value, preventing bias against long-lived technologies and ensuring economic consistency.

### Stochastic Operation

Operational decisions are scenario-dependent, while investment decisions are shared across scenarios to ensure robust system design.

---

### Objective

The dynamic formulation minimizes the Expected Net Present Cost (NPC) of the system, including:

- Investment annuities
- Operational expenditures
- Externalities
- Salvage value of remaining assets

This approach captures demand evolution, aging, financing structure, and long-term trade-offs that cannot be represented in steady-state models. :contentReference[oaicite:3]{index=3}

---

## Mathematical Structure

Both planning modes share a bottom-up cost accounting framework:

- Capacity sizing decisions determine investment costs
- Dispatch decisions determine operational costs
- Externalities can be internalized
- Energy balance constraints ensure feasibility

Hourly resolution is typically used for operations, and multiple scenarios can represent uncertainty in demand, resources, or grid conditions. :contentReference[oaicite:4]{index=4}

---

## Solvers

Supported optimization solvers:

- **HiGHS** — open-source linear solver (default)
- **Gurobi** — commercial solver (optional)
