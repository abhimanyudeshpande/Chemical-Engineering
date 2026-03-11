# Batch Reactor Kinetics Simulator

A Python simulation tool for modeling consecutive first-order reactions (A → B → C) in a batch reactor. Implements and compares two numerical ODE solvers — Forward Euler and Midpoint (RK2) — and computes engineering metrics including final conversion, time to 90% conversion, peak intermediate concentration, and mass balance verification. Built as an introduction to reaction kinetics and numerical methods for chemical engineering.

---

## What This Models

The simulator solves the following reaction system:

```
A --k1--> B --k2--> C
```

Where:
- **A** is the starting reactant (decays exponentially)
- **B** is an intermediate product (rises then falls — forms a hump)
- **C** is the final product (increases monotonically toward a plateau)

The governing ODEs are:

```
dA/dt = -k1 * A
dB/dt =  k1 * A  -  k2 * B
dC/dt =  k2 * B
```

This reaction pattern appears in real-world systems including drug metabolism, industrial intermediate synthesis, and staged polymerization.

---

## Features

- **Two numerical methods** — Forward Euler and Midpoint (RK2) simulated simultaneously
- **Side-by-side comparison** — both methods plotted on the same axes per species so accuracy differences are immediately visible
- **Engineering metrics** printed to the terminal:
  - Final conversion of A
  - Time to 90% conversion of A
  - Peak concentration of B and the time it occurs
  - Final concentrations of A, B, and C
  - Mass balance check (A + B + C should equal A₀)
  - Time when [B] first exceeds a user-defined threshold
- **Error analysis** — max absolute difference between Euler and Midpoint for each species
- **Simple config block** — all inputs live in one clearly labeled section, nothing else needs to be touched

---

## Files

| File | Description |
|------|-------------|
| `batch_reactor_abc.py` | Core simulator: A → B → C with Euler vs Midpoint comparison |
| `batch_reactor_arrhenius.py` | Extension: same reaction with temperature-dependent rate constants via the Arrhenius equation |
| `batch_reactor_ab.py` | Simpler A → B simulator used for timestep convergence analysis |

---

## Inputs (Config Block)

All parameters are set in the `CONFIG` block at the top of `main()`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `a0_m` | Initial concentration of A (mol/L) | `1.0` |
| `k1_per_s` | Rate constant for A → B (1/s) | `0.015` |
| `k2_per_s` | Rate constant for B → C (1/s) | `0.006` |
| `t_end_s` | Total simulation time (s) | `600.0` |
| `dt_s` | Timestep size (s) — smaller is more accurate | `5.0` |
| `b_threshold_m` | Concentration threshold for [B] reporting (mol/L) | `0.2` |

---

## Example Output

```
--- Batch Reactor Summary: A → B → C (Midpoint method) ---
k1 = 0.0150 1/s,  k2 = 0.0060 1/s,  dt = 5.0 s
Final conversion of A  : 99.988%
Time to 90% conversion : 154.00 s
Peak [B]               : 0.5429 M  at  t = 102.00 s
Final concentrations   : A = 0.0001 M,  B = 0.0453 M,  C = 0.9545 M
Mass balance (A+B+C)   : 1.0000 M  (should be ~1.0000 M)
[B] first exceeds 0.20 M  at  t = 16.00 s

--- Euler vs Midpoint: Max Absolute Difference ---
  Species A : 0.003241 M
  Species B : 0.001876 M
  Species C : 0.002104 M
  (dt = 5.0 s — reduce dt to shrink these errors)
```

---

## How to Run

**Requirements:** Python 3.9+, NumPy, Matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

Run the simulator:
```bash
python batch_reactor_abc.py
```

---

## Numerical Methods

**Forward Euler** is the simplest ODE integration method. At each timestep it freezes the slope at the current state and steps straight forward. It is fast but accumulates error, especially with large timesteps.

**Midpoint (RK2)** improves on Euler by taking a half-step first, evaluating the slope at that midpoint, and using that slope for the full step. This extra evaluation makes it significantly more accurate for the same timestep size.

A key learning feature of this project is that you can directly observe the tradeoff: increasing `dt_s` makes the Euler and Midpoint curves diverge visibly, while decreasing `dt_s` makes them converge. The error table quantifies this difference numerically.

---

## Physical Behavior to Observe

- **[A]** decays exponentially — faster with larger k1
- **[B]** forms a hump — peak shifts earlier if k2 >> k1 (B consumed quickly), later if k1 >> k2
- **[C]** grows monotonically toward 1.0 M
- **Mass balance** (A + B + C) stays constant at A₀ — a built-in sanity check

If k2 >> k1, B barely accumulates and its peak is very small. If k1 >> k2, B builds up significantly before being consumed. Changing these rate constants in the config block lets you explore both regimes.

---

## Arrhenius Extension

`batch_reactor_arrhenius.py` replaces the fixed k1 and k2 values with the Arrhenius equation:

```
k(T) = A * exp( -Ea / (R * T) )
```

Where **A** is the pre-exponential factor, **Ea** is the activation energy (J/mol), **R** is the gas constant (8.314 J/mol·K), and **T** is temperature in Kelvin. Rate constants are no longer manually set — they are calculated automatically from temperature.

The config block lets you set a list of temperatures to compare. The simulator runs a full A → B → C simulation at each temperature and produces two plots:

- **Plot 1** — concentration curves for all species at all temperatures on one graph, colored by temperature
- **Plot 2** — two panels showing how peak [B] concentration and the time it occurs shift as temperature increases

The key insight this reveals: because Ea2 > Ea1 in the default config, raising temperature speeds up B → C *more* than A → B. This means higher temperature actually reduces how much B accumulates — a real selectivity tradeoff that chemical engineers have to manage in practice.

### Arrhenius Config Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `a1_factor` | Pre-exponential factor for A → B (1/s) | `1.0e6` |
| `ea1_j_per_mol` | Activation energy for A → B (J/mol) | `50000` |
| `a2_factor` | Pre-exponential factor for B → C (1/s) | `1.0e6` |
| `ea2_j_per_mol` | Activation energy for B → C (J/mol) | `60000` |
| `temperatures_k` | List of temperatures to simulate (K) | `[300, 320, 340, 360]` |

Run it with:
```bash
python batch_reactor_arrhenius.py
```
