from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# RESULTS CONTAINER
# Holds the time array and concentrations for one simulation run.
# =============================================================================

@dataclass(frozen=True)
class BatchABCResult:
    t_s: np.ndarray   # time points (s)
    a_m: np.ndarray   # concentration of A (M)
    b_m: np.ndarray   # concentration of B (M)
    c_m: np.ndarray   # concentration of C (M)


# =============================================================================
# SHARED HELPERS
# =============================================================================

R_J_PER_MOL_K: float = 8.314   # universal gas constant (J / mol·K) — never changes

def arrhenius(
    a_factor: float, ea_j_per_mol: float, temp_k: float
) -> float:
    """
    Compute a rate constant at a given temperature using the Arrhenius equation:

        k(T) = A * exp( -Ea / (R * T) )

    Where:
        A          = pre-exponential factor (same units as k, i.e. 1/s here)
        Ea         = activation energy (J/mol) — how much energy the reaction needs
        R          = 8.314 J/(mol·K)
        T          = temperature (K)

    Higher temperature → larger k → faster reaction.
    Higher Ea → k is more sensitive to temperature changes.
    """
    return a_factor * np.exp(-ea_j_per_mol / (R_J_PER_MOL_K * temp_k))


def _validate_inputs(
    a0_m: float, t_end_s: float, dt_s: float, temp_k: float
) -> None:
    if a0_m <= 0:
        raise ValueError("a0_m must be > 0")
    if t_end_s <= 0:
        raise ValueError("t_end_s must be > 0")
    if dt_s <= 0 or dt_s >= t_end_s:
        raise ValueError("dt_s must be > 0 and < t_end_s")
    if temp_k <= 0:
        raise ValueError("temp_k must be > 0 (temperature in Kelvin)")


def _make_time_grid(t_end_s: float, dt_s: float) -> tuple[int, np.ndarray]:
    n_steps: int = int(np.floor(t_end_s / dt_s)) + 1
    t_s: np.ndarray = np.linspace(0.0, dt_s * (n_steps - 1), n_steps)
    return n_steps, t_s


def _rhs(
    a: float, b: float, k1: float, k2: float
) -> tuple[float, float, float]:
    """
    Instantaneous reaction rates:
        dA/dt = -k1 * A
        dB/dt =  k1 * A  -  k2 * B
        dC/dt =  k2 * B
    """
    da_dt: float = -k1 * a
    db_dt: float = (k1 * a) - (k2 * b)
    dc_dt: float =  k2 * b
    return da_dt, db_dt, dc_dt


# =============================================================================
# SIMULATOR (Midpoint / RK2 method)
# Takes Arrhenius parameters instead of fixed k values.
# k1 and k2 are computed from temperature inside this function.
# =============================================================================

def simulate_midpoint_arrhenius(
    *,
    a0_m:        float,   # initial concentration of A (M)
    a1_factor:   float,   # pre-exponential factor for A → B (1/s)
    ea1_j_per_mol: float, # activation energy for A → B (J/mol)
    a2_factor:   float,   # pre-exponential factor for B → C (1/s)
    ea2_j_per_mol: float, # activation energy for B → C (J/mol)
    temp_k:      float,   # reactor temperature (K)
    t_end_s:     float,   # total simulation time (s)
    dt_s:        float,   # timestep (s)
) -> BatchABCResult:
    """
    Simulate A -> B -> C using the Midpoint (RK2) method, with rate constants
    computed from the Arrhenius equation at a fixed reactor temperature.
    """
    _validate_inputs(a0_m, t_end_s, dt_s, temp_k)
    n_steps, t_s = _make_time_grid(t_end_s, dt_s)

    # Compute k1 and k2 from temperature — this is the Arrhenius step
    k1: float = arrhenius(a1_factor, ea1_j_per_mol, temp_k)
    k2: float = arrhenius(a2_factor, ea2_j_per_mol, temp_k)

    a_m = np.zeros(n_steps, dtype=float)
    b_m = np.zeros(n_steps, dtype=float)
    c_m = np.zeros(n_steps, dtype=float)
    a_m[0] = a0_m

    for i in range(1, n_steps):
        a = a_m[i - 1]
        b = b_m[i - 1]
        c = c_m[i - 1]

        da1, db1, dc1 = _rhs(a, b, k1, k2)

        a_mid = a + da1 * (dt_s / 2.0)
        b_mid = b + db1 * (dt_s / 2.0)
        c_mid = c + dc1 * (dt_s / 2.0)

        da2, db2, dc2 = _rhs(a_mid, b_mid, k1, k2)

        a_m[i] = max(a + da2 * dt_s, 0.0)
        b_m[i] = max(b + db2 * dt_s, 0.0)
        c_m[i] = max(c + dc2 * dt_s, 0.0)

    return BatchABCResult(t_s=t_s, a_m=a_m, b_m=b_m, c_m=c_m)


# =============================================================================
# METRICS
# =============================================================================

def time_to_conversion_a(
    *, t_s: np.ndarray, a_m: np.ndarray, a0_m: float, target_conversion: float
) -> float:
    if not (0.0 < target_conversion < 1.0):
        raise ValueError("target_conversion must be between 0 and 1 (exclusive)")
    x_a = (a0_m - a_m) / a0_m
    idx = int(np.argmax(x_a >= target_conversion))
    if x_a[idx] < target_conversion:
        raise ValueError("Target conversion not reached within the simulation window.")
    return float(t_s[idx])


def peak_b(*, t_s: np.ndarray, b_m: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmax(b_m))
    return float(b_m[idx]), float(t_s[idx])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    # =========================================================================
    # CONFIG — change anything in this block
    # =========================================================================

    a0_m:   float = 1.0      # initial concentration of A (M)
    t_end_s: float = 600.0   # total simulation time (s)
    dt_s:   float = 1.0      # timestep (s)

    # --- Arrhenius parameters for reaction 1: A → B ---
    a1_factor:     float = 1.0e6    # pre-exponential factor (1/s)
    ea1_j_per_mol: float = 50_000   # activation energy (J/mol) — ~50 kJ/mol is typical

    # --- Arrhenius parameters for reaction 2: B → C ---
    a2_factor:     float = 1.0e6    # pre-exponential factor (1/s)
    ea2_j_per_mol: float = 60_000   # activation energy (J/mol) — higher than Ea1

    # --- Temperatures to compare (in Kelvin: 0°C = 273 K, 25°C = 298 K) ---
    # Try adding or removing temperatures from this list
    temperatures_k: list[float] = [300, 320, 340, 360]

    # Threshold: print when [B] first crosses this concentration
    b_threshold_m: float = 0.2   # (M)
    # =========================================================================

    # --- Pre-print the k values at each temperature so you can see the effect ---
    print("\n--- Arrhenius: Rate Constants vs Temperature ---")
    print(f"{'Temp (K)':<12} {'Temp (°C)':<12} {'k1 (1/s)':<16} {'k2 (1/s)':<16} {'k1/k2 ratio'}")
    for T in temperatures_k:
        k1 = arrhenius(a1_factor, ea1_j_per_mol, T)
        k2 = arrhenius(a2_factor, ea2_j_per_mol, T)
        print(f"{T:<12.1f} {T - 273.15:<12.1f} {k1:<16.6f} {k2:<16.6f} {k1/k2:.4f}")

    # --- Run a simulation at each temperature and collect results ---
    results: list[tuple[float, BatchABCResult]] = []

    for T in temperatures_k:
        res = simulate_midpoint_arrhenius(
            a0_m=a0_m,
            a1_factor=a1_factor,
            ea1_j_per_mol=ea1_j_per_mol,
            a2_factor=a2_factor,
            ea2_j_per_mol=ea2_j_per_mol,
            temp_k=T,
            t_end_s=t_end_s,
            dt_s=dt_s,
        )
        results.append((T, res))

    # --- Print summary for each temperature ---
    print("\n--- Batch Reactor Summary at Each Temperature ---")
    print(f"{'Temp (K)':<10} {'Conv. of A':<14} {'t90 (s)':<12} {'Peak [B] (M)':<16} {'t_peak (s)'}")

    for T, res in results:
        a_final = float(res.a_m[-1])
        x_final = (a0_m - a_final) / a0_m
        b_max, t_bmax = peak_b(t_s=res.t_s, b_m=res.b_m)

        try:
            t90 = time_to_conversion_a(t_s=res.t_s, a_m=res.a_m, a0_m=a0_m, target_conversion=0.90)
            t90_str = f"{t90:.1f}"
        except ValueError:
            # At very low temperatures, 90% conversion might not be reached in time
            t90_str = "not reached"

        print(f"{T:<10.1f} {x_final:<14.3%} {t90_str:<12} {b_max:<16.4f} {t_bmax:.1f}")

    # --- Threshold check at each temperature ---
    print(f"\n--- Time when [B] first exceeds {b_threshold_m:.2f} M ---")
    for T, res in results:
        mask = res.b_m >= b_threshold_m
        if np.any(mask):
            t_hit = float(res.t_s[int(np.argmax(mask))])
            print(f"  T = {T:.0f} K : {t_hit:.2f} s")
        else:
            print(f"  T = {T:.0f} K : [B] never reaches {b_threshold_m:.2f} M")

    # -------------------------------------------------------------------------
    # PLOT 1 — concentration curves at each temperature (one line per species
    #          per temperature, colored by temperature)
    # -------------------------------------------------------------------------
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(temperatures_k)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for (T, res), color in zip(results, colors):
        label_a = f"[A]  T={T:.0f}K"
        label_b = f"[B]  T={T:.0f}K"
        label_c = f"[C]  T={T:.0f}K"

        ax.plot(res.t_s, res.a_m, linestyle=":",  color=color, alpha=0.6, label=label_a)
        ax.plot(res.t_s, res.b_m, linestyle="-",  color=color,            label=label_b)
        ax.plot(res.t_s, res.c_m, linestyle="--", color=color, alpha=0.6, label=label_c)

        # Mark peak B for each temperature
        peak_idx = int(np.argmax(res.b_m))
        ax.scatter([res.t_s[peak_idx]], [res.b_m[peak_idx]], color=color, zorder=5, s=50)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration (M)")
    ax.set_title("Batch Reactor: A → B → C at Multiple Temperatures (Arrhenius)")
    ax.legend(fontsize=7, ncol=2, loc="center right")
    ax.grid(True)
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # PLOT 2 — peak [B] vs temperature
    # Shows the selectivity tradeoff: higher T doesn't always mean more B
    # -------------------------------------------------------------------------
    peak_b_values = [peak_b(t_s=res.t_s, b_m=res.b_m)[0] for _, res in results]
    peak_b_times  = [peak_b(t_s=res.t_s, b_m=res.b_m)[1] for _, res in results]

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 4))

    ax2a.plot(temperatures_k, peak_b_values, "o-", color="darkorange")
    ax2a.set_xlabel("Temperature (K)")
    ax2a.set_ylabel("Peak [B] (M)")
    ax2a.set_title("Peak Concentration of B vs Temperature")
    ax2a.grid(True)

    ax2b.plot(temperatures_k, peak_b_times, "o-", color="steelblue")
    ax2b.set_xlabel("Temperature (K)")
    ax2b.set_ylabel("Time of Peak [B] (s)")
    ax2b.set_title("Time of Peak B vs Temperature")
    ax2b.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()