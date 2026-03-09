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
    t_s:  np.ndarray   # time points (s)
    a_m:  np.ndarray   # concentration of A (M)
    b_m:  np.ndarray   # concentration of B (M)
    c_m:  np.ndarray   # concentration of C (M)


# =============================================================================
# SHARED HELPERS
# These are used internally by both simulators — you don't need to touch them.
# =============================================================================

def _validate_inputs(
    a0_m: float, k1_per_s: float, k2_per_s: float, t_end_s: float, dt_s: float
) -> None:
    """Raise a clear error if any input value is physically impossible."""
    if a0_m <= 0:
        raise ValueError("a0_m must be > 0")
    if k1_per_s < 0 or k2_per_s < 0:
        raise ValueError("k1_per_s and k2_per_s must be >= 0")
    if t_end_s <= 0:
        raise ValueError("t_end_s must be > 0")
    if dt_s <= 0 or dt_s >= t_end_s:
        raise ValueError("dt_s must be > 0 and < t_end_s")


def _make_time_grid(t_end_s: float, dt_s: float) -> tuple[int, np.ndarray]:
    """Build a uniform time grid and return (number_of_steps, time_array)."""
    n_steps: int = int(np.floor(t_end_s / dt_s)) + 1
    t_s: np.ndarray = np.linspace(0.0, dt_s * (n_steps - 1), n_steps)
    return n_steps, t_s


def _rhs(
    a: float, b: float, k1: float, k2: float
) -> tuple[float, float, float]:
    """
    The right-hand side of the ODE system — i.e. the instantaneous rates:
        dA/dt = -k1 * A
        dB/dt =  k1 * A  -  k2 * B
        dC/dt =  k2 * B
    """
    da_dt: float = -k1 * a
    db_dt: float = (k1 * a) - (k2 * b)
    dc_dt: float =  k2 * b
    return da_dt, db_dt, dc_dt


# =============================================================================
# SIMULATORS
# Two separate functions — one per numerical method.
# Both accept the same inputs and return a BatchABCResult.
# =============================================================================

def simulate_euler(
    *,
    a0_m: float,
    k1_per_s: float,
    k2_per_s: float,
    t_end_s: float,
    dt_s: float,
) -> BatchABCResult:
    """
    Simulate A -> B -> C using the Euler method.

    Euler is the simplest approach:
      - Look at the slope right now
      - Take a straight-line step of size dt in that direction
      - Repeat

    It's fast but less accurate, especially with large dt.
    """
    _validate_inputs(a0_m, k1_per_s, k2_per_s, t_end_s, dt_s)
    n_steps, t_s = _make_time_grid(t_end_s, dt_s)

    a_m = np.zeros(n_steps, dtype=float)
    b_m = np.zeros(n_steps, dtype=float)
    c_m = np.zeros(n_steps, dtype=float)

    a_m[0] = a0_m  # everything starts as A

    for i in range(1, n_steps):
        a = a_m[i - 1]
        b = b_m[i - 1]
        c = c_m[i - 1]

        # Slope at the start of this step
        da, db, dc = _rhs(a, b, k1_per_s, k2_per_s)

        # Step straight forward
        a_m[i] = max(a + da * dt_s, 0.0)
        b_m[i] = max(b + db * dt_s, 0.0)
        c_m[i] = max(c + dc * dt_s, 0.0)

    return BatchABCResult(t_s=t_s, a_m=a_m, b_m=b_m, c_m=c_m)


def simulate_midpoint(
    *,
    a0_m: float,
    k1_per_s: float,
    k2_per_s: float,
    t_end_s: float,
    dt_s: float,
) -> BatchABCResult:
    """
    Simulate A -> B -> C using the Midpoint method (also called RK2).

    Midpoint is smarter than Euler:
      1) Peek at the slope at the START of the step
      2) Use that to estimate where we'd be halfway through
      3) Compute the slope at that halfway point
      4) Use the halfway slope for the full step

    This extra "peek" makes it noticeably more accurate for the same dt.
    """
    _validate_inputs(a0_m, k1_per_s, k2_per_s, t_end_s, dt_s)
    n_steps, t_s = _make_time_grid(t_end_s, dt_s)

    a_m = np.zeros(n_steps, dtype=float)
    b_m = np.zeros(n_steps, dtype=float)
    c_m = np.zeros(n_steps, dtype=float)

    a_m[0] = a0_m

    for i in range(1, n_steps):
        a = a_m[i - 1]
        b = b_m[i - 1]
        c = c_m[i - 1]

        # (1) Slope at the start
        da1, db1, dc1 = _rhs(a, b, k1_per_s, k2_per_s)

        # (2) Estimate state at the halfway point
        a_mid = a + da1 * (dt_s / 2.0)
        b_mid = b + db1 * (dt_s / 2.0)
        c_mid = c + dc1 * (dt_s / 2.0)

        # (3) Slope at the halfway point
        da2, db2, dc2 = _rhs(a_mid, b_mid, k1_per_s, k2_per_s)

        # (4) Full step using the halfway slope
        a_m[i] = max(a + da2 * dt_s, 0.0)
        b_m[i] = max(b + db2 * dt_s, 0.0)
        c_m[i] = max(c + dc2 * dt_s, 0.0)

    return BatchABCResult(t_s=t_s, a_m=a_m, b_m=b_m, c_m=c_m)


# =============================================================================
# METRICS
# Engineering numbers you'd care about in a real reactor.
# =============================================================================

def time_to_conversion_a(
    *, t_s: np.ndarray, a_m: np.ndarray, a0_m: float, target_conversion: float
) -> float:
    """Return the first time at which conversion of A hits target_conversion."""
    if not (0.0 < target_conversion < 1.0):
        raise ValueError("target_conversion must be between 0 and 1 (exclusive)")
    x_a = (a0_m - a_m) / a0_m
    idx = int(np.argmax(x_a >= target_conversion))
    if x_a[idx] < target_conversion:
        raise ValueError("Target conversion not reached within the simulation window.")
    return float(t_s[idx])


def peak_b(*, t_s: np.ndarray, b_m: np.ndarray) -> tuple[float, float]:
    """Return (peak concentration of B, time at which it occurs)."""
    idx = int(np.argmax(b_m))
    return float(b_m[idx]), float(t_s[idx])


# =============================================================================
# MAIN — run the simulation and produce outputs
# =============================================================================

def main() -> None:

    # =========================================================================
    # CONFIG — change anything in this block
    # =========================================================================
    a0_m:      float = 1.0      # initial concentration of A (M)
    k1_per_s:  float = 0.015    # rate constant: A -> B  (1/s)
    k2_per_s:  float = 0.006    # rate constant: B -> C  (1/s)
    t_end_s:   float = 600.0    # total simulation time (s)
    dt_s:      float = 5.0      # timestep size — larger = faster but less accurate (s)

    # Threshold: print when [B] first crosses this concentration
    b_threshold_m: float = 0.2  # (M)
    # =========================================================================

    # --- Run both simulations on the same inputs ---
    euler   = simulate_euler(
        a0_m=a0_m, k1_per_s=k1_per_s, k2_per_s=k2_per_s, t_end_s=t_end_s, dt_s=dt_s
    )
    midpoint = simulate_midpoint(
        a0_m=a0_m, k1_per_s=k1_per_s, k2_per_s=k2_per_s, t_end_s=t_end_s, dt_s=dt_s
    )

    # --- Print summary using the more accurate midpoint result ---
    a_final = float(midpoint.a_m[-1])
    b_final = float(midpoint.b_m[-1])
    c_final = float(midpoint.c_m[-1])
    x_final = (a0_m - a_final) / a0_m

    t90  = time_to_conversion_a(t_s=midpoint.t_s, a_m=midpoint.a_m, a0_m=a0_m, target_conversion=0.90)
    b_max, t_bmax = peak_b(t_s=midpoint.t_s, b_m=midpoint.b_m)

    print("\n--- Batch Reactor Summary: A → B → C (Midpoint method) ---")
    print(f"k1 = {k1_per_s:.4f} 1/s,  k2 = {k2_per_s:.4f} 1/s,  dt = {dt_s} s")
    print(f"Final conversion of A  : {x_final:.3%}")
    print(f"Time to 90% conversion : {t90:.2f} s")
    print(f"Peak [B]               : {b_max:.4f} M  at  t = {t_bmax:.2f} s")
    print(f"Final concentrations   : A = {a_final:.4f} M,  B = {b_final:.4f} M,  C = {c_final:.4f} M")
    print(f"Mass balance (A+B+C)   : {a_final + b_final + c_final:.4f} M  (should be ~{a0_m:.4f} M)")

    mask = midpoint.b_m >= b_threshold_m
    if np.any(mask):
        t_hit = float(midpoint.t_s[int(np.argmax(mask))])
        print(f"[B] first exceeds {b_threshold_m:.2f} M  at  t = {t_hit:.2f} s")
    else:
        print(f"[B] never exceeds {b_threshold_m:.2f} M")

    # --- Error comparison: Euler vs Midpoint ---
    # Max absolute difference between the two methods at any point in time.
    # A large number means Euler is drifting from the more accurate Midpoint result.
    # Try increasing dt to make the gap wider, or decreasing dt to make them agree.
    err_a = float(np.max(np.abs(euler.a_m - midpoint.a_m)))
    err_b = float(np.max(np.abs(euler.b_m - midpoint.b_m)))
    err_c = float(np.max(np.abs(euler.c_m - midpoint.c_m)))

    print("\n--- Euler vs Midpoint: Max Absolute Difference ---")
    print(f"  Species A : {err_a:.6f} M")
    print(f"  Species B : {err_b:.6f} M")
    print(f"  Species C : {err_c:.6f} M")
    print(f"  (dt = {dt_s} s — reduce dt to shrink these errors)")

    # -------------------------------------------------------------------------
    # PLOT — all species on one axes, Euler (dashed) vs Midpoint (solid)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    species = [
        # (euler array,   midpoint array,  species label, line color)
        (euler.a_m, midpoint.a_m, "A", "steelblue"),
        (euler.b_m, midpoint.b_m, "B", "darkorange"),
        (euler.c_m, midpoint.c_m, "C", "seagreen"),
    ]

    for e_arr, m_arr, label, color in species:
        ax.plot(euler.t_s,    e_arr, linestyle="--", color=color, alpha=0.7, label=f"[{label}] Euler")
        ax.plot(midpoint.t_s, m_arr, linestyle="-",  color=color,            label=f"[{label}] Midpoint")

    # Mark peak B (Midpoint)
    peak_idx = int(np.argmax(midpoint.b_m))
    t_peak   = float(midpoint.t_s[peak_idx])
    b_peak   = float(midpoint.b_m[peak_idx])

    ax.scatter([t_peak], [b_peak], color="red", zorder=5)
    ax.annotate(
        f"Peak B\n{b_peak:.3f} M at {t_peak:.1f} s",
        xy=(t_peak, b_peak),
        xytext=(t_peak + 40, b_peak),
        arrowprops=dict(arrowstyle="->"),
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration (M)")
    ax.set_title(f"Batch Reactor: A → B → C   (dt = {dt_s} s)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()