"""Signal timing: fixed-time plan vs. demand-responsive (Webster) plan.

Four independent phases (N, S, E, W), as in the notebook prototype.  Delay uses
the HCM signalised-intersection delay equation (uniform + incremental term),
so it behaves sensibly when an approach is over-saturated.

These are planning-level estimates, NOT a validated optimiser: they ignore
turning movements, pedestrian clearance, coordination and detector data.
"""
from __future__ import annotations
import numpy as np

APPROACHES = ("North", "South", "East", "West")
SAT_FLOW = 1800.0      # veh / hour of green / lane
LOST_PER_PHASE = 4.0   # s (start-up + clearance)
MIN_GREEN = 7.0
C_MIN, C_MAX = 40.0, 150.0


def flows_for(inter, sensor_volume: float) -> np.ndarray:
    return inter.scale * float(sensor_volume) * np.asarray(inter.shares, dtype=float)


def flow_ratios(q, lanes) -> np.ndarray:
    return np.asarray(q, float) / (SAT_FLOW * np.asarray(lanes, float))


def _apply_min_green(g: np.ndarray, total: float) -> np.ndarray:
    g = np.maximum(np.asarray(g, float), MIN_GREEN)
    excess = g.sum() - total
    if excess > 1e-9:
        free = g > MIN_GREEN + 1e-9
        room = (g[free] - MIN_GREEN).sum()
        if room > 0:
            g[free] -= (g[free] - MIN_GREEN) * min(1.0, excess / room)
    return g


def fixed_plan(cycle: float = 90.0) -> dict:
    n = len(APPROACHES)
    total = cycle - LOST_PER_PHASE * n
    return {"cycle": float(cycle), "green": np.full(n, total / n), "method": "Fixed (equal split)"}


def webster_plan(q, lanes) -> dict:
    y = flow_ratios(q, lanes)
    Y = float(y.sum())
    L = LOST_PER_PHASE * len(y)
    cycle = C_MAX if Y >= 0.95 else float(np.clip((1.5 * L + 5.0) / (1.0 - Y), C_MIN, C_MAX))
    total = cycle - L
    g = total * (y / Y) if Y > 0 else np.full(len(y), total / len(y))
    return {"cycle": cycle, "green": _apply_min_green(g, total), "method": "Webster (adaptive)", "Y": Y}


def proportional_plan(q, lanes, cycle: float = 100.0) -> dict:
    y = flow_ratios(q, lanes)
    Y = float(y.sum())
    total = cycle - LOST_PER_PHASE * len(y)
    g = total * (y / Y) if Y > 0 else np.full(len(y), total / len(y))
    return {"cycle": float(cycle), "green": _apply_min_green(g, total), "method": "Proportional (fixed cycle)", "Y": Y}


METHODS = ("Best of both", "Webster (adaptive cycle)", "Re-split only (keep current cycle)")


def optimise(q, lanes, method: str = METHODS[0], current_cycle: float = 90.0) -> dict:
    """Return the recommended plan.  'Best of both' never recommends a plan with
    more modelled delay than either candidate."""
    web = webster_plan(q, lanes)
    split = proportional_plan(q, lanes, cycle=current_cycle)
    if method.startswith("Webster"):
        return web
    if method.startswith("Re-split"):
        return split
    return min((web, split), key=lambda p: evaluate(q, lanes, p)["avg_delay"])


def approach_delay(q, lanes, green, cycle) -> np.ndarray:
    """Average control delay per vehicle (s) for each approach (HCM form)."""
    q = np.asarray(q, float)
    lam = np.asarray(green, float) / cycle
    cap = SAT_FLOW * np.asarray(lanes, float) * lam           # veh/h
    x = np.divide(q, cap, out=np.zeros_like(q), where=cap > 0)
    d1 = 0.5 * cycle * (1 - lam) ** 2 / (1 - np.minimum(1.0, x) * lam)
    T, k, I = 0.25, 0.5, 1.0
    d2 = 900 * T * ((x - 1) + np.sqrt((x - 1) ** 2 + 8 * k * I * x / (np.maximum(cap, 1e-9) * T)))
    return np.where(q > 0, d1 + d2, 0.0)


def evaluate(q, lanes, plan: dict) -> dict:
    d = approach_delay(q, lanes, plan["green"], plan["cycle"])
    q = np.asarray(q, float)
    tot = float(q.sum())
    avg = float((d * q).sum() / tot) if tot > 0 else 0.0
    cap = SAT_FLOW * np.asarray(lanes, float) * plan["green"] / plan["cycle"]
    x = np.divide(q, cap, out=np.zeros_like(q), where=cap > 0)
    return {"delay_per_approach": d, "avg_delay": avg, "veh_hours": float((d * q).sum() / 3600.0),
            "x": x, "max_x": float(x.max()) if len(x) else 0.0}


def status(Y: float) -> str:
    return "Over capacity" if Y >= 0.95 else "Near capacity" if Y >= 0.80 else "Stable"
