"""
inventory.py — Core inventory-control mathematics for the AI-in-Operations dashboard.

Everything here is a pure function: no data files, no global state, no randomness
except where a seed is passed in explicitly. This is the layer that the Streamlit
app and the tests both stand on, so it is deliberately boring and verifiable.

Key formulas (from the lecturer's slide, Session II):
    Safety Stock  SS  = Z * sqrt( sigma_D^2 * L  +  D_bar^2 * sigma_L^2 )
    Reorder Point ROP = D_bar * L  +  SS

Where:
    Z        = service-level factor  (standard-normal quantile of the service level)
    sigma_D  = standard deviation of demand per period
    L        = mean lead time (in periods)
    D_bar    = mean demand per period
    sigma_L  = standard deviation of lead time (in periods)

The second term under the root (D_bar^2 * sigma_L^2) is what captures *lead-time
uncertainty* — it vanishes when the supplier is perfectly reliable (sigma_L = 0),
which is exactly the lever the dashboard lets a user toggle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Service level  <->  Z factor
# --------------------------------------------------------------------------- #
def z_from_service_level(service_level: float) -> float:
    """Standard-normal quantile for a cycle service level.

    service_level is a probability in (0, 1). A 50% service level returns 0.0
    (no safety stock), 95% returns ~1.645, 99% returns ~2.326.
    """
    if not 0.0 < service_level < 1.0:
        raise ValueError("service_level must be strictly between 0 and 1")
    return float(stats.norm.ppf(service_level))


def service_level_from_z(z: float) -> float:
    """Inverse of z_from_service_level — the service level implied by a Z factor."""
    return float(stats.norm.cdf(z))


# --------------------------------------------------------------------------- #
# Combined standard deviation of demand over the lead time
# --------------------------------------------------------------------------- #
def sigma_demand_over_leadtime(
    sigma_d: float,
    lead_time_mean: float,
    demand_mean: float,
    sigma_l: float = 0.0,
) -> float:
    """sqrt( sigma_D^2 * L  +  D_bar^2 * sigma_L^2 ).

    This is the spread of total demand accumulated over one (uncertain) lead time,
    assuming per-period demand is independent across days. It is the quantity that
    both the safety-stock formula and the expected-shortage calculation need.
    """
    if sigma_d < 0 or sigma_l < 0:
        raise ValueError("standard deviations cannot be negative")
    if lead_time_mean < 0:
        raise ValueError("lead time cannot be negative")
    var = (sigma_d ** 2) * lead_time_mean + (demand_mean ** 2) * (sigma_l ** 2)
    return float(np.sqrt(var))


# --------------------------------------------------------------------------- #
# Safety stock and reorder point
# --------------------------------------------------------------------------- #
def safety_stock(
    service_level: float,
    sigma_d: float,
    lead_time_mean: float,
    demand_mean: float,
    sigma_l: float = 0.0,
) -> float:
    """SS = Z * sqrt( sigma_D^2 * L + D_bar^2 * sigma_L^2 ).

    Note sigma_D here should come from *forecast error* (the residuals of the
    demand model on a holdout window), not from the raw demand series — that is
    the honest link between the forecasting tab and this one.
    """
    z = z_from_service_level(service_level)
    sigma_lt = sigma_demand_over_leadtime(sigma_d, lead_time_mean, demand_mean, sigma_l)
    return float(z * sigma_lt)


def reorder_point(
    demand_mean: float,
    lead_time_mean: float,
    ss: float,
) -> float:
    """ROP = D_bar * L + SS. The level at which a replenishment order is triggered."""
    return float(demand_mean * lead_time_mean + ss)


# --------------------------------------------------------------------------- #
# Expected shortage per cycle  (closed form, Normal loss function)
# --------------------------------------------------------------------------- #
def expected_shortage_per_cycle(
    service_level: float,
    sigma_d: float,
    lead_time_mean: float,
    demand_mean: float,
    sigma_l: float = 0.0,
) -> float:
    """Expected units short in one replenishment cycle, E[(D_L - ROP)^+].

    Uses the standard-normal loss function  L(z) = phi(z) - z*(1 - Phi(z)),
    scaled by the lead-time demand spread. At service level 50% (z=0) this equals
    0.3989 * sigma_lt; as service level rises it shrinks toward zero.
    """
    z = z_from_service_level(service_level)
    sigma_lt = sigma_demand_over_leadtime(sigma_d, lead_time_mean, demand_mean, sigma_l)
    unit_loss = stats.norm.pdf(z) - z * (1.0 - stats.norm.cdf(z))
    return float(sigma_lt * unit_loss)


# --------------------------------------------------------------------------- #
# Total relevant cost as a function of service level  (the excess-cost curve)
# --------------------------------------------------------------------------- #
@dataclass
class CostResult:
    service_level: float
    z: float
    safety_stock: float
    reorder_point: float
    holding_cost: float
    stockout_cost: float
    total_cost: float


def total_cost(
    service_level: float,
    sigma_d: float,
    lead_time_mean: float,
    demand_mean: float,
    holding_cost_per_unit_per_year: float,
    stockout_cost_per_unit: float,
    sigma_l: float = 0.0,
    orders_per_year: float = 12.0,
) -> CostResult:
    """Annual cost of holding safety stock vs. the cost of expected stockouts.

    - holding cost  = h * SS            (cost of carrying the buffer for a year)
    - stockout cost = p * E[short] * N  (per-cycle expected shortage, N cycles/year)

    The cheapest service level is the trough of (holding + stockout). Raising the
    service level buys fewer stockouts at the price of more holding — this function
    is exactly what the optimiser minimises and what the cost tab plots.
    """
    z = z_from_service_level(service_level)
    ss = safety_stock(service_level, sigma_d, lead_time_mean, demand_mean, sigma_l)
    rop = reorder_point(demand_mean, lead_time_mean, ss)
    holding = holding_cost_per_unit_per_year * ss
    shortage_units = expected_shortage_per_cycle(
        service_level, sigma_d, lead_time_mean, demand_mean, sigma_l
    )
    stockout = stockout_cost_per_unit * shortage_units * orders_per_year
    return CostResult(
        service_level=service_level,
        z=z,
        safety_stock=ss,
        reorder_point=rop,
        holding_cost=holding,
        stockout_cost=stockout,
        total_cost=holding + stockout,
    )


# --------------------------------------------------------------------------- #
# Monte Carlo validation — does the formula actually deliver the service level?
# --------------------------------------------------------------------------- #
def monte_carlo_service_level(
    reorder_point_value: float,
    sigma_d: float,
    lead_time_mean: float,
    demand_mean: float,
    sigma_l: float = 0.0,
    n_trials: int = 10_000,
    seed: int = 42,
    demand_sampler=None,
) -> dict:
    """Simulate many replenishment cycles and measure the achieved service level.

    For each trial: draw a lead time (Normal(L, sigma_L), rounded, floored at 1),
    then draw that many days of demand and sum them. A stockout happens when that
    lead-time demand exceeds the reorder point. The achieved cycle service level is
    the fraction of trials with NO stockout; it should match the target the ROP was
    built for, within sampling noise. This is the independent check on the maths.

    demand_sampler: optional callable (rng, n_days) -> ndarray of daily demand,
    so the empirical / forecast-residual distribution can replace the Normal later.
    """
    rng = np.random.default_rng(seed)

    if sigma_l > 0:
        lead_times = rng.normal(lead_time_mean, sigma_l, size=n_trials)
        lead_times = np.maximum(1, np.rint(lead_times)).astype(int)
    else:
        lead_times = np.full(n_trials, int(round(lead_time_mean)), dtype=int)

    lead_time_demand = np.empty(n_trials)
    for i, L in enumerate(lead_times):
        if demand_sampler is not None:
            daily = demand_sampler(rng, L)
        else:
            daily = rng.normal(demand_mean, sigma_d, size=L)
        lead_time_demand[i] = daily.sum()

    stockouts = lead_time_demand > reorder_point_value
    achieved = 1.0 - stockouts.mean()
    shortage = np.maximum(0.0, lead_time_demand - reorder_point_value)
    return {
        "achieved_service_level": float(achieved),
        "stockout_rate": float(stockouts.mean()),
        "mean_shortage_units": float(shortage.mean()),
        "n_trials": n_trials,
    }
