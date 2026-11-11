"""
bullwhip.py — A 4-stage supply chain where the bullwhip effect EMERGES.

The lecturer's reference notebook hard-codes amplification (`amplification = 1.2`),
which defeats the purpose: it asserts the bullwhip instead of demonstrating it. Here
amplification is a *consequence* of a standard order-up-to policy with moving-average
forecasting — nobody sets it; it falls out of the dynamics.

Order equation at each tier (the canonical demand-signal-processing rule,
Chen, Drezner, Ryan & Simchi-Levi 2000):

    forecast        D_hat_t = moving average of the demand SIGNAL the tier observes
    order-up-to     S_t     = L * D_hat_t + z * sigma_hat_t
    order placed    O_t     = D_t + (S_t - S_{t-1})        (floored at 0)

The term (S_t - S_{t-1}) is the culprit: a tier chasing its own updating forecast
over-orders on the way up and under-orders on the way down, so the order stream is
MORE variable than the demand it saw. Pass that amplified stream upstream and the
variance compounds tier by tier.

Two AI levers (the exercise's "redesign with AI"):
    shared_info     — every tier forecasts off REAL customer demand ("single source
                      of truth", i.e. Walmart Retail Link). Forecasts stop chasing
                      already-distorted orders, so amplification stops compounding.
    ai_forecasting  — a smoother, less twitchy forecast (longer effective memory,
                      less noise) so (S_t - S_{t-1}) shrinks at every tier.
"""

from __future__ import annotations

import numpy as np

TIER_NAMES = ["Retailer", "Distributor", "Wholesaler", "Manufacturer"]


def _forecast(signal: np.ndarray, window: int, ai: bool):
    """Moving-average forecast of a demand signal, returning per-period mean & std.
    AI mode uses a longer, calmer memory and trusts the signal more (less noise)."""
    T = len(signal)
    w = window * 2 if ai else window
    mu = np.zeros(T)
    sigma = np.zeros(T)
    for t in range(T):
        lo = max(0, t - w)
        hist = signal[lo:t] if t > 0 else signal[:1]
        mu[t] = hist.mean()
        sigma[t] = hist.std() if len(hist) > 1 else 0.0
    if ai:
        sigma = sigma * 0.5
    return mu, sigma


def simulate(
    customer_demand,
    n_tiers: int = 4,
    lead_time: int = 2,
    ma_window: int = 4,
    z: float = 1.0,
    shared_info: bool = False,
    ai_forecasting: bool = False,
) -> dict:
    """Run the chain and return each tier's order stream plus its amplification ratio
    (variance of that tier's orders divided by variance of true customer demand)."""
    customer = np.asarray(customer_demand, dtype=float)
    T = len(customer)
    cust_var = customer.var()

    streams = [customer]          # tier 0's observed demand IS customer demand
    amplification = [1.0]         # customer vs itself
    labels = ["Customer"]

    observed = customer           # demand signal entering the current tier
    for i in range(n_tiers):
        forecast_basis = customer if shared_info else observed
        mu, sigma = _forecast(forecast_basis, ma_window, ai_forecasting)

        S = lead_time * mu + z * sigma
        dS = np.diff(S, prepend=S[0])
        orders = np.clip(observed + dS, 0, None)

        streams.append(orders)
        amplification.append(orders.var() / cust_var if cust_var > 0 else np.nan)
        labels.append(TIER_NAMES[i] if i < len(TIER_NAMES) else f"Tier {i+1}")

        observed = orders          # the next tier upstream sees these orders

    return {
        "labels": labels,            # ["Customer", "Retailer", ... "Manufacturer"]
        "streams": streams,          # list of per-period arrays, same order as labels
        "amplification": amplification,
        "customer_var": float(cust_var),
    }


def allocate_stock(total_stock: float, retailer_demands) -> np.ndarray:
    """Simple optimised allocation of a shared pool across retailers, in proportion
    to each retailer's recent mean demand (fill-rate-fair). Used by the dashboard's
    'optimised allocation' toggle in the bullwhip view."""
    d = np.asarray(retailer_demands, dtype=float)
    if d.sum() <= 0:
        return np.full(len(d), total_stock / len(d))
    return total_stock * d / d.sum()


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    demand = rng.poisson(20, 100).astype(float)
    print("Variance amplification by tier (ratio vs customer demand):\n")
    print(f"{'scenario':38s} " + "  ".join(f"{n[:5]:>6s}" for n in TIER_NAMES))
    for name, kw in [
        ("1. No AI (reactive, siloed)", dict()),
        ("2. + AI forecasting", dict(ai_forecasting=True)),
        ("3. + shared POS (single src of truth)", dict(shared_info=True)),
        ("4. + both", dict(shared_info=True, ai_forecasting=True)),
    ]:
        out = simulate(demand, **kw)
        amps = out["amplification"][1:]  # skip customer
        print(f"{name:38s} " + "  ".join(f"{a:6.2f}" for a in amps))
