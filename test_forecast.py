"""
test_forecast.py — Smoke + no-leakage checks for the forecasting module.

The headline test is NO FUTURE LEAKAGE: every engineered feature at row t must be
computable from data up to t-1. If a lag or rolling feature accidentally included
the current day, the holdout score would be dishonestly good — the exact mistake
in the lecturer's reference notebook (it scored the model on its training data).
"""

import numpy as np
import pandas as pd
import pytest

import forecast as fc


def _toy_series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    dow = dates.dayofweek.to_numpy()
    units = 50 + 15 * (dow >= 5) + rng.normal(0, 5, n)  # weekend bump + noise
    return pd.DataFrame({
        "date": dates, "sku": "TOY", "units": np.clip(units, 0, None).round(),
        "sell_price": 1.0, "snap_CA": 0, "has_event": 0,
    })


def test_make_features_no_leakage():
    df = _toy_series()
    d, feats = fc.make_features(df)
    # lag_1 at row t must equal units at row t-1 (pure past)
    assert d["lag_1"].iloc[5] == pytest.approx(d["units"].iloc[4])
    # rolling mean uses shift(1), so it must not include the current row
    manual = d["units"].iloc[1:8].mean()
    assert d["roll_mean_7"].iloc[8] == pytest.approx(manual)


def test_wape_handles_zeros():
    assert fc.wape(np.array([0, 0, 0]), np.array([1, 1, 1])) != fc.wape(  # denom 0 -> nan
        np.array([0, 0, 0]), np.array([1, 1, 1])
    ) or np.isnan(fc.wape(np.array([0, 0, 0]), np.array([1, 1, 1])))
    assert fc.wape(np.array([10, 10]), np.array([10, 10])) == pytest.approx(0.0)


def test_run_forecast_shapes_and_sigma():
    df = _toy_series()
    r = fc.run_forecast(df, model="best")
    assert len(r.holdout) == len(r.residuals)
    assert r.sigma_d >= 0
    assert r.demand_mean > 0
    assert set(["naive", "seasonal_naive"]).issubset(r.all_metrics.keys())


def test_lightgbm_beats_naive_on_seasonal_data():
    # On a clean weekly-seasonal series, the ML model should not be worse than naive.
    df = _toy_series(n=500, seed=3)
    r = fc.run_forecast(df, model="best")
    assert r.metrics["WAPE"] <= r.all_metrics["naive"]["WAPE"] + 1e-6
