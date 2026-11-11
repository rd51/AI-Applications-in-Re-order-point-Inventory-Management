"""
export_data.py — Precompute everything the browser dashboard needs, into one JSON.

This runs the REAL Python pipeline (forecast.run_forecast with LightGBM) once per SKU
per model, then dumps the honest results — held-out forecasts, residuals, sigma_D,
demand history, weekly demand — so the self-contained HTML dashboard can do all the
cheap inventory maths (safety stock, ROP, cost curve, Monte Carlo, bullwhip) live in
the browser with NO server and NO Python. The heavy/one-off work (training the trees,
scoring the holdout) is baked in here; nothing is fabricated.

Run:  python export_data.py      ->  dashboard_data.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import forecast as fc

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "m5_filtered.csv")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_data.json")

MODELS = ["naive", "seasonal_naive", "lightgbm"]
RECENT_CHART_DAYS = 120


def iso(dates) -> list[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]


def build_sku(df_sku: pd.DataFrame) -> dict:
    sku = str(df_sku["sku"].iloc[0])
    sku_df = df_sku.sort_values("date").reset_index(drop=True)

    per_model = {}
    all_metrics = {}
    for m in MODELS:
        try:
            r = fc.run_forecast(sku_df, model=m)
        except Exception as e:  # e.g. lightgbm missing
            print(f"  [skip] {sku}/{m}: {e}")
            continue
        all_metrics[m] = r.all_metrics.get(m)
        per_model[m] = {
            "sigma_d": r.sigma_d,
            "metrics": r.all_metrics.get(m),
            "holdout": {
                "date": iso(r.holdout["date"]),
                "y_true": [float(v) for v in r.holdout["y_true"]],
                "y_pred": [round(float(v), 3) for v in r.holdout["y_pred"]],
            },
            "residuals": [round(float(v), 4) for v in r.residuals],
        }

    # model-independent facts (demand_mean, zero %, intermittent) from any run
    ref = fc.run_forecast(sku_df, model="naive")
    best_model = min(all_metrics, key=lambda k: all_metrics[k]["WAPE"])

    # recent daily demand for the overview chart
    recent = sku_df.tail(RECENT_CHART_DAYS)

    # weekly demand (positive weeks) for the bullwhip simulation
    weekly = (sku_df.set_index("date")["units"].resample("W").sum())
    weekly = weekly[weekly > 0].to_numpy()

    return {
        "sku": sku,
        "demand_mean": ref.demand_mean,
        "zero_day_pct": ref.zero_day_pct,
        "intermittent": bool(ref.intermittent),
        "best_model": best_model,
        "all_metrics": all_metrics,
        "models": per_model,
        "recent": {
            "date": iso(recent["date"]),
            "units": [float(v) for v in recent["units"]],
        },
        # full daily demand history -> empirical bootstrap sampler for Monte Carlo
        "daily_units": [float(v) for v in sku_df["units"].to_numpy()],
        "weekly_demand": [round(float(v), 2) for v in weekly],
        "n_days": int(len(sku_df)),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    skus = list(df["sku"].unique())
    out = {
        "meta": {
            "store": "Walmart M5 - Store CA_3",
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "horizon_days": fc.HORIZON,
            "seed": fc.SEED,
            "n_rows": int(len(df)),
        },
        "skus": [],
    }
    for sku in skus:
        print(f"Processing {sku} ...")
        out["skus"].append(build_sku(df[df["sku"] == sku]))

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\nWrote {OUT_PATH}  ({size_kb:,.0f} KB)")
    for s in out["skus"]:
        wapes = " | ".join(f"{m}:{v['WAPE']}" for m, v in s["all_metrics"].items())
        print(f"  {s['sku']:18s} best={s['best_model']:14s} Dbar={s['demand_mean']:.1f}  [{wapes}]")


if __name__ == "__main__":
    main()
