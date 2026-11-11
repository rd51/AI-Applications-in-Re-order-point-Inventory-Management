"""
data_prep.py — Turn the raw M5 Walmart competition files into one tidy, compact CSV.

WHAT IS REAL vs ASSUMED
    REAL (straight from Walmart's M5 dataset):
        - daily unit sales per SKU
        - the event calendar (Super Bowl, Easter, religious/national holidays)
        - SNAP benefit days in California (a real demand driver for food)
        - weekly sell prices (promotions show up as price dips)
    ASSUMED (set by the user on the dashboard, NOT invented data):
        - supplier lead time and its variability
        - unit holding cost and unit stockout cost
    Nothing in this file fabricates demand. We only select, reshape and join
    columns that already exist in the real data.

SELECTION
    Store  : CA_3  (highest-volume store in the dataset)
    SKUs   : 3 fast movers across the three categories + 1 intermittent slow mover,
             so the dashboard can show both where the classic formula works and
             where it breaks.

OUTPUT
    data/m5_filtered.csv  — one row per (sku, date) with units, price, calendar
                            features and event/SNAP flags. A few MB, not 430 MB.

Run from inside the "Lab Work 1" folder:
    python data_prep.py
"""

from __future__ import annotations

import os

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths — resolved relative to this script so it runs from anywhere.
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "m5-forecasting-accuracy")
OUT_DIR = os.path.join(HERE, "data")
OUT_CSV = os.path.join(OUT_DIR, "m5_filtered.csv")

STORE = "CA_3"

# item_id -> friendly label shown on the dashboard
SKU_MAP = {
    "FOODS_3_090":     "A_Food_Fast",      # ~131 units/day, high volume
    "HOUSEHOLD_1_118": "B_Household_Clean",# ~23 units/day, very low intermittency
    "HOBBIES_1_234":   "C_Hobby_Variable", # ~16 units/day, choppy
    "HOBBIES_2_097":   "Slow_Intermittent",# ~0.04 units/day, ~98% zero-days
}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. Sales: keep only our store + items, then melt wide days to long ---
    sales = pd.read_csv(os.path.join(RAW, "sales_train_evaluation.csv"))
    mask = (sales.store_id == STORE) & (sales.item_id.isin(SKU_MAP))
    sales = sales[mask].copy()
    if sales.empty:
        raise SystemExit(f"No rows matched store={STORE} and the chosen items.")

    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(
        id_vars=id_vars, value_vars=day_cols,
        var_name="d", value_name="units",
    )

    # --- 2. Calendar: map d_x -> real date + events + SNAP + weekday/month ---
    cal = pd.read_csv(os.path.join(RAW, "calendar.csv"))
    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA",
    ]
    long = long.merge(cal[cal_cols], on="d", how="left")
    long["date"] = pd.to_datetime(long["date"])

    # --- 3. Prices: weekly sell price per (store, item, week). A promo = a dip. ---
    prices = pd.read_csv(os.path.join(RAW, "sell_prices.csv"))
    prices = prices[(prices.store_id == STORE) & (prices.item_id.isin(SKU_MAP))]
    long = long.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left",
    )

    # --- 4. Drop each SKU's pre-launch period (no price = not yet on sale) ---
    # Before an item's first priced week it simply wasn't stocked; those zeros are
    # not real demand, so trimming them avoids teaching the model a fake history.
    long = long.sort_values(["item_id", "date"])
    long = long[long.groupby("item_id")["sell_price"].transform(
        lambda s: s.ffill().notna()
    )]

    # --- 5. Friendly label + has_event flag, tidy columns, save ---
    long["sku"] = long["item_id"].map(SKU_MAP)
    long["has_event"] = long["event_name_1"].notna().astype(int)

    keep = [
        "date", "sku", "item_id", "cat_id", "units", "sell_price",
        "weekday", "wday", "month", "year", "snap_CA",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2", "has_event",
    ]
    out = long[keep].sort_values(["sku", "date"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    # --- 6. Console summary so we can eyeball the result ---
    print(f"Saved {OUT_CSV}")
    print(f"Rows: {len(out):,}  |  Date range: {out.date.min().date()} -> {out.date.max().date()}")
    print("\nPer-SKU summary:")
    g = out.groupby("sku").agg(
        rows=("units", "size"),
        avg_daily_units=("units", "mean"),
        zero_day_pct=("units", lambda s: (s == 0).mean() * 100),
        avg_price=("sell_price", "mean"),
    ).round(2)
    print(g.to_string())


if __name__ == "__main__":
    main()
