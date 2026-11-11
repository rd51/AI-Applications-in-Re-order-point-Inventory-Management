# AI Inventory Intelligence Dashboard

**AI in Operations — Session II lab.** An interactive, **self-contained HTML dashboard**
that turns a classical reorder-point calculator into an AI-assisted decision tool, built
on **real Walmart demand data** (the M5 dataset). It covers demand forecasting, dynamic
safety stock, Monte Carlo validation, cost optimisation, and an emergent bullwhip-effect
simulation.

The dashboard is a single file (`dashboard.html`) with the chart library and data baked
in — no server, no install, no Python. Open it in any browser. The Python files are only
needed to **regenerate** the baked-in forecast data or to run the tests.

---

## What it does (four tabs)

1. **Overview** — KPI cards (forecast error, safety stock, reorder point, projected
   annual cost) and a demand-vs-forecast chart.
2. **Demand Forecasting** — naïve and seasonal-naïve baselines vs a LightGBM model,
   scored **honestly on a held-out 28-day window**. Shows the forecast-error
   distribution and the σ_D that feeds the safety-stock formula. Switch models (or let
   it auto-pick the best) from the sidebar; re-run `export_data.py` to refresh on new data.
3. **Safety Stock & ROP** — the formula live from your sliders, plus a **Monte Carlo
   reality check** that simulates thousands of lead-time windows and measures the
   service level actually achieved.
4. **Optimise & Bullwhip** — the holding-vs-stockout cost curve with its cheapest
   point, a **constrained cost-minimising search** for the cheapest service level above
   a floor you set, a tornado sensitivity chart, and a 4-stage **bullwhip simulation**
   where amplification *emerges* from the ordering rule (not hard-coded).

## Real vs. assumed data (stated honestly in the app)

- **Real** (straight from Walmart's M5 dataset): daily unit sales per SKU, weekly
  sell prices / promotions, the event calendar (Super Bowl, Easter, holidays), and
  California SNAP benefit days.
- **Assumed** (your inputs, i.e. the sliders — *not* synthesised data): supplier lead
  time and its variability, unit holding cost, unit stockout cost. These are the
  numbers a real planner types in from a supplier contract and the finance team.

No demand is ever fabricated. The data-prep script only selects and reshapes columns
that already exist in the real data.

## Honest findings (what the dashboard actually shows)

- Machine learning clearly beats the naïve baseline on the **high-volume, structured**
  SKUs (Product A food ~19% WAPE vs 25%; Product B household ~23% vs 33%).
- Machine learning **loses** on the choppy hobby item and the intermittent slow mover —
  so the app auto-selects the *best* model per SKU rather than assuming ML always wins.
- The safety-stock formula is **mis-calibrated for the intermittent slow mover** (≈97%
  zero-sales days); the Monte Carlo check with real bootstrapped demand reveals the gap.
- The bullwhip effect **compounds** up the chain (the manufacturer can see ~25× the
  variability of real customer demand), and turning on **shared POS data** ("single
  source of truth") and **AI forecasting** visibly damps it.

## How this fixes the reference notebook's flaws

| Reference notebook | This project |
|---|---|
| Forecast scored on its own training data | Honest 28-day holdout; no-leakage unit test |
| No random seeds | All randomness seeded (`SEED = 42`) |
| Safety stock a fixed input (`= 200`) | Calculated from Z, σ_D, lead time & its variance |
| Bullwhip hard-coded (`amplification = 1.2`) | Amplification emerges from an order-up-to rule |
| Toy 12-point synthetic demand | 5+ years of real daily Walmart sales |

## Files

```
Lab Work 1/
├── dashboard.html          # THE DASHBOARD — self-contained, open in any browser
├── dashboard_template.html # page source (readable; Plotly + data are inlined at build)
├── dashboard_data.json     # precomputed forecasts / σ_D / demand, baked into the page
├── plotly.min.js           # chart library, inlined into dashboard.html at build time
├── build_dashboard.py      # inlines plotly + data into the template -> dashboard.html
├── export_data.py          # runs the real pipeline -> dashboard_data.json
├── inventory.py            # safety stock, ROP, cost curve, Monte Carlo (pure functions)
├── forecast.py             # naïve / seasonal-naïve / LightGBM, honest holdout, σ_D
├── bullwhip.py             # emergent 4-stage bullwhip simulation + AI levers
├── data_prep.py            # filters raw M5 -> data/m5_filtered.csv
├── test_inventory.py       # 17 formula tests (incl. Monte Carlo cross-check)
├── test_forecast.py        # forecasting smoke + no-leakage tests
├── requirements.txt        # deps for the regen pipeline + tests only (not the dashboard)
├── data/
│   └── m5_filtered.csv     # the compact, tidy real-demand input the pipeline reads
└── m5-forecasting-accuracy/   # raw Kaggle files (only needed to re-run data_prep)
```

The inventory maths (safety stock, ROP, cost curve, Monte Carlo, bullwhip) are ported
to JavaScript inside the page and run live in your browser; they are verified to match
`inventory.py` to rounding. The forecasts are **not** re-run in the browser — they are
computed once by the real LightGBM/baseline pipeline and baked into `dashboard_data.json`.

## Run it

**Just open the dashboard — nothing to install:**

```bash
# open dashboard.html in your default browser (any OS: just double-click it)
start dashboard.html
```

Optionally serve it over `http://localhost` instead of `file://`:

```bash
python -m http.server 8000
# then open http://localhost:8000/dashboard.html
```

**Regenerate the data / rebuild the page** (only if the underlying data changes):

```bash
pip install -r requirements.txt

# (optional) regenerate the filtered data from the raw Kaggle files:
python data_prep.py

# recompute the forecasts and rebuild the self-contained page:
python export_data.py        # -> dashboard_data.json
python build_dashboard.py    # inlines plotly + data -> dashboard.html

# run the formula tests:
pytest -q
```

The pipeline only needs `data/m5_filtered.csv`, which is already included — you do not
need the raw 430 MB Kaggle files unless you want to re-run `data_prep.py`.

## Deployment

Because `dashboard.html` is a single self-contained file, it deploys anywhere that serves
static files — GitHub Pages, Netlify, Cloudflare Pages, or a Hugging Face Space
(`sdk: static`) — with no build step and no server. You can also just email or share the
file directly; it runs offline.

## Data & references

- Demand data: **M5 Forecasting – Accuracy** (Walmart), Kaggle.
- Single source of truth / supplier data sharing: **Walmart Luminate & Retail Link**.
- Probabilistic demand forecasting: **Amazon Science**.
- Bullwhip effect: **Lee, Padmanabhan & Whang (1997)**.
