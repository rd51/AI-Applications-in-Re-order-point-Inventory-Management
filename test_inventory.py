"""
test_inventory.py — Property-based sanity checks on the inventory maths.

These tests encode what MUST be true for the formulas to be trustworthy, so that
the lecturer (or a future me) can see the dashboard isn't just plotting numbers.
Run with:  pytest test_inventory.py -v
"""

import numpy as np
import pytest

import inventory as inv


# --------------------------------------------------------------------------- #
# Service-level <-> Z
# --------------------------------------------------------------------------- #
def test_z_at_50pct_is_zero():
    assert inv.z_from_service_level(0.50) == pytest.approx(0.0, abs=1e-9)


def test_z_known_values():
    assert inv.z_from_service_level(0.95) == pytest.approx(1.6448536, abs=1e-5)
    assert inv.z_from_service_level(0.99) == pytest.approx(2.3263479, abs=1e-5)


def test_z_roundtrip():
    for sl in (0.80, 0.90, 0.95, 0.975, 0.99):
        assert inv.service_level_from_z(inv.z_from_service_level(sl)) == pytest.approx(sl)


def test_service_level_bounds_rejected():
    with pytest.raises(ValueError):
        inv.z_from_service_level(0.0)
    with pytest.raises(ValueError):
        inv.z_from_service_level(1.0)


# --------------------------------------------------------------------------- #
# Safety stock
# --------------------------------------------------------------------------- #
def test_ss_zero_at_50pct_service():
    # Z = 0  =>  no safety stock, regardless of variability
    assert inv.safety_stock(0.50, sigma_d=10, lead_time_mean=5, demand_mean=100) == pytest.approx(0.0)


def test_ss_scales_sqrt2_when_sigma_d_doubles_no_leadtime_variance():
    # With sigma_L = 0, SS = Z * sigma_D * sqrt(L), so doubling sigma_D doubles SS.
    base = inv.safety_stock(0.95, sigma_d=10, lead_time_mean=4, demand_mean=100, sigma_l=0)
    doubled = inv.safety_stock(0.95, sigma_d=20, lead_time_mean=4, demand_mean=100, sigma_l=0)
    assert doubled == pytest.approx(2 * base)


def test_ss_matches_hand_calculation():
    # Z(95%)=1.6449, sigma_D=10, L=4, sigma_L=0 -> SS = 1.6449 * 10 * sqrt(4) = 32.897
    ss = inv.safety_stock(0.95, sigma_d=10, lead_time_mean=4, demand_mean=100, sigma_l=0)
    assert ss == pytest.approx(1.6448536 * 10 * 2, abs=1e-3)


def test_leadtime_variance_increases_ss():
    # Adding lead-time uncertainty can only increase the buffer.
    no_lt_var = inv.safety_stock(0.95, sigma_d=10, lead_time_mean=5, demand_mean=100, sigma_l=0)
    with_lt_var = inv.safety_stock(0.95, sigma_d=10, lead_time_mean=5, demand_mean=100, sigma_l=2)
    assert with_lt_var > no_lt_var


def test_ss_monotonic_in_service_level():
    levels = [0.80, 0.90, 0.95, 0.99]
    ss = [inv.safety_stock(sl, sigma_d=10, lead_time_mean=5, demand_mean=100, sigma_l=1) for sl in levels]
    assert all(b > a for a, b in zip(ss, ss[1:]))


# --------------------------------------------------------------------------- #
# Reorder point
# --------------------------------------------------------------------------- #
def test_rop_decomposition():
    ss = inv.safety_stock(0.95, sigma_d=10, lead_time_mean=5, demand_mean=100)
    rop = inv.reorder_point(demand_mean=100, lead_time_mean=5, ss=ss)
    assert rop == pytest.approx(100 * 5 + ss)


# --------------------------------------------------------------------------- #
# Cost curve
# --------------------------------------------------------------------------- #
def test_total_cost_components_add_up():
    r = inv.total_cost(
        0.95, sigma_d=10, lead_time_mean=5, demand_mean=100,
        holding_cost_per_unit_per_year=2.0, stockout_cost_per_unit=20.0,
    )
    assert r.total_cost == pytest.approx(r.holding_cost + r.stockout_cost)


def test_holding_cost_rises_with_service_level():
    def h(sl):
        return inv.total_cost(
            sl, sigma_d=10, lead_time_mean=5, demand_mean=100,
            holding_cost_per_unit_per_year=2.0, stockout_cost_per_unit=20.0,
        ).holding_cost
    assert h(0.99) > h(0.90) > h(0.80)


def test_stockout_cost_falls_with_service_level():
    def s(sl):
        return inv.total_cost(
            sl, sigma_d=10, lead_time_mean=5, demand_mean=100,
            holding_cost_per_unit_per_year=2.0, stockout_cost_per_unit=20.0,
        ).stockout_cost
    assert s(0.99) < s(0.90) < s(0.80)


def test_total_cost_has_interior_minimum():
    # Sweep service levels; the cheapest point should not be at either extreme,
    # i.e. there is a genuine trade-off the optimiser can find.
    levels = np.linspace(0.50, 0.999, 200)
    costs = [
        inv.total_cost(
            sl, sigma_d=15, lead_time_mean=6, demand_mean=120,
            holding_cost_per_unit_per_year=1.5, stockout_cost_per_unit=25.0,
        ).total_cost
        for sl in levels
    ]
    best = int(np.argmin(costs))
    assert 0 < best < len(levels) - 1


# --------------------------------------------------------------------------- #
# Monte Carlo cross-check  (the formula validates itself)
# --------------------------------------------------------------------------- #
def test_monte_carlo_matches_target_service_level_no_leadtime_var():
    target = 0.95
    ss = inv.safety_stock(target, sigma_d=10, lead_time_mean=5, demand_mean=100, sigma_l=0)
    rop = inv.reorder_point(100, 5, ss)
    out = inv.monte_carlo_service_level(
        rop, sigma_d=10, lead_time_mean=5, demand_mean=100, sigma_l=0,
        n_trials=60_000, seed=7,
    )
    # Achieved service level should land within ~1.5% of target.
    assert out["achieved_service_level"] == pytest.approx(target, abs=0.015)


def test_monte_carlo_is_reproducible():
    a = inv.monte_carlo_service_level(600, 10, 5, 100, n_trials=5000, seed=1)
    b = inv.monte_carlo_service_level(600, 10, 5, 100, n_trials=5000, seed=1)
    assert a["achieved_service_level"] == b["achieved_service_level"]


def test_higher_rop_gives_higher_service_level():
    low = inv.monte_carlo_service_level(520, 10, 5, 100, n_trials=20000, seed=3)
    high = inv.monte_carlo_service_level(560, 10, 5, 100, n_trials=20000, seed=3)
    assert high["achieved_service_level"] > low["achieved_service_level"]
