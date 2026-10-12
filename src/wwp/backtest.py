"""Replay unseen harvests (2025, 2026) day by day under different rostering policies.

Every policy faces the same real weather and the same random draws (common random
numbers), so cost differences come from the decisions alone. Unpicked bins carry over
to the next day, so a bad call keeps costing money.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C
from .config import FarmParams
from .decision import band_probs, best_crew
from .workforce import (ImpactModel, day_outcome, is_harvest_day, new_ready_bins,
                        productive_hours, season_rate_factor)

POLICIES = [
    "Full crew (status quo)",
    "Crew sized to ripe fruit (no weather)",
    "Forecast rule of thumb",
    "AI planner: history-only model",
    "AI planner: MOS model",
    "Perfect forecast, same planner (upper bound)",
]


def _sized_to_fruit(ready, rate, farm):
    return min(farm.crew_pool, math.ceil(ready / (rate * farm.shift_hours)))


def run(obs: pd.DataFrame, test: pd.DataFrame, pred_mos: pd.DataFrame,
        pred_hist: pd.DataFrame, impact: ImpactModel, farm: FarmParams = C.FARM,
        seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_day = {d: g for d, g in obs.groupby(obs.index.normalize())}
    days = [d for d in test.index if is_harvest_day(d)]
    rng = np.random.default_rng(seed)
    actual = {d: (productive_hours(by_day[d], farm, rng),
                  farm.pick_rate * season_rate_factor(d) * rng.lognormal(0, 0.08))
              for d in days}

    rows = []
    for policy in POLICIES:
        backlog, season = 0.0, None
        for d in days:
            if d.year != season:
                backlog, season = 0.0, d.year
            ready = backlog + new_ready_bins(d, farm)
            rate = impact.rate(d)
            H, true_rate = actual[d]
            if d.dayofweek == 6:
                n = 0
            elif policy == POLICIES[0]:
                n = farm.crew_pool
            elif policy == POLICIES[1]:
                n = _sized_to_fruit(ready, rate, farm)
            elif policy == POLICIES[2]:
                f = test.loc[d]
                base = _sized_to_fruit(ready, rate, farm)
                n = (0 if f.nwp_rain_hours >= C.WASHOUT_HOURS
                     else base // 2 if f.nwp_rain_mm >= C.RAIN_MM else base)
            elif policy in POLICIES[3:5]:
                p = (pred_hist if policy == POLICIES[3] else pred_mos).loc[d]
                n = best_crew(band_probs(p.p_wet1, p.p_washout, p.p_wet7),
                              ready, rate, farm, impact)
            else:
                costs = day_outcome(np.arange(farm.crew_pool + 1), H, true_rate, ready, farm)
                n = int(np.argmin(costs["total_cost"]))
            if d.dayofweek == 6:
                out = day_outcome(0, 0.0, true_rate, ready, farm)
            else:
                out = day_outcome(n, H, true_rate, ready, farm)
            backlog = float(out["unpicked"])
            rows.append({"policy": policy, "date": d, "crew": n, "productive_hours": H,
                         "obs_rain_mm": test.loc[d, "obs_rain_mm"],
                         **{k: float(v) for k, v in out.items()},
                         "washout_paid_hours": float(out["paid_hours"]) if H < 2 else 0.0})
    daily = pd.DataFrame(rows)

    summary = daily.groupby("policy", sort=False).agg(
        avoidable_cost=("total_cost", "sum"), idle_cost=("idle_cost", "sum"),
        overhead_cost=("overhead_cost", "sum"),
        delay_cost=("delay_cost", "sum"), labour_cost=("labour_cost", "sum"), bins_picked=("picked", "sum"),
        washout_paid_hours=("washout_paid_hours", "sum"),
        worker_hours_offered=("worker_hours_offered", "sum"))
    weeks = len(days) / 7
    summary["hours_per_worker_per_week"] = (
        summary["worker_hours_offered"] / farm.crew_pool / weeks)
    base = summary.loc[POLICIES[1], "avoidable_cost"]
    best = summary.loc[POLICIES[-1], "avoidable_cost"]
    summary["saving_vs_no_weather"] = base - summary["avoidable_cost"]
    summary["share_of_possible_saving"] = summary["saving_vs_no_weather"] / (base - best)
    return summary, daily


def bootstrap_savings(daily: pd.DataFrame, n_boot: int = 2000, seed: int = 3) -> pd.DataFrame:
    """90% interval for each policy's saving vs the no-weather baseline.

    Resamples whole weeks (block bootstrap) because a bad call carries backlog into the
    following days, so single days are not independent.
    """
    cost = daily.pivot(index="date", columns="policy", values="total_cost")[POLICIES]
    diff = cost[POLICIES[1]].to_numpy()[:, None] - cost.to_numpy()
    week = (cost.index.year * 100 + cost.index.isocalendar().week.to_numpy())
    blocks = [diff[week == w].sum(axis=0) for w in np.unique(week)]
    blocks = np.array(blocks)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
    sims = blocks[idx].sum(axis=1)
    return pd.DataFrame({"saving": blocks.sum(axis=0),
                         "ci_low": np.percentile(sims, 5, axis=0),
                         "ci_high": np.percentile(sims, 95, axis=0)}, index=POLICIES)


def sensitivity(obs, test, pred_mos, pred_hist, impact, farm: FarmParams = C.FARM,
                delay_costs=(5, 10, 15, 25, 40), min_paid=(2, 4)) -> pd.DataFrame:
    """How much is the forecast worth under different farm economics?

    The answer depends on two numbers a workforce platform would know per customer: what a day of
    delay costs, and how many hours a called-in worker is guaranteed.
    """
    from dataclasses import replace
    rows = []
    for mp in min_paid:
        for dc in delay_costs:
            f = replace(farm, delay_cost=dc, min_paid_hours=mp)
            _, daily = run(obs, test, pred_mos, pred_hist, impact, f)
            ci = bootstrap_savings(daily)
            mos, best = ci.loc[POLICIES[4]], ci.loc[POLICIES[-1]]
            rows.append({"min_paid_hours": mp, "delay_cost_per_bin_day": dc,
                         "ai_mos_saving": round(mos.saving), "ai_mos_ci_low": round(mos.ci_low),
                         "ai_mos_ci_high": round(mos.ci_high),
                         "perfect_forecast_saving": round(best.saving),
                         "share_captured": round(mos.saving / best.saving, 2)
                         if best.saving > 0 else np.nan})
    return pd.DataFrame(rows)
