"""End-to-end: data -> features -> weather models -> impact model -> backtest -> report.

    python -m wwp.pipeline            (from the repo root, with PYTHONPATH=src)
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import backtest as B  # noqa: E402
from . import config as C  # noqa: E402
from . import data_weather as W  # noqa: E402
from . import features as F  # noqa: E402
from . import weather_models as M  # noqa: E402
from . import workforce as WF  # noqa: E402

warnings.filterwarnings("ignore")

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#b4b2aa", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRAY,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _bar_models(metrics: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (key, title) in zip(axes, [("rain", "Rain >= 1 mm in work hours"),
                                       ("washout", "Washout (>= 4 wet work hours)")]):
        m = {k: v for k, v in metrics[key].items() if not k.startswith("_")}
        names, vals = list(m), [m[k]["pr_auc"] for k in m]
        colors = [BLUE if n.startswith("MOS") else GRAY for n in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=colors, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", color=MUTED, fontsize=9)
        ax.axvline(metrics[key]["_base_rate_test"], color=ORANGE, lw=1.5, ls="--")
        ax.text(metrics[key]["_base_rate_test"] + 0.01, y[0] + 0.45, "no-skill", color=MUTED,
                fontsize=8)
        ax.set_yticks(y, names)
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("PR-AUC on 2025-2026 (higher is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _reliability(test: pd.DataFrame, pred: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
    for col, target, color, label in [("p_rain", "y_rain", BLUE, "Rain >= 1 mm"),
                                      ("p_washout", "y_washout", ORANGE, "Washout")]:
        bins = pd.cut(pred[col], [0, .1, .2, .35, .5, .65, .8, 1.0], include_lowest=True)
        g = pd.DataFrame({"p": pred[col], "y": test.loc[pred.index, target]}).groupby(
            bins, observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        g = g[g.n >= 5]
        ax.plot(g.p, g.y, color=color, lw=2, marker="o", ms=6, label=label)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration, MOS retrained yearly (2025-26)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bar_backtest(ci: pd.DataFrame, path):
    ci = ci.drop(index=B.POLICIES[1])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(ci))[::-1]
    colors = [BLUE if "AI planner" in p else GRAY for p in ci.index]
    ax.barh(y, ci.saving / 1000, color=colors, height=0.6)
    ax.errorbar(ci.saving / 1000, y, xerr=[(ci.saving - ci.ci_low) / 1000,
                                           (ci.ci_high - ci.saving) / 1000],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, ci.index)
    ax.set_xlabel("Saving vs 'crew sized to ripe fruit, no weather'\n"
                  "NZ$ thousands, 2025 + 2026 harvests, bars show 90% CI")
    ax.set_title("Backtest: avoidable cost saved vs ignoring the weather")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    print("1/6 weather data")
    obs, nwp = W.fetch_observed(), W.fetch_nwp_day_ahead()
    print("2/6 features")
    df = F.build_dataset(obs, nwp)
    print("3/6 weather models")
    serving, hist_models, metrics, test = M.train_and_evaluate(df)
    print("4/6 harvest log + impact model")
    log = WF.generate_harvest_log(obs, df, range(2010, 2024))
    impact = WF.ImpactModel().fit(log)
    print("5/6 backtest")
    pred_mos = M.walk_forward(df, hist_models, test)
    no_nwp = test.copy()
    no_nwp[F.NWP_COLS] = np.nan
    pred_hist = M.WeatherRiskModel(hist_models).predict(no_nwp)
    summary, daily = B.run(obs, test, pred_mos, pred_hist, impact)
    ci = B.bootstrap_savings(daily)
    print("     sensitivity grid")
    sens = B.sensitivity(obs, test, pred_mos, pred_hist, impact)

    print("6/6 artifacts")
    for d in (C.MODEL_DIR, C.REPORT_DIR, C.DATA_DIR / "synthetic"):
        d.mkdir(parents=True, exist_ok=True)
    log.to_csv(C.DATA_DIR / "synthetic" / "harvest_log_2010_2023.csv")
    replay = df[df.index > C.VALID_END].join(pred_mos, rsuffix="_pred")
    joblib.dump({"serving": serving, "impact": impact, "replay": replay},
                C.MODEL_DIR / "bundle.joblib")
    summary.round(2).to_csv(C.REPORT_DIR / "backtest_summary.csv")
    ci.round(0).to_csv(C.REPORT_DIR / "backtest_savings_ci.csv")
    daily.to_csv(C.REPORT_DIR / "backtest_daily.csv", index=False)
    sens.to_csv(C.REPORT_DIR / "sensitivity.csv", index=False)
    metrics["impact_model"] = impact.summary()
    metrics["backtest"] = {p: {"avoidable_cost": round(float(r.avoidable_cost)),
                               "saving_vs_no_weather": round(float(ci.loc[p, "saving"])),
                               "saving_90ci": [round(float(ci.loc[p, "ci_low"])),
                                               round(float(ci.loc[p, "ci_high"]))],
                               "share_of_possible_saving": round(float(r.share_of_possible_saving), 2),
                               "washout_paid_hours": round(float(r.washout_paid_hours)),
                               "hours_per_worker_per_week": round(float(r.hours_per_worker_per_week), 1)}
                           for p, r in summary.iterrows()}
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    _bar_models(metrics, C.REPORT_DIR / "model_comparison.png")
    _reliability(test, pred_mos, C.REPORT_DIR / "calibration.png")
    _bar_backtest(ci, C.REPORT_DIR / "backtest_savings.png")
    print(summary[["avoidable_cost", "share_of_possible_saving"]].round(2))
    print(sens.to_string(index=False))
    print("done ->", C.REPORT_DIR)


if __name__ == "__main__":
    main()
