"""Fast tests: no network, no trained model needed (except the API smoke test)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wwp import config as C  # noqa: E402
from wwp import features as F  # noqa: E402
from wwp.decision import DRY, band_probs, best_crew, recommend  # noqa: E402
from wwp.workforce import BANDS, ImpactModel, day_outcome  # noqa: E402


def _hourly(days=12, rain_on=None):
    idx = pd.date_range("2025-03-01", periods=24 * days, freq="h")
    df = pd.DataFrame({
        "temperature_2m": 18.0, "relative_humidity_2m": 70.0, "precipitation": 0.0,
        "pressure_msl": 1015.0, "cloud_cover": 50.0, "wind_speed_10m": 10.0,
        "wind_direction_10m": 270.0, "wind_gusts_10m": 20.0}, index=idx)
    if rain_on is not None:
        df.loc[rain_on, "precipitation"] = 2.0
    return df


def _impact():
    imp = ImpactModel()
    imp.hours = {"0": np.full(50, 8.0), "1-3": np.full(50, 7.0),
                 "4-6": np.full(50, 4.0), "7+": np.full(50, 0.3)}
    imp.rate_early = imp.rate_main = 0.8
    return imp


def test_features_never_use_hours_after_cutoff():
    """Rain at 20:00 on day D must not appear in the features for day D+1."""
    late = _hourly()
    late.loc["2025-03-10 20:00", "precipitation"] = 50.0
    base = F.history_features(_hourly())
    feat = F.history_features(late)
    assert feat.loc["2025-03-11", "today_rain_mm"] == base.loc["2025-03-11", "today_rain_mm"]


def test_targets_count_only_working_hours():
    obs = _hourly(rain_on=pd.date_range("2025-03-05 07:00", "2025-03-05 11:00", freq="h"))
    obs.loc["2025-03-06 02:00", "precipitation"] = 30.0          # night rain
    t = F.targets(obs)
    assert t.loc["2025-03-05", "obs_rain_hours"] == 5
    assert t.loc["2025-03-05", "y_washout"] == 1
    assert t.loc["2025-03-06", "obs_rain_mm"] == 0               # 2am does not count


def test_band_probs_sum_to_one_and_are_nested():
    p = band_probs(0.6, 0.7, 0.9)    # inconsistent input gets clipped to nested
    assert set(p) == set(BANDS)
    assert abs(sum(p.values()) - 1) < 1e-9
    assert min(p.values()) >= 0


def test_washout_costs_guaranteed_hours():
    farm = C.FARM
    out = day_outcome(10, 0.0, 0.8, 200, farm)
    assert out["picked"] == 0
    assert out["idle_cost"] == pytest.approx(10 * farm.min_paid_hours * farm.wage)


def test_higher_washout_risk_never_increases_crew():
    imp, farm = _impact(), C.FARM
    crews = [best_crew(band_probs(p, p, p), 150, 0.8, farm, imp) for p in np.linspace(0, 1, 11)]
    assert all(a >= b for a, b in zip(crews, crews[1:]))
    assert crews[0] > 0 and crews[-1] == 0


def test_dry_day_crew_matches_ripe_fruit():
    imp, farm = _impact(), C.FARM
    n = best_crew(DRY, 128, 0.8, farm, imp)                       # 128 bins / 6.4 per picker
    assert n == 20


def test_recommend_shape():
    rec = recommend({"p_wet1": 0.9, "p_washout": 0.8, "p_wet7": 0.7}, 150, 0.8, C.FARM, _impact())
    assert len(rec["cost_curve"]) == C.FARM.crew_pool + 1
    assert rec["options"][2]["option"] == "AI recommendation"
    assert "washout" in rec["headline"].lower()


@pytest.mark.skipif(not (C.MODEL_DIR / "bundle.joblib").exists(), reason="model not trained")
def test_api_smoke():
    from fastapi.testclient import TestClient
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.main import app
    c = TestClient(app)
    day = c.get("/api/dates").json()["dates"][0]
    r = c.get("/api/plan", params={"date": day})
    assert r.status_code == 200 and "actual" in r.json()
    r = c.post("/api/recommend", json={"p_wet1": 0.2, "p_washout": 0.05, "ready_bins": 120})
    assert r.status_code == 200
    assert c.get("/api/plan", params={"date": "1999-01-01"}).status_code == 404
