# Code walkthrough

A step-by-step guide to the whole codebase, in the order the data flows.

---

## Step 0: The big picture

**The question:** at 6 pm a manager picks how many apple pickers to call for tomorrow. If it rains, called-in workers still get paid for guaranteed hours. If too few are called, ripe fruit loses value on the tree.

**The flow:**

```
weather download → daily features → rain-risk model → P(wet-hour band)
                                                        ↓
simulated harvest log → impact model (productive hours per band) → cost optimiser → crew size
                                                        ↓
                                     backtest (would this have saved money?) + API/dashboard
```

`pipeline.main()` runs the whole chain, and its 6 printed steps match the steps below.

---

## Step 1: [config.py](../src/wwp/config.py), every assumption in one place

- **Location and time zone** for Hastings, NZ.
- **Time split** ([config.py:25-27](../src/wwp/config.py#L25-L27)):
  - train ≤ 2023
  - validate 2024
  - test 2025 onward
  - The data is never shuffled, so future data can't leak into training.
- **Work window** 07:00–17:00 (10 hours). **Decision cutoff** 18:00.
- **Rain definitions:**
  - "wet hour" = ≥ 0.3 mm in that hour
  - "rain day" = ≥ 1 mm during work hours
  - "washout" = ≥ 4 wet work hours
- **`FarmParams`**, a frozen dataclass of farm economics: 40 pickers, $30/h wage, 4 guaranteed hours, $25 daily overhead per worker, 0.8 bins per picker per hour, $15 per ripe bin per day of delay. Being frozen, it's changed with `dataclasses.replace(...)`, which the backtest and API both do.

## Step 2: [data_weather.py](../src/wwp/data_weather.py), downloading weather

- `_get_hourly` calls an Open-Meteo URL and returns an hourly DataFrame indexed by time.
- `fetch_observed` gets **what actually happened** (ERA5 reanalysis, 2010–2026). It downloads one year at a time (`_year_chunks`) and caches the result to `data/raw/*.csv`.
- `fetch_nwp_day_ahead` gets **what the forecast said the day before** (the `*_previous_day1` variables, available from 2024). This matters: training on the forecast as it was issued avoids using future knowledge.
- `fetch_live` gets the last 7 days plus 3 forecast days, for the live API endpoint.

## Step 3: [features.py](../src/wwp/features.py), one row per target day T

This is the most important file for correctness.

- **`targets(obs)`** describes what happened on day T during work hours: total mm, number of wet hours, max gust and max temperature. From these it builds 0/1 labels: `y_rain`, `y_wet1` (≥ 1 wet hour), `y_washout` (≥ 4), `y_wet7` (≥ 7).
- **`history_features(obs)`** is what the manager can know at 18:00 on day T−1:
  - It keeps only hours before 18:00 (`before_cut`, [line 47](../src/wwp/features.py#L47)).
  - Snapshot values come from the 17:00 hour: humidity, cloud, pressure and wind. Wind is converted into u/v components because a direction angle like 359° vs 1° doesn't work as a number.
  - Pressure tendency is the change from 05:00 to 17:00. Falling pressure suggests an incoming front.
  - Rain yesterday and over the previous 7 days use `.shift(1)`, so only past days are counted.
  - **The key trick** ([line 74](../src/wwp/features.py#L74)): `d.index + 1 day`. Features computed on day T−1 get relabelled as the row for day T.
  - Day of year is encoded as sin/cos so that 31 Dec and 1 Jan sit next to each other.
- **`nwp_features(nwp)`** summarises the forecast for day T's work hours. It drops days with incomplete hours.
- **`build_dataset`** joins history features and targets (inner join), then forecast features (left join). Rows from before 2024 have no forecast, so those columns are NaN.
- The column lists at [lines 101-111](../src/wwp/features.py#L101-L111) define the inputs for each model. `MOS_COLS` = the forecast plus a few local signals.

## Step 4: [weather_models.py](../src/wwp/weather_models.py), predicting rain risk

**A ladder of models**, each of which has to beat the simpler one before it:

1. climatology
2. persistence
3. logistic regression
4. LightGBM on history only
5. the forecast alone, calibrated
6. **MOS** (forecast + local signals)

What each piece does:

- **`TARGETS`**: the four yes/no questions (rain ≥ 1 mm, ≥ 1, ≥ 4 and ≥ 7 wet hours). Three stacked thresholds give a distribution over wet hours.
- **`_linear`** builds a pipeline: log-transform rain columns (they are very skewed) → standardise → model.
- **`fit_history_models`** trains a LightGBM classifier per target, with early stopping on 2024 data. It also trains a Tweedie regressor for expected mm (Tweedie suits data that is often 0 and sometimes large). This model is the **fallback** when no forecast is available.
- **`fit_forecast_models`** trains small logistic models: `*_nwp` uses one raw forecast column, and `*_mos` uses all of `MOS_COLS`. They are small because there are only about 2.7 years of forecasts.
- **`WeatherRiskModel.predict`** is the serving wrapper:
  - It uses MOS on rows that have a forecast and the history model on the rest.
  - It then **forces the probabilities to be nested** ([line 114](../src/wwp/weather_models.py#L114)): P(≥ 7) ≤ P(≥ 4) ≤ P(≥ 1). Separately trained models can disagree, and the band maths in step 6 would otherwise produce negative probabilities.
- **`walk_forward`**: for each test year, it retrains the forecast models on earlier years only, the way production would retrain every winter.
- **`train_and_evaluate`** scores each rung on the 2025–26 test rows. It uses PR-AUC and Brier score because rain is rare: an "always dry" model is 82% accurate and useless. It returns the serving model, which is retrained on all forecast data.

## Step 5: [workforce.py](../src/wwp/workforce.py), the simulator, cost function and impact model

**Simulation rules** ([lines 17-21](../src/wwp/workforce.py#L17-L21), `productive_hours`):

- A wet hour stops picking, and so does the hour after it (drying).
- Gusts ≥ 65 km/h stop picking.
- Hours ≥ 30 °C count as 0.7 of an hour.
- The total is capped at an 8-hour shift and gets ±5% random noise.

**Ripening:** `new_ready_bins` follows a sine curve from 25% up to 100% of peak (180 bins/day) mid-season, then back down. Crews are 15% slower in the first two weeks.

**`day_outcome(n, H, rate, ready)`** is the core cost function. Read it slowly:

| quantity | formula |
|---|---|
| picked | min(n·H·rate, ready) |
| worked hours | min(H, hours needed to clear the fruit) |
| paid hours | n · max(guaranteed 4 h, worked) |
| **idle** cost | wage × (paid − actually worked) |
| **overhead** | $25 · n · (unproductive share of the shift) |
| **delay** | $15 × bins left unpicked |
| **total** | idle + overhead + delay |

Wages for actual picking are left out on purpose. The fruit has to be picked eventually, so that money is spent no matter what. Only *wasted* money counts (design log item 1 in the README).

**Worked example** (128 bins ready, rate 0.8, dry day H = 8):

- n = 20 → can pick exactly 128 → idle 0, overhead 0, delay 0 → **$0**
- n = 19 → picks 121.6 and leaves 6.4 bins → delay **$96**
- n = 21 → finishes in 7.6 h, so a little overhead is wasted → **$25**
- If it turns out to be a washout (H = 0), n = 20 costs 80 h × $30 + $500 overhead + $1,920 delay = **$4,820**, while n = 0 costs only the **$1,920** delay. So the forecast changes the answer.

The vectorised shapes matter in step 6: `n` can be a column vector and `H` a row vector, so one call evaluates every crew size against every scenario.

**`generate_harvest_log`** simulates 2010–2023 seasons:

- The backlog carries over between days.
- Sundays have no picking.
- The "historical manager" calls 85–100% of the crew, or 70% of that if it was raining at roster time.
- The output looks like a timesheet export. This is the function you'd replace with real data.

**`ImpactModel`** learns from that log:

- It stores the actual productive hours seen for each wet-hour band ("0", "1-3", "4-6", "7+").
- It stores the median picking rate for early season and for the main season.

## Step 6: [decision.py](../src/wwp/decision.py), choosing the crew

- **`band_probs`** turns the three stacked probabilities into four band probabilities. For example, P(1-3) = P(≥ 1) − P(≥ 4).
- **`expected_outcomes`**: for every n from 0 to 40 and each band, it runs `day_outcome` over *all historical productive-hour values in that band*. It averages within the band and then weights by the band's probability. The result is an expected-cost curve over n.
- **`best_crew`** = argmin of that curve.
- **`recommend`** builds the output for people:
  - four options: full crew, dry-day crew, AI pick, and nobody
  - a headline chosen by comparing the AI crew with the dry-day crew
  - reasons in plain text
  - the cost curve, which the dashboard plots

## Step 7: [backtest.py](../src/wwp/backtest.py), would it have saved money?

- **`run`** replays every harvest day of 2025–26 under 6 policies. Each day's real productive hours and picking rate are drawn **once**, before the policy loop ([line 40](../src/wwp/backtest.py#L40)). This means all policies face identical conditions ("common random numbers"), and differences come from the decisions alone.
  - Policy 2 ("sized to ripe fruit") is the **baseline**: ceil(ready / (rate × 8)).
  - The "perfect forecast" policy optimises using the true H. That gives the ceiling for any weather model.
  - The backlog carries over, so a bad call keeps costing on later days.
- **`bootstrap_savings`** produces confidence intervals by resampling **whole weeks**, not single days. Days are linked through the backlog, so they aren't independent.
- **`sensitivity`** reruns everything over a grid of delay costs and guaranteed hours. That grid produces the README's "when does weather AI pay" table.

## Step 8: [pipeline.py](../src/wwp/pipeline.py), running everything

`main()` runs steps 2→7:

- It builds `pred_hist` by blanking the forecast columns ([line 112](../src/wwp/pipeline.py#L112)), which forces the history-only fallback.
- It then writes:
  - `models/bundle.joblib` (serving model + impact model + replay table)
  - CSVs and `metrics.json`
  - 3 charts in `reports/`

## Step 9: [app/main.py](../app/main.py), the API

- `bundle()` loads the joblib file lazily, once, and returns 503 if the pipeline hasn't been run.
- `FarmOverrides` / `WhatIf` are Pydantic models that check inputs. `_farm` applies only the overrides that were actually supplied.
- Endpoints:
  - `/api/plan?date=` replays a past day from the stored predictions and adds what actually happened.
  - `/api/plan/live` fetches live weather → builds features → `serving.predict` → recommendation.
  - `POST /api/recommend` is a what-if: you supply the probabilities yourself.
  - `/` serves the dashboard.

## Step 10: [tests/test_core.py](../tests/test_core.py)

Each test is a small, readable statement of a rule the code must follow:

- **No leakage:** rain at 20:00 doesn't change the next day's features.
- **Targets:** they count only working hours.
- **Bands:** probabilities sum to 1 and are never negative.
- **Washout:** a washout costs exactly the guaranteed hours.
- **Monotonic:** more rain risk never means a bigger crew.
- **Dry-day crew:** it equals ripe fruit ÷ capacity (the 128-bin example above).
- **API smoke test:** runs only if a trained model exists.

---

## A few things worth knowing as you read

- [features.py:54](../src/wwp/features.py#L54) hardcodes `0.3` instead of using `C.WET_HOUR_MM`. It's harmless now, but it would go out of sync if the config changed.
- `labour_cost` is calculated in `day_outcome` but isn't included in `total_cost`. That's intentional, for the reason in step 5.
- In `generate_harvest_log`, `daily.loc[d, "today_rain_mm"]` means rain on the *day before d*, up to 18:00, because of the re-indexing in step 3.

**How to learn it hands-on:** run `pytest -q`, then open a Python shell and call `day_outcome` and `recommend` with your own numbers, as in the worked example.
