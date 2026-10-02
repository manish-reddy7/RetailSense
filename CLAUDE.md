# RetailSense — project instructions for Claude Code

## What this project is
RetailSense: A Disagreement-Triggered Multi-Agent Framework for Real-Time Inventory
Demand Forecasting and Restocking Decisions. A 16-week single-student capstone.

Specialised agents (Historical, Weather, Trend, Calendar) forecast daily demand.
An orchestrator computes a normalised disagreement score D. If D < tau: confidence-weighted
consensus. If D >= tau: an LLM Critic explains the conflict, then ONE bounded revision round.
The final forecast feeds an (s, S) inventory policy -> RESTOCK / HOLD / REDUCE, evaluated in a
simulator against baselines.

Full spec: docs/RetailSense_Complete_Project_Blueprint.pdf (source of truth).
Also read: docs/rules.md, docs/architecture.md, docs/design.md, docs/phases.md.
If the blueprint and an older doc disagree, the blueprint wins.

## Frozen decisions
- Data: India kirana chain from configs/india_generator.yaml (fixed seed). Store MUM_1, dept KIRANA,
  all 60 items (ranked on train volume). Dataset id `india`; M5 was removed 2026-09-30.
- Splits: train d_1–d_1799, validation d_1800–d_1885, test d_1886–d_1941. Never shuffle.
- Horizon: days 1–7, one direct model PER horizon h; its sales/price features lagged >= h days.
  (Changed from "one model, lags >= 7" after Phase 1 validation, approved 2026-09-29.)
- Historical Agent: LightGBM (Tweedie) + quantile models (alpha 0.1 and 0.9).
- Weather/Trend/Calendar Agents: LightGBM residual adjusters on OUT-OF-FOLD baseline residuals
  (5 expanding time folds). Forecast = baseline + adjustment.
- Confidence: c_i = 1 / (1 + trailing 28-day WAPE of agent i on that item).
- Disagreement: w_i = c_i / sum(c); y_bar = sum(w_i * y_i);
  D = sqrt(sum(w_i * (y_i - y_bar)^2)) / max(y_bar, 1). Also compute D_range as a sensitivity check.
- tau = 80th percentile of validation D, frozen before test.
- Revision: LLM may set confidence multiplier m_i in [0.25, 2.0] and a revised value inside the
  agent's own [lower, upper] band. Code clips and computes the final number. The LLM NEVER
  writes the final forecast or an order quantity.
- Inventory: L=3, R=7, z=1.65, holding 25%/yr of price, order cost $5, lost-sales cost 1x price,
  case pack 6. s = D_L + z*sigma_L; S = D_{L+R} + z*sigma_{L+R}; sigma_h = (upper-lower)/2.563.
- Weather: Open-Meteo historical archive, Mumbai (19.08, 72.88) for MH.
- Interest: weekly interest index in Google Trends CSV format (data/raw/india/trends/).

## Non-negotiable rules
1. No future-data leakage. Every feature row has `available_at`; tests assert
   available_at <= prediction_time. Weekly signals usable only after their week closes.
2. Never touch the test split except in the final experiment run. Tune only on validation.
3. Never claim live POS access or real store sales.
4. LLM outputs are pydantic-validated JSON. On failure: one retry, then fall back to fast
   consensus and log it.
5. Cache every LLM and API call on disk (data/cache/). Temperature 0. Model id comes from
   config, API key from the environment variable in .env — never hard-code keys.
6. Every baseline and RetailSense go through the identical inventory simulator.
7. Report results honestly, including when RetailSense does not win.
8. Stay in MVP scope: no RL, no dynamic pricing, no extra agents, no multi-round deliberation.

## Engineering conventions
- Python 3.11. Package in src/retailsense/. Config in configs/*.yaml, loaded with pydantic.
- Stack: pandas, numpy, pyarrow, lightgbm, statsmodels, scikit-learn, langgraph, pydantic v2,
  openmeteo-requests, requests-cache, scipy, mlflow, streamlit, plotly, pytest, ruff.
- Every module gets tests in tests/. Run `pytest -q` and `ruff check .` before saying a task is done.
- Log every experiment to MLflow (params, metrics, config hash, git commit).
- Store run outputs in SQLite (data/retailsense.db) using the tables in blueprint Section 10.1.
- Small, reviewable changes. After each phase, update docs/PROGRESS.md with: what was built,
  metrics obtained, open issues, and decisions made.
- If something in the spec is ambiguous or seems wrong, stop and ask me rather than guessing.