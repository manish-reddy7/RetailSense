# 3 Method

## 3.1 Agents and output schema

RetailSense uses specialised agents that each see one kind of evidence. Every agent returns the
same schema-validated object: a point forecast for horizons h = 1..7 days, a 10-90% band, a
confidence c in [0, 1], key factors and stated limitations. The orchestrator re-validates each
output. An invalid or abstaining agent is excluded from the consensus and flagged. It is never
converted to a zero forecast.

- **Historical agent.** It uses LightGBM with a Tweedie objective for the point forecast and two
  quantile models (alpha = 0.1, 0.9) for the band. Its evidence is lagged sales (lag h, 7, 14 and
  28; rolling mean and standard deviation over 7, 28 and 56 days; share of zero-sales days), price
  (level, one-week change, ratio to the 52-week maximum) and item identity. There is one direct
  model per horizon h, and every sales and price feature is lagged at least h days. If the
  quantile models cross the point forecast, the band is widened to include it.
- **Context agents: Calendar, Weather and Trend.** These are residual adjusters. Each agent
  forecasts `b + g_i(b, x_i)`, where b is the Historical agent's forecast and g_i is a LightGBM
  regressor. g_i sees b and the agent's own evidence x_i and nothing else. g_i is trained on
  *out-of-fold* baseline residuals from five expanding time folds, pooled over horizons, so that
  it never learns from in-sample baseline errors. The agent's band is the baseline band shifted
  by its adjustment.
  - Calendar evidence: weekday, month, week of year, day of month, event type, days to and since
    the nearest festival or holiday.
  - Weather evidence: target-day maximum and minimum temperature and precipitation, their
    anomalies against a same-season climatology built from previous years only, hot-day and
    rainy-day flags, and 3-day changes.
  - Trend evidence: scale-invariant weekly search-interest features.
- **Confidence.** For each agent, item and origin, `c_i = 1 / (1 + WAPE_i)`. WAPE_i is computed
  over the agent's out-of-sample forecasts for the trailing 28 revealed days. Items with fewer
  than 14 such days fall back to the pooled department WAPE.

## 3.2 Disagreement score

At the disagreement horizon h = 1, with weights `w_i = c_i / sum_j c_j` and consensus
`y_bar = sum_i w_i y_i`:

    D = sqrt( sum_i w_i (y_i - y_bar)^2 ) / max(y_bar, 1)

This is a confidence-weighted coefficient of variation. The floor of one unit keeps
slow-moving items from producing extreme values. D is unchanged when every forecast is scaled
by the same constant, and it is 0 when all forecasts agree (both properties are unit-tested).
Two variants are used as sensitivity checks: `D_range = (max y_i - min y_i) / max(y_bar, 1)`,
and the unnormalised weighted standard deviation `D_std`.

## 3.3 Calibration and routing

The threshold tau is the 80th percentile of D over the validation split, so on validation about
20% of forecasts are deliberated. It is computed once and written into the configuration before
any test data are read. Test results never feed back into it. If D < tau, the forecast is the
confidence-weighted consensus (the *fast path*). If D >= tau, the case is deliberated.

## 3.4 Critic and bounded revision

Deliberation is one round, implemented as a LangGraph state machine (load context -> agents ->
disagreement -> route -> consensus or critic -> revise -> inventory).

- **Critic.** The critic receives each agent's h = 1 forecast, band, confidence, evidence
  description, key factors and limitations, plus the item's last 14 days of revealed sales. It
  returns a JSON report: the main conflict, the conflicting agents, a strong/moderate/weak
  judgement of each agent's evidence, missing or stale evidence, and one question per agent. The
  report schema has no numeric field, so the critic cannot emit a demand number. tau is not in
  the prompt.
- **Revision.** Each agent's revision call receives its own view, the critic report and its
  question. It returns a confidence multiplier m_i, a revised value y'_i, a rationale, and
  citations that must refer to evidence in the critic report. Holding position is presented as
  a normal answer.
- **Code does the arithmetic.** Code clips m_i to [0.25, 2.0] and y'_i to the agent's own
  [lower, upper] band, then computes the final forecast `sum_i w'_i y'_i` with
  `w'_i proportional to c_i m_i`. Horizons h = 2..7 keep the original agent forecasts with the
  new weights w'_i. Bands are the w'-weighted agent bands.
- **Failure handling.** Invalid output gets one corrective retry. If it is still invalid, the
  case falls back to the fast path and the failure is logged. Calls use a pinned model at
  temperature 0. They are cached on disk, keyed by model, prompt version, prompt text and input.

## 3.5 Inventory policy

A periodic-review (s, S) policy turns each system's forecast into an action. It uses lead time
L = 3 days, coverage R = 7 days, service level 95% (z = 1.65) and case pack 6. The band gives
`sigma_h = (upper_h - lower_h) / 2.563`. Then:

- `s = sum_{h<=L} f_h + z sqrt(sum_{h<=L} sigma_h^2)`
- `S = sum_{h<=L+R} f_h + z sqrt(sum_{h<=L+R} sigma_h^2)`

Forecasts beyond day 7 repeat day 7. With inventory position IP = on hand + on order:

- IP <= s: RESTOCK S - IP, rounded up to whole case packs.
- IP above S plus one week of forecast demand: REDUCE.
- Otherwise: HOLD.

The policy is a pure function of numbers. No LLM output reaches the order quantity except
through the code-computed forecast, and a test asserts that no module in the inventory package
imports the LLM layer.
