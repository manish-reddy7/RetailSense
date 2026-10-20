# When Does Disagreement-Triggered LLM Deliberation Help Demand Forecasting? A Controlled Simulation and a Real-Data Study

**Author:** [Author name]
**Affiliation:** [Department, University]
**Email:** [Corresponding author email]

---

## Abstract

Calling a large language model (LLM) on every demand forecast is costly. This paper tests whether
disagreement among specialised forecasting agents can decide when an LLM should deliberate. In
RetailSense, a LightGBM Historical agent and residual-adjusting Calendar, Weather and Trend agents
forecast daily item demand. A confidence-weighted disagreement score D triggers an LLM critic and
one bounded revision round when D reaches a threshold τ, frozen at the 80th percentile of
validation D. The LLM never sets the final forecast or the order quantity: code clips its
proposals, and every system passes through the same (s, S) inventory simulator. The protocol is
applied, with one locked test run per dataset, to 60 items of a simulated Mumbai kirana store
with logged demand drivers and to 60 items of real sales from the Czech online grocer Rohlik.
Disagreement is a weak risk signal (India test: Spearman ρ = 0.072, AUROC = 0.539; Rohlik test:
ρ = 0.119, AUROC = 0.579). Selective deliberation matches always-on accuracy with 73% (India) and
83% (Rohlik) fewer LLM calls, but it is not reliably better than a random trigger at the same rate
(Diebold-Mariano p = 0.08 and 0.57). Deliberation improved triggered cases only on the India test
split (Wilcoxon p = 0.007), and the effect did not appear on validation or on Rohlik. A single
all-features LightGBM (India) and ETS (Rohlik) are more accurate than every multi-agent system.
Agent forecasts are correlated above 0.99, so D mostly measures adjustment size, and only one 8B
LLM critic was tested.

**Keywords:** demand forecasting, multi-agent systems, large language models, forecast
disagreement, selective deliberation, inventory control, LightGBM

---

## I. Introduction

A store manager who restocks shelves every week has to decide how much of each item to order
before demand is known. Ordering too little loses sales; ordering too much ties up cash and shelf
space. The forecast behind that decision draws on several kinds of evidence: recent sales, the
calendar (weekdays, festivals and public holidays), the weather and shifts in customer interest.
Machine-learning models such as LightGBM [1] combine these signals well, and gradient-boosted
models dominated the large M5 retail benchmark [2]. Large language models (LLMs) may add a
different ability: in principle they can read conflicting evidence and reason about which source
to trust [3]. Each LLM call, however, costs time and money. Calling the model on every item and
every day is expensive, and the value of doing so is not obvious [4].

The closest prior system, and the base work for this paper, is MARIOD [5]. It couples a
transformer demand forecaster with hierarchical multi-agent reinforcement learning at store,
distribution-centre and corporate level, and reports lower forecast error and fewer stockouts
than deep-learning and reinforcement-learning baselines. RetailSense addresses the same problem,
joint demand forecasting and restocking with specialised agents, but changes three things. Its
agents are split by *evidence source* rather than by supply-chain level. They coordinate through
an explicit disagreement score and one bounded LLM revision rather than a learned communication
protocol. Orders come from a transparent (s, S) policy rather than a learned policy. The
evaluation protocol also differs: a single locked test run with a frozen threshold, missing sales
left missing rather than zero-filled, and results reported where RetailSense loses.

This paper asks whether an LLM can be used *selectively*, only on the forecasts where it is most
likely to help. The idea comes from forecast combination and ensemble forecasting. Combining
forecasts usually beats a single forecast [6]–[8], and the spread between forecasters is a signal
of uncertainty, although an imperfect one [9], [10]. If several forecasters disagree, the case
may be risky, and a closer look may be worth its cost.

### A. Approach

RetailSense builds a small set of specialised agents, each of which sees one kind of evidence. A
Historical agent forecasts from lagged sales and price. Calendar, Weather and Trend agents adjust
that baseline using only their own signal. Each agent returns a daily forecast for the next seven
days, an uncertainty band and a confidence. The orchestrator computes a normalised disagreement
score D, a confidence-weighted coefficient of variation of the agents' next-day forecasts with the
denominator floored at one unit. If D is below a threshold τ, the system returns the
confidence-weighted consensus. If D is at or above τ, an LLM critic explains the conflict and each
agent receives one bounded revision. This design follows critique-then-revise and multi-agent
debate methods [3], [11] but limits them to a single round. It is also a form of selective LLM
use, close to cascades and routers [12], [13].

The LLM never produces the final forecast or an order quantity. Unlike LLM-as-forecaster methods
[14], [15], it may only propose a confidence multiplier and a revised value for each agent's
forecast. Code clips the multiplier to [0.25, 2.0] and the revised value to the agent's own
10–90% band, then computes the final forecast as a confidence-weighted average. The forecast
feeds a periodic-review (s, S) inventory policy [16] that outputs RESTOCK, HOLD or REDUCE, and the
order quantity is always computed by code.

### B. Research Question and Hypotheses

The main research question is: *does disagreement-triggered deliberation among heterogeneous
forecasting agents improve demand forecasting and inventory restocking decisions compared with
statistical forecasting, single-agent forecasting, simple multi-agent consensus and always-on
deliberation?* It is split into five hypotheses:

- **H1.** Normalised disagreement D is positively associated with forecast error.
- **H2.** Selective deliberation lowers error on triggered cases relative to no deliberation.
- **H3.** Selective deliberation matches always-on deliberation at a fraction of the LLM cost.
- **H4.** Disagreement is a better trigger than chance.
- **H5.** Better forecasts translate into better inventory outcomes.

H4 is the central test of the idea: if a random trigger at the same rate does as well, the
disagreement signal adds nothing.

### C. Contributions

1. **A frozen protocol with leakage controls.** Every feature row records when it became
   available, and tests assert that no feature is newer than the prediction time. τ is set as the
   80th percentile of validation D, and τ and the prompt are frozen before the test split is read.
   Each dataset's test split is read by a single final run. The India run was resumed once, after
   the first of two GPU out-of-memory crashes (a retry absorbed the second), and the Rohlik run was
   resumed twice, after a laptop restart and a manual stop. Each resume used the same configuration
   hash and replayed completed cases from the cache with no new LLM calls.
2. **A simulator with logged ground-truth drivers.** The primary dataset is a simulated Indian
   kirana (neighbourhood grocery) chain. Its sales are generated, not observed. They are built from
   real Open-Meteo weather [17] and a real Indian festival calendar, and every multiplier behind
   each simulated sale is logged, so agent behaviour can be checked against the true drivers.
3. **An evaluation through one inventory simulator.** Nine systems, from seasonal naive to
   always-on deliberation, pass through the same (s, S) simulator, because forecast accuracy and
   inventory cost do not always agree [18], [19]. A random-trigger control at the same rate
   isolates the value of D.
4. **A real-data study.** The protocol is applied to real daily sales from Rohlik, a Czech online
   grocer [20], with the same code and prompt, but with three agents (no search-interest data, so
   no Trend agent) and a shorter ETS fit window.

### D. Summary of Findings

The results are mostly negative and are reported as such. On the India test split, RetailSense
does not beat a single LightGBM that sees all features (h = 1 WAPE 0.3775 vs 0.3688). D is only
weakly associated with error (Spearman ρ = 0.072, AUROC = 0.539), so H1 is weakly supported. On
triggered cases, deliberation improved 30.3% of forecasts and worsened 24.6% (Wilcoxon
p = 0.007), but this small effect was not significant on validation. RetailSense matches always-on
accuracy with 1,331 rather than 5,018 LLM calls per 1,000 forecasts (−73%), which supports H3 on
accuracy and LLM cost but not on inventory cost (always-on $199,868 vs RetailSense $206,313; the
difference is not significant). H4 is not supported: the disagreement trigger is not reliably
better than a random trigger at the same rate. H5 is not supported for RetailSense: ETS gives the
lowest inventory cost ($138,621 vs $206,313).

On the Rohlik test split, D is weakly but significantly associated with error (ρ = 0.119, 95% CI
[0.063, 0.176]; AUROC = 0.579). Deliberation leaves triggered cases unchanged (27.6% improved,
26.6% worsened, Wilcoxon p = 1.00), and selective triggering is no better than random at the same
17% rate (WAPE difference −0.0002, CI [−0.0010, 0.0006], DM p = 0.57). Selective deliberation uses
83% fewer LLM calls than always-on (696 vs 4,051 per 1,000 forecasts) at no significant accuracy
cost. ETS is the most accurate Rohlik system (h = 1 WAPE 0.150 vs 0.175 for RetailSense,
DM p < 0.001), and the all-features LightGBM has the lowest inventory cost ($133,467 vs $287,344).

Taken together, disagreement between these agents is a weak risk signal, and a small 8B LLM critic
does not use it to improve forecasts selectively. The agents' forecasts are highly correlated
(0.994–0.997 on Rohlik, and above the 0.95 design gate on India too), so D mostly measures the size
of small context adjustments, and only one 8B local LLM was tested. These findings are consistent
with work showing that LLM self-correction is often unreliable [21], although this setting gives
the model some external evidence.

### E. Paper Organisation

Section II reviews related work. Section III describes the agents, the disagreement score, the
critic and revision, and the inventory policy. Section IV sets out the data, the systems compared
and the statistical tests. Section V reports the results for each hypothesis. Section VI discusses
why the disagreement signal is weak and lists threats to validity. Section VII concludes.

---

## II. Related Work

RetailSense draws on eight lines of work; the last (Section II-H) contains the base paper. For
each, this section states what it established and how this paper relates to it.

### A. Forecast Combination

Combining forecasts is an old and well-supported idea. Bates and Granger [6] showed that a
combination weighted by past error can have a lower mean squared error than any of its inputs.
Timmermann [7] explains why combinations tend to beat individual forecasts on average and how
combination weights are estimated, and Wang et al. [8] review simple, time-varying, nonlinear and
cross-learning combinations of point and probabilistic forecasts. The RetailSense fast path is a
combination of this kind: agent forecasts are weighted by a confidence derived from trailing
error. This paper differs in what it asks of the combination: the spread of the agent forecasts
around the combination is also used to decide whether further processing is worthwhile.
Timmermann [7] also notes that combining forecasts is mainly useful when the underlying
information sets cannot be pooled into one model. On the India test split, the multi-agent systems
do not beat one LightGBM trained on every feature (h = 1 WAPE 0.3775 vs 0.3688).

### B. Disagreement as an Uncertainty Signal

Zarnowitz and Lambros [9] found that disagreement among forecasters is positively correlated with
uncertainty but understates it. In weather prediction, the spread-skill literature shows that the
correlation between ensemble spread and forecast error can be modest even for a perfect ensemble,
and that spread is informative mainly when it is extreme [10]. These results motivate treating
D as a proxy for risk rather than a measure of it, and they set a realistic expectation for H1.
They also suggest why a trigger at the 80th percentile of validation D may not select the cases
that deliberation can improve. RetailSense differs from both settings in that its agents are not
independent forecasters or perturbed ensemble members. The context agents are residual adjusters
on a shared baseline, so their spread measures the size of their adjustments. Pairwise agent
forecast correlations exceed the 0.95 design gate (0.994–0.997 on Rohlik validation, and India
also fails), and D adds almost nothing to the Historical band width as a predictor of large errors
(Section V).

### C. Multi-Agent LLM Debate, Self-Refinement and Social Compliance

Multi-agent debate, in which several LLM instances exchange answers over rounds, has been reported
to improve reasoning and factuality [3]. Self-Refine shows that a critique-then-revise loop can
improve outputs without retraining [11]. RetailSense borrows the structure of both: a critic
explains the conflict and each agent receives one bounded revision. Two findings temper the
expected benefit. Without external feedback, LLM self-correction often fails and can make answers
worse [21]. RLHF-trained assistants also tend to change their answers under social pressure, a
behaviour described as sycophancy [22]. A deliberation step that makes agents move whenever they
are challenged would therefore measure compliance rather than evidence. For this reason the design
tracks the hold rate, the share of revisions in which an agent keeps its position, as a gate, and
presents holding as a normal answer. It also constrains the deliberation output: the LLM can only
rescale an agent's weight within [0.25, 2.0] and move its value inside the agent's own band. The
task differs from the debate literature, which targets question answering and reasoning; here
deliberation acts on numeric forecasts produced by other models.

### D. LLMs for Time-Series Forecasting

A separate line of work uses LLMs as forecasters. Gruver et al. [14] show that LLMs can forecast
numeric series zero-shot when the series is encoded as text, and Time-LLM [15] repurposes a frozen
LLM as a forecaster by reprogramming its inputs. Tan et al. [4] question the benefit: removing the
LLM from such methods, or replacing it with a simple attention layer, does not hurt accuracy.
RetailSense takes the opposite design choice to the LLM-as-forecaster work. The LLM never produces
the demand number or the order quantity; it only critiques forecasts made by gradient-boosted
models and proposes bounded re-weightings and in-band revisions, which code clips before computing
the final value. The small or null LLM gains observed here are consistent with the scepticism of
[4].

### E. Retail Demand Forecasting and Gradient Boosting

In the M5 competition, a large retail forecasting benchmark, LightGBM-based methods dominated [2].
LightGBM is a gradient-boosted decision tree implementation designed for speed on large tabular
data [1]. These results justify LightGBM as the Historical agent and as the model behind every
context agent, and they justify a single all-features LightGBM as a strong baseline. The
evaluation uses two datasets: a documented simulator of an Indian kirana store with real weather
and a real festival calendar as demand drivers, which is not a real store's sales, and real daily
sales from the Czech online grocer Rohlik [20]. Neither is part of the M5 benchmark.

### F. Forecast Accuracy versus Inventory Performance

Standard accuracy measures do not translate directly into inventory efficiency [18], and the
mapping from forecast error to inventory outcome is nonlinear, so the most accurate model is not
always the best for inventory [19]. For this reason every system in this paper, baselines
included, passes through the same periodic-review (s, S) simulator, with safety stock set by the
standard textbook formula [16]. H5 is tested on simulated inventory cost and service, not on WAPE
alone. The results illustrate the point made in [19]: on the India test split, ETS has the lowest
total inventory cost although it is not the most accurate system.

### G. Cost-Aware and Selective LLM Use

FrugalGPT studies cascades and selective invocation of LLMs to trade cost against accuracy [12],
and RouteLLM learns routers that decide per query whether to call the more expensive model [13].
Triggering deliberation only when D ≥ τ is a simple, unlearned instance of the same idea: the
expensive step runs only on cases whose D reaches a fixed threshold, calibrated to trigger 20% of
validation cases. This paper differs in how the router is evaluated. It is compared against a
random trigger at the same rate, and a cost-accuracy curve is traced over trigger rates, which
separates the value of the signal from the value of deliberating more cases.

### H. Multi-Agent Systems for Integrated Forecasting and Inventory (Base Paper)

MARIOD [5] treats forecasting and inventory as one problem. A transformer with cross-modal
attention forecasts demand from sales, promotions and sensor streams, and hierarchical agents at
store, distribution-centre and corporate level learn order and reallocation policies with a
hierarchical PPO variant, trained jointly with the forecaster. On Dunnhumby, Favorita and UCI
Online Retail it reports a MAPE of 15.6% against 17.2% for the Temporal Fusion Transformer, and a
stockout rate of 3.5% against 5.2% for a hierarchical MARL baseline.

RetailSense keeps the base paper's goal, specialised agents whose forecasts drive restocking
decisions, and differs in four ways:

1. **Agent split.** MARIOD's agents are supply-chain levels acting on a shared learned
   representation. RetailSense's agents are evidence sources (history, calendar, weather,
   interest), and each sees only its own signal, so their disagreement can be read as conflicting
   evidence, the quantity this paper tests.
2. **Coordination.** MARIOD coordinates through learned graph-attention messages. RetailSense
   coordinates through an explicit disagreement score and, only when D ≥ τ, one bounded LLM
   critique-and-revision round. Every revision can be inspected and is clipped by code.
3. **Decision rule.** MARIOD learns the order policy end to end. RetailSense computes orders with a
   standard (s, S) policy [16], so differences in inventory outcome come only from the forecasts,
   and every baseline goes through the same simulator.
4. **Evaluation protocol.** MARIOD uses a 70/15/15 split, fills missing sales with zero and reports
   single point estimates. RetailSense leaves missing sales missing, freezes τ on validation before
   one locked test run, reports bootstrap intervals and significance tests, and reports the cases
   where it does not beat its baselines.

The two papers therefore answer different questions. MARIOD asks how far a jointly trained deep
system can push accuracy and service level. RetailSense asks whether disagreement between simple
specialised agents indicates *when* extra reasoning is worth its cost.

---

## III. Method

### A. Agents and Output Schema

Each agent returns the same schema-validated object: a point forecast for horizons h = 1, …, 7
days, a 10–90% band, a confidence c ∈ [0, 1], key factors and stated limitations. The
orchestrator re-validates each output. An invalid or abstaining agent is excluded from the
consensus and flagged; it is never converted to a zero forecast.

- **Historical agent.** LightGBM with a Tweedie objective gives the point forecast, and two
  quantile models (α = 0.1, 0.9) give the band. Its evidence is lagged sales (lags h, 7, 14 and
  28; rolling mean and standard deviation over 7, 28 and 56 days; share of zero-sales days), price
  (level, one-week change, ratio to the 52-week maximum) and item identity. There is one direct
  model per horizon h, and every sales and price feature is lagged at least h days. If the
  quantile models cross the point forecast, the band is widened to include it.
- **Context agents (Calendar, Weather, Trend).** These are residual adjusters. Each agent
  forecasts b + g_i(b, x_i), where b is the Historical agent's forecast and g_i is a LightGBM
  regressor that sees b and the agent's own evidence x_i and nothing else. g_i is trained on
  *out-of-fold* baseline residuals from five expanding time folds, pooled over horizons, so it
  never learns from in-sample baseline errors. The agent's band is the baseline band shifted by
  its adjustment.
  - Calendar evidence: weekday, month, week of year, day of month, event type, and days to and
    since the nearest festival or holiday.
  - Weather evidence: target-day maximum and minimum temperature and precipitation, their
    anomalies against a same-season climatology built from previous years only, hot-day and
    rainy-day flags, and 3-day changes.
  - Trend evidence: scale-invariant weekly search-interest features.
- **Confidence.** For each agent, item and origin, c_i = 1 / (1 + WAPE_i), where WAPE_i is computed
  over the agent's out-of-sample forecasts for the trailing 28 revealed days. Items with fewer
  than 14 such days fall back to the pooled department WAPE.

### B. Disagreement Score

At the disagreement horizon h = 1, with weights w_i = c_i / Σ_j c_j and consensus
ȳ = Σ_i w_i y_i:

    D = sqrt( Σ_i w_i (y_i − ȳ)² ) / max(ȳ, 1)                                    (1)

This is a confidence-weighted coefficient of variation. The one-unit floor keeps slow-moving items
from producing extreme values. D is unchanged when every forecast is scaled by the same constant,
and it is 0 when all forecasts agree; both properties are unit-tested. Two variants serve as
sensitivity checks: D_range = (max_i y_i − min_i y_i) / max(ȳ, 1), and the unnormalised weighted
standard deviation D_std.

### C. Calibration and Routing

τ is the 80th percentile of D over the validation split, so about 20% of validation forecasts are
deliberated. It is computed once and written into the configuration before any test data are read,
and test results never feed back into it. If D < τ, the forecast is the confidence-weighted
consensus (the *fast path*). If D ≥ τ, the case is deliberated.

### D. Critic and Bounded Revision

Deliberation is one round, implemented as a LangGraph state machine (load context → agents →
disagreement → route → consensus or critic → revise → inventory).

- **Critic.** The critic receives each agent's h = 1 forecast, band, confidence, evidence
  description, key factors and limitations, plus the item's last 14 days of revealed sales. It
  returns a JSON report: the main conflict, the conflicting agents, a strong/moderate/weak
  judgement of each agent's evidence, missing or stale evidence, and one question per agent. The
  report schema has no numeric field, so the critic cannot emit a demand number, and τ is not in
  the prompt.
- **Revision.** Each agent's revision call receives its own view, the critic report and its
  question. It returns a confidence multiplier m_i, a revised value y'_i, a rationale, and
  citations that must refer to evidence in the critic report. Holding position is presented as a
  normal answer.
- **Arithmetic in code.** Code clips m_i to [0.25, 2.0] and y'_i to the agent's own [lower, upper]
  band, then computes the final forecast Σ_i w'_i y'_i with w'_i ∝ c_i m_i. Horizons h = 2, …, 7
  keep the original agent forecasts with the new weights w'_i, and bands are the w'-weighted agent
  bands.
- **Failure handling.** Invalid output receives one corrective retry. If it is still invalid, the
  case falls back to the fast path and the failure is logged. Calls use a pinned model at
  temperature 0 and are cached on disk, keyed by model, prompt version, prompt text and input.

### E. Inventory Policy

A periodic-review (s, S) policy [16] turns each system's forecast into an action. It uses lead
time L = 3 days, coverage R = 7 days, service level 95% (z = 1.65) and case pack 6. The band gives
σ_h = (upper_h − lower_h) / 2.563. Then

    s = Σ_{h≤L} f_h + z · sqrt( Σ_{h≤L} σ_h² )                                     (2)
    S = Σ_{h≤L+R} f_h + z · sqrt( Σ_{h≤L+R} σ_h² )                                 (3)

Forecasts beyond day 7 repeat day 7. With inventory position IP = on hand + on order, the policy
orders S − IP rounded up to whole case packs if IP ≤ s (RESTOCK), flags REDUCE if IP exceeds S plus
one week of forecast demand, and otherwise HOLDs. The policy is a pure function of numbers. No LLM
output reaches the order quantity except through the code-computed forecast, and a test asserts
that no module in the inventory package imports the LLM layer.

---

## IV. Experimental Setup

### A. Data

**India (simulated).** The primary data describe a kirana chain in India. Store MUM_1 (Mumbai) and
all 60 of its items are used, ranked by total sales over the training days only, so test sales
cannot influence which items are chosen. Daily sales, prices, promotions and a weekly
search-interest index are generated by a documented simulator with a fixed seed; weather from
Open-Meteo [17] and the festival calendar (Diwali, Holi, Eid, Dussehra and national holidays) are
real. Demand is the product of base level, weekday, growth, weather, festival, payday, promotion
and interest-shock multipliers, with gamma-Poisson noise and occasional stock-outs. Every
multiplier is logged per row, so the evidence source that actually drove demand on each day is
known. The data are replayed; there is no live point-of-sale feed.

**Rohlik (real).** The second dataset is the Kaggle Rohlik Sales Forecasting Challenge v2 [20],
real daily sales of a Czech online grocer. Warehouse Prague_1 is used from 1 Aug 2020 to 2 Jun 2024
(d_1–d_1402). Kaggle's own test period has no public sales, so only the training file is used,
with the same split lengths as India. A day without a sales record is left missing, never zero.
Items must have a sales record on every day from d_1000 (27 Apr 2023); 63 of 857 qualify, and the
top 60 by training volume are used. Weather is taken from Open-Meteo for Prague (50.08° N,
14.44° E), and the calendar uses Czech public holidays. No search-interest data are available, so
the Trend agent is omitted, and the ETS fit window is 91 days so that its input never contains a
gap.

TABLE I
DATA SPLITS

| Dataset | Split | Days | Dates | Use |
|---|---|---|---|---|
| India | Train | d_1–d_1799 | 5 Jan 2019 – 8 Dec 2023 | fit every model (with internal expanding folds) |
| India | Validation | d_1800–d_1885 | 9 Dec 2023 – 3 Mar 2024 | early stopping, confidence, calibration of τ, all design decisions |
| India | Test | d_1886–d_1941 | 4 Mar – 28 Apr 2024 | final numbers, one run |
| Rohlik | Train | d_1–d_1260 | 1 Aug 2020 – 12 Jan 2024 | as India |
| Rohlik | Validation | d_1261–d_1346 | 13 Jan – 7 Apr 2024 | as India |
| Rohlik | Test | d_1347–d_1402 | 8 Apr – 2 Jun 2024 | final numbers, one run |

The splits are chronological. Every model is fitted on train. Validation and test forecasts are
made from rolling daily origins with the same models, and each origin uses only data revealed up
to that origin. A rolling-origin backtest repeats the evaluation on three 56-day windows before
validation (India: d_1632–d_1687, d_1688–d_1743, d_1744–d_1799); for each window, every model is
refitted on the days before it.

### B. Signals and Assumptions

- **Leakage control.** Every feature row carries an `available_at` date, and tests assert that no
  feature is available after the prediction time. Actual demand is reachable only from evaluation
  code.
- **Weather.** The frozen runs use the Open-Meteo historical archive value for the target day,
  i.e. weather as a perfect forecast. A post-hoc check against Open-Meteo's archive of *issued*
  forecasts bounds the effect (Section VI-F), and a noise ablation tests sensitivity (Section V-G).
- **Interest signals.** The weekly interest index comes in Google Trends format, in two overlapping
  windows each scaled to its own maximum. A week counts as available only after it closes, and only
  scale-invariant features are used. It is a demand *proxy*.
- **Missing signals.** A missing signal is left missing, never zero-filled or forward-filled.

### C. Language Model

All critic and revision calls use Llama 3.1 8B [23], served locally through Ollama at
temperature 0 with a single frozen prompt version (v1) on both datasets. An earlier pilot with
gemini-2.5-flash-lite was dropped after a 100% hold rate.

### D. Systems Compared

TABLE II
SYSTEMS COMPARED

| # | System | Isolates |
|---|---|---|
| 1 | Seasonal naive (same weekday last week) | floor |
| 2 | ETS(A,N,A) per item, weekly season | classical baseline |
| 3 | One LightGBM with all features | does splitting evidence into agents beat one model? |
| 4 | Historical agent only | value of context |
| 5 | Multi-agent simple average | value of combining |
| 6 | Multi-agent confidence-weighted, no deliberation | value of confidence weighting |
| 7 | Random-trigger deliberation at RetailSense's rate | value of D as the trigger |
| 8 | Always-on deliberation | upper bound; cost reference |
| 9 | RetailSense (D ≥ τ triggers deliberation) | proposed method |

Systems 7–9 share one set of deliberation outputs per forecast point. Because τ is not in any
prompt, the three arms are paired subsets of the same LLM answers.

### E. Metrics and Tests

- **Forecast metrics.** MAE, RMSE, WAPE (primary) and sMAPE at h = 1 and pooled over h = 1–7, plus
  10–90% band coverage, each with 95% confidence intervals from a bootstrap over items.
- **Mechanism metrics.** H1: Spearman correlation between D and relative absolute error, AUROC of
  D for the worst 20% of errors, and a decile plot. H2: the share of triggered cases improved or
  worsened by deliberation, with a Wilcoxon signed-rank test. Cost: LLM calls, tokens and latency
  per 1,000 forecasts, and a cost-accuracy curve over trigger rates from 0% to 100%.
- **Inventory.** Every system runs through the same daily simulator (Section III-E) with holding
  cost 25% of price per year, order cost $5 and lost-sales cost equal to 1× price. Reported:
  stockout rate, fill rate, overstock and total cost.
- **Statistics.** RetailSense is compared with each other system using a Diebold-Mariano (DM) test
  [24] on daily mean absolute error (plus per-item tests), a paired bootstrap over items (1,000
  resamples) for WAPE and cost differences, and a Wilcoxon signed-rank test on per-item WAPE.
- **Ablations.** Calibrated vs fixed τ; D vs D_range vs unnormalised D; weights-only vs
  weights-and-values revision; dropping each context agent; noisy weather forecasts; lead time 1, 3
  and 7 days.
- **Software and tracking.** All runs are logged to MLflow with the configuration hash and git
  commit, per-case outputs are stored in SQLite, and the unit and leakage tests run on every change.

---

## V. Results

All India numbers come from the single frozen test run (MLflow run
`14e1f0e0999f451897d3e61011a56d49`, configuration hash `acbb90af8345`). τ = 0.076898 and prompt v1
were frozen on validation before the test split was read. Differences are RetailSense minus the
comparison system, with 95% paired bootstrap intervals over items.

### A. Forecast Accuracy

The single LightGBM with all features is the most accurate system on test, with an h = 1 WAPE of
0.3688 against 0.3775 for RetailSense (Table III). Its advantage is significant (difference
+0.0087 [0.0015, 0.0161], DM p = 0.003), and it is also the most accurate system in all three
backtest windows. RetailSense is significantly more accurate than ETS (−0.0129 [−0.0232, −0.0042],
DM p = 0.005) and than seasonal naive. Against the Historical agent alone the difference is not
significant (−0.0018 [−0.0036, 0.0000], DM p = 0.22). All multi-agent systems (5–9) lie within
0.0005 WAPE of each other. Splitting the evidence into agents therefore costs accuracy relative to
one model that sees every feature.

TABLE III
INDIA TEST RESULTS

| # | System | h = 1 WAPE [95% CI] | h = 1–7 WAPE | Total cost |
|---|---|---|---|---|
| 1 | Seasonal naive | 0.5530 [0.5261, 0.5860] | 0.5488 | $199,786 |
| 2 | ETS | 0.3904 [0.3712, 0.4121] | 0.4030 | **$138,621** |
| 3 | LightGBM, all features | **0.3688** [0.3555, 0.3835] | **0.3782** | $183,250 |
| 4 | Historical agent | 0.3793 [0.3664, 0.3954] | 0.3902 | $232,910 |
| 5 | Multi-agent mean | 0.3778 [0.3646, 0.3931] | 0.3886 | $203,357 |
| 6 | Multi-agent, confidence-weighted | 0.3777 [0.3646, 0.3931] | 0.3885 | $205,862 |
| 7 | Random trigger | 0.3778 [0.3647, 0.3931] | 0.3885 | $206,393 |
| 8 | Always-on deliberation | 0.3773 [0.3645, 0.3926] | 0.3884 | $199,868 |
| 9 | RetailSense | 0.3775 [0.3643, 0.3929] | 0.3885 | $206,313 |

### B. H1: Does Disagreement Predict Error?

D is weakly associated with error: Spearman ρ = 0.072 [0.039, 0.108] and AUROC = 0.539
[0.505, 0.565] for flagging the worst 20% of errors. Both intervals exclude chance, but the effect
is small, and in the backtests the AUROC is 0.526, 0.512 and 0.516. On validation, D added nothing
beyond the Historical agent's band width and item volume. H1 is supported only weakly.

### C. H2: Does Deliberation Help the Triggered Cases?

Of the 892 triggered test points (trigger rate 26.5%), 30.3% improved and 24.6% worsened after
deliberation, and the mean absolute error fell from 12.661 to 12.633 (Wilcoxon p = 0.007). No case
fell back to the fast path, and agents held their position in 58.7% of revisions, so the
revisions are not simple compliance with the critic. The same comparison on validation was not
significant (p = 0.32). H2 is supported on test, with a small effect that was not seen on
validation.

### D. H3: Selective vs Always-On Deliberation

RetailSense and always-on deliberation are indistinguishable in accuracy (+0.0002 [−0.0003,
0.0008], DM p = 0.58), but RetailSense uses 1,331 rather than 5,018 LLM calls per 1,000 forecasts
(−73%) and 6.9 s rather than 20.7 s mean latency per forecast (latency includes queueing behind
four parallel requests on one GPU). Always-on has a lower inventory cost ($199,868 vs $206,313),
but the per-item difference is not significant (+$107 [−6, 266]). H3 is supported for accuracy
and LLM cost.

### E. H4: Disagreement vs a Random Trigger

Triggering on D is not reliably better than triggering at random at the same rate. On test the
WAPE difference is −0.0003 [−0.0006, −0.00001], but DM p = 0.08 and the item-level Wilcoxon
p = 0.07, and the inventory cost difference is −$1.3 per item [−23, 20]. In all three backtests the
random trigger has the lower WAPE. Along the cost-accuracy curve, selective triggering is about two
random-draw standard deviations better at trigger rates of 10–20% but worse at 40–80%. H4 is not
supported.

### F. H5: From Forecasts to Inventory

ETS has the lowest total test cost ($138,621), followed by LightGBM with all features ($183,250);
RetailSense costs $206,313. ETS is significantly cheaper than RetailSense (+$1,128 per item
[220, 2,211]), even though RetailSense is more accurate than ETS. The cost ranking differs from the
WAPE ranking because the policy also uses the bands, and lost-sales cost drives the differences
($63,722 for ETS vs $136,574 for RetailSense). With a 7-day lead time, seasonal naive is cheapest
and RetailSense is the cheapest multi-agent system. H5 is not supported for RetailSense.

### G. Robustness and Ablations

Across the three backtest windows, LightGBM with all features is always the most accurate; the
cheapest system is ETS in one window and a multi-agent consensus in the other two. Calibrating τ
matters (a fixed τ of 0.3 triggers on 0.3% of points), normalising D matters (the unnormalised
spread has AUROC 0.465), and revised values add a little over re-weighting alone (WAPE 0.37752 vs
0.37768). Removing the Weather or Trend agent improves WAPE (0.37552 and 0.37650 vs 0.37772
without deliberation), while removing Calendar worsens it (0.38199). Adding noise to the weather
inputs raises the trigger rate from 26.5% to 36.5% and changes a third of the triggered set
(Jaccard 0.64).

### H. Case Study

The case was chosen by a fixed rule, the highest-D triggered point: item sevai_2 on d_1907 with
D = 0.602. The consensus was 3.35 units against an actual 2.0. The critic described the conflict
only in general terms (the agents use different types of evidence), the revisions left the
forecast at 3.35, and the policy acted on that number. With the 8B local model, the critic rarely
names which specific evidence is wrong, which is consistent with the small effects in
Sections V-C to V-E.

### I. Real-Data Replication: Rohlik Test Split

The same protocol was run once on the Rohlik test split (MLflow run
`e571b9d28541434984de0c346044a3c5`, configuration hash `4d5d100636f4`). τ = 0.065352 and prompt v1
were frozen on Rohlik validation before the run. Rohlik uses three agents (no Trend agent).

TABLE IV
ROHLIK TEST RESULTS

| # | System | h = 1 WAPE [95% CI] | h = 1–7 WAPE | Total cost |
|---|---|---|---|---|
| 1 | Seasonal naive | 0.2582 [0.1857, 0.3383] | 0.2522 | $509,438 |
| 2 | ETS | **0.1504** [0.1308, 0.1736] | 0.1892 | $262,890 |
| 3 | LightGBM, all features | 0.1665 [0.1376, 0.1959] | **0.1858** | **$133,467** |
| 4 | Historical agent | 0.1760 [0.1463, 0.2043] | 0.1975 | $314,936 |
| 6 | Multi-agent, confidence-weighted | 0.1746 [0.1502, 0.1974] | 0.1993 | $287,802 |
| 7 | Random trigger | 0.1747 [0.1509, 0.1971] | 0.1993 | $289,319 |
| 8 | Always-on deliberation | 0.1741 [0.1505, 0.1961] | 0.1994 | $278,104 |
| 9 | RetailSense | 0.1745 [0.1504, 0.1970] | 0.1993 | $287,344 |

RetailSense is significantly less accurate than ETS (WAPE difference +0.024 [0.007, 0.037],
DM p < 0.001) and not significantly different from LightGBM (+0.008, DM p = 0.17).

- **H1.** Spearman ρ = 0.119 [0.063, 0.176]; AUROC for the worst 20% = 0.579 [0.538, 0.619]. The
  association is weak but significant, and stronger in the three backtests (ρ 0.28–0.44, AUROC
  0.69–0.78).
- **H2.** 576 triggered cases (17.2%): 27.6% improved, 26.6% worsened, Wilcoxon p = 1.00. Hold
  rate 53.3%; fallback rate 0.17%.
- **H3.** 696 vs 4,051 LLM calls per 1,000 forecasts (−83%); WAPE vs always-on +0.0004, DM
  p = 0.59.
- **H4.** Selective minus random WAPE −0.0002 [−0.0010, 0.0006], DM p = 0.57.
- **H5.** LightGBM has the lowest cost; RetailSense costs more than both LightGBM and ETS.

---

## VI. Discussion

### A. Why Disagreement Carries Little Signal

Disagreement among forecasters is known to be only a partial proxy for uncertainty [9], and even a
well-built ensemble can show a modest spread-skill relationship [10]. RetailSense adds a structural
reason. The context agents (three on India, two on Rohlik) are residual adjusters: each forecasts
the Historical baseline plus a small learned adjustment. Their forecasts are therefore highly
correlated with the baseline and with each other. On Rohlik validation the pairwise agent forecast
correlation is 0.994–0.997, and India also fails the planned 0.95 gate. D then measures the size of
the adjustments rather than a genuine conflict between independent views of demand.

The adjustments themselves are weak because the context agents do not see item identity. They
cannot route a signal to the items it affects, so they apply roughly the same adjustment to every
item. Because the India data are simulated, the planted effects are known and this can be checked
directly. On validation the Weather agent added about +2 units to soap and atta, whose true
weather effect is 0, and +2.3 to tea, whose true effect is −1.4. The Trend agent added a flat +1.3
where the true ice-cream interest shock was +11.7. The ablations agree: removing the Weather or
Trend agent improves test WAPE, while only Calendar helps.

### B. Why the Critic Helped Little

Self-critique and multi-agent debate can improve LLM reasoning [3], [11], but without reliable
external feedback self-correction often fails and can make answers worse [21]. The critic here, an
8B local model [23], usually described conflicts in general terms, for example that the agents use
different types of evidence, and rarely named which evidence was wrong (Section V-H). On India
test, 30.3% of 892 triggered points improved and 24.6% worsened (Wilcoxon p = 0.007), but this did
not replicate on India validation (p = 0.32). On Rohlik validation, 24.4% improved and 29.4%
worsened (MAE 51.21 to 51.01, p = 0.26). Hold rates of 58.7% (India test) and 55.3% (Rohlik
validation) indicate that agents were not simply complying with the critic, the failure mode that
sycophancy research warns about [22], so the prompt does not appear coercive. Whether the limit is
the critic's strength or the weak, shared evidence of the agents cannot be separated with a single
8B model. This fits wider evidence that LLM components often add little to numeric forecasting
accuracy [4]. RetailSense deliberately keeps the LLM away from the demand number itself, unlike
LLM-as-forecaster designs [14], [15], so a weak critic can only make small, bounded changes.

### C. What Differs on Real Data

On Rohlik validation, D is a stronger risk signal than on India: Spearman ρ = 0.099
[0.053, 0.135] and AUROC = 0.570 [0.535, 0.595]. The top decile of D has mean relative error 0.4366
against 0.2143 in the bottom decile. Even so, D adds almost nothing beyond the Historical band
width (AUROC 0.7057 alone, 0.7058 with D). Within volume tertiles, D is close to uninformative
(ρ −0.028 to 0.049; AUROC 0.49–0.53), and adding D to band width plus volume leaves AUROC at
0.7124. Most of D's pooled signal therefore reflects item volume rather than real conflict between
agents.

On a 1,500-case stratified validation sample, always-on deliberation lowered WAPE from 0.1619 to
0.1602 (about 1%); this gap was not tested for significance. Selective triggering did not capture
it: selective reached 0.1618 against 0.1615 (sd 0.0006) for a random trigger at the same 20% rate.
On the cost-accuracy curve, selective is worse than the random mean at every rate from 0.05 to
0.30, and better mainly at rates of 0.65–0.85 (for example 0.1598 vs 0.1607, sd 0.0006, at 0.75).
The Rohlik test split confirms the picture. D was slightly more informative than on India
(ρ = 0.119, AUROC = 0.579) and clearly more so in the three backtest windows (ρ 0.28–0.44, AUROC
0.69–0.78), yet triggered cases improved and worsened about equally (27.6% vs 26.6%, p = 1.00),
selective triggering matched a random trigger (DM p = 0.57), and always-on was no longer better
than selective (WAPE 0.1741 vs 0.1745, DM p = 0.59). Disagreement marks riskier forecasts somewhat
better on real data, but the critic does not repair those forecasts selectively.

### D. Why Accuracy and Cost Rankings Differ

The ranking by WAPE does not carry over to inventory cost. ETS is less accurate than RetailSense on
India test but has the lowest total cost ($138,621 against $206,313). The (s, S) policy uses the
bands as well as the point forecast [16], and lost-sales cost dominates the differences ($63,722
for ETS against $136,574 for RetailSense). This agrees with earlier work showing that accuracy
measures do not map directly onto inventory performance [18] and that the most accurate model is
not always the best for inventory [19]. On Rohlik validation the cheapest system was the
Historical agent ($193,385), while the most accurate, LightGBM with all features, cost $422,135.
Evaluating forecasts through a common simulator is therefore necessary, not optional.

### E. Practical Implications

Forecast combination is a strong default [6]–[8], yet a single LightGBM with all features remains
the most accurate system on India test, so splitting evidence across agents costs accuracy here.
The clearest practical result is cost: on India test, selective deliberation showed no detectable
accuracy difference from always-on with 73% fewer LLM calls, in the spirit of cascades and routers
that invoke an expensive model only when needed [12], [13]. The saving is only worth having if
deliberation helps, however, and the random-trigger control shows no reliable advantage for D over
chance. Selective deliberation could pay off when the agents hold genuinely independent evidence,
so that D measures real conflict; when context agents can target specific items; and when the
critic is strong enough to identify which evidence is wrong. On Rohlik validation, where always-on
helped, a trigger that selects the cases deliberation can fix, rather than the cases that look
risky, would be required.

### F. Threats to Validity

- **Failed independence gate.** Pairwise agent forecast correlation exceeds the planned 0.95 gate
  on both datasets (Rohlik 0.994–0.997), so D largely measures adjustment size. This is a design
  failure, not a minor caveat. A post-hoc diagnostic shows the pooled correlation is dominated by
  between-item sales level: the adjustments themselves correlate only 0.14–0.57 on India and 0.35
  on Rohlik, while the India Trend agent is almost a copy of the Historical agent (within-item
  correlation 1.000). The gate definition was not changed after the fact.
- **Simulated India sales.** The India sales are simulated with known drivers, so India results
  describe method behaviour, not a real store. Rohlik provides real sales, and its test results
  agree with India on every hypothesis except H2, where the small India effect did not replicate.
- **Single seed in the frozen run.** The test runs used one LightGBM seed (42). A post-hoc check
  refit systems 1–6 on validation under four more seeds (without the LLM, so systems 7–9 were not
  rerun). The seed standard deviation of h = 1 WAPE is 0.0004–0.0006 on India and 0.0007–0.0023 on
  Rohlik, so gaps between systems 5–9 (below 0.0005) are within seed noise. The all-features
  LightGBM beat the confidence-weighted consensus under every seed (gap 0.0097 ± 0.0006 India,
  0.0098 ± 0.0026 Rohlik). H1 is seed-sensitive on India: ρ ranged 0.027–0.064 (mean 0.044) and
  AUROC 0.505–0.535, and seed 42 gave the highest value of both, so the reported India H1 is at the
  favourable end. On Rohlik, ρ was 0.085–0.104.
- **One LLM.** Only an 8B local model was tested. A stronger critic could change H2–H4.
- **Weather inputs.** The study feeds the archive value for the target day, i.e. weather as a
  perfect forecast. A post-hoc comparison with Open-Meteo's archive of *issued* short-lead
  forecasts bounds the effect. For Mumbai the two archives return identical values on every day of
  the last training year, validation and test, so the India inputs equal forecasts available at
  h = 1; because the India demand was generated from this same series, the Weather agent still sees
  its true driver without error, a property of the simulation. For Prague the issued forecasts
  differ (MAE 0.9 °C maximum, 1.1 °C minimum temperature, 0.6 mm precipitation on validation), yet
  replacing the validation inputs with them, models unchanged, moved h = 1 WAPE by −0.0001 for the
  Weather agent and −0.0004 for the all-features control. The archive holds the shortest-lead
  forecast only, so for h > 1 the real penalty is larger; the noise ablation covers that direction.
- **Censored demand.** Sales are capped on stock-out days and treated as demand: 1.2% of India test
  item-days and 5.0% of Rohlik test item-days. Scoring without those days changes h = 1 WAPE by at
  most 0.016 and leaves the test ranking of systems 1–6 unchanged on both datasets; on Rohlik
  validation, ETS and the Historical agent, 0.0007 apart, swap places.
- **Fixed inventory parameters.** L = 3, R = 7, z = 1.65 and the costs are illustrative, and
  z = 1.65 is not cost-optimal when a lost sale costs far more than a day of holding. In validation
  sensitivity runs a 0.99 service level lowers every system's India cost by 26–33%, but ETS remains
  the cheapest, so the cost ranking is not an artefact of z.
- **Context agents cannot target items.** A residual agent sees the baseline and its own evidence
  but no item identity, so it applies about the same adjustment to every item. A post-hoc
  validation run gave the context agents item identity, nothing else changed: the consensus
  improved slightly (India WAPE 0.3770 to 0.3751, Rohlik 0.1689 to 0.1684), but the gate still
  failed (0.984–0.9996 and 0.988–0.996), the Rohlik agents individually got worse, and D became a
  weaker risk signal (ρ 0.064 to 0.034 India, 0.099 to 0.078 Rohlik). Confidence weights are
  uninformative (Spearman with error −0.09 to −0.12; the most confident agent is the most accurate
  on 33% of points).
- **Fixed trigger rule.** A learned trigger modelling the deliberation gain from D, D_range, the
  Historical band width and volume (time-blocked cross-validation on the 1,500 stored validation
  deliberations, no new LLM calls) was no better than random at any trigger rate on either dataset
  (Spearman with the realised gain 0.023 and −0.022), although a perfect trigger would lower WAPE
  by about 1% (India) and 4% (Rohlik).
- **Generated interest signal.** The India search-interest index is produced by the generator and
  is built to lead demand, so the Trend agent is tested on a signal designed to be useful.
- **Rohlik without a Trend agent.** Items were chosen by having a sales record on every day from
  d_1000, which uses presence (not sales) in the evaluation period. Rohlik has no search-interest
  data, so it uses three agents. A post-hoc validation follow-up added a Trend agent on real Google
  Trends data for Czechia ("potraviny" and "rohlik", no LLM step): the Trend agent alone was less
  accurate than the Historical agent (h = 1 WAPE 0.1801 vs 0.1717), the consensus worsened (0.1689
  to 0.1710), and H1 moved from ρ = 0.099 to 0.116 with overlapping intervals. The frozen
  three-agent results stand.
- **Validation window.** The India validation window (December–March) has no rainy days and no
  major festival, so τ and all design choices were made on a period with little weather or festival
  evidence.
- **Missing comparisons.** There is no deep-learning or foundation-model baseline, and no ablation
  that revises without the critic's conflict analysis. A post-hoc validation check of other
  gradient-boosting learners (XGBoost, scikit-learn HGB) changed h = 1 WAPE by at most 3%, with no
  learner best on both datasets.
- **Validation-test mismatch.** The India test H2 effect (p = 0.007) was not significant on
  validation (p = 0.32), and H2 was not significant on Rohlik validation either.
- **Process.** The go/pivot decision required after the early H1 smoke test (ρ = 0.064 < 0.15) was
  recorded retrospectively, after both test runs. The Rohlik deliberation arms on validation used a
  1,500-case stratified sample.
- **Latency.** Latency includes queueing behind four parallel requests on one laptop GPU, and a
  replay from cache measures cache latency. Only accuracy, LLM calls and tokens are reproducible.

All post-hoc checks above are validation-only or read from stored outputs; none of them changed
a frozen configuration or re-read a test split for tuning.

---

## VII. Conclusion and Future Work

### A. Answers to the Hypotheses

RetailSense was evaluated on a simulated Indian kirana store and on real sales from the Rohlik
online grocer, with one frozen test run per dataset.

- **H1 (D predicts error): weakly supported.** India test: ρ = 0.072, AUROC = 0.539. Rohlik test:
  ρ = 0.119 [0.063, 0.176], AUROC = 0.579 [0.538, 0.619]. On validation, the Historical agent's
  band width and item volume predicted error better than D, and D added almost nothing beyond them
  (Rohlik AUROC 0.7124 with and without D).
- **H2 (deliberation helps triggered cases): supported on India test only, with a small,
  tentative effect.** 30.3% improved and 24.6% worsened (p = 0.007), an effect not seen on India
  validation (p = 0.32), Rohlik validation (p = 0.26) or Rohlik test (p = 1.00).
- **H3 (selective matches always-on at lower cost): supported for accuracy and LLM cost, not for
  inventory.** RetailSense used 73% (India) and 83% (Rohlik) fewer LLM calls with no significant
  WAPE difference (DM p = 0.58 and 0.59).
- **H4 (D beats a random trigger): not supported.** Selective minus random WAPE was −0.0003 on India
  (DM p = 0.08) and −0.0002 on Rohlik (DM p = 0.57); random had the lower WAPE in all three India
  backtests and on Rohlik validation.
- **H5 (better forecasts give better inventory): not supported for RetailSense.** ETS had the
  lowest India test cost ($138,621 vs $206,313) and LightGBM the lowest Rohlik test cost ($133,467
  vs $287,344).

A single all-features LightGBM on India (WAPE 0.3688) and ETS on Rohlik (0.150) were more accurate
than every multi-agent system.

### B. Main Lesson

Disagreement can signal risk only if the agents disagree for informative reasons. Here the context
agents are a shared baseline plus a small adjustment, and their forecasts are correlated above the
0.95 gate, so D measures the size of the adjustment rather than real conflict between evidence
sources. This fits earlier findings that disagreement is a weak proxy for uncertainty [9] and that
spread predicts skill mainly at the extremes [10]. The critic's mixed effect on triggered cases is
consistent with the limits of LLM self-correction [21], and the gap between forecast accuracy and
inventory cost supports evaluating forecasts through a simulator [18], [19].

### C. Future Work

1. **Independent context evidence.** Giving context agents item identity was tested post hoc and
   did not fix the independence gate. Agents built on evidence independent of the sales history,
   rather than residual adjusters on a shared baseline, are needed.
2. **Stronger critics.** All results use one 8B local model. A stronger LLM might name specific
   wrong evidence and change the H2–H4 results.
3. **Multi-seed runs including the LLM arms.** The post-hoc seed check covered systems 1–6 on
   validation only. Differences below about 0.001 WAPE between the deliberation arms need
   multi-seed evidence on fresh data and replication on further real datasets.
4. **Multi-day issued weather forecasts.** The available archive of issued forecasts covers only
   the shortest lead; testing the Weather agent at h > 1 requires forecasts issued at each lead.
5. **Learning when to deliberate.** A learned router on band width, volume and D was tested post
   hoc and was no better than random, because no pre-LLM signal predicted the gain. A router that
   also uses the critic's own output, in the spirit of cost-aware LLM cascades [12], [13], is the
   remaining option.

---

## References

[1] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 30, Long Beach, CA, USA, 2017, pp. 3146–3154.

[2] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions," *Int. J. Forecast.*, vol. 38, no. 4, pp. 1346–1364, 2022, doi: 10.1016/j.ijforecast.2021.11.013.

[3] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, "Improving factuality and reasoning in language models through multiagent debate," in *Proc. 41st Int. Conf. Mach. Learn. (ICML)*, Vienna, Austria, 2024, PMLR vol. 235, pp. 11733–11763.

[4] M. Tan, M. A. Merrill, V. Gupta, T. Althoff, and T. Hartvigsen, "Are language models actually useful for time series forecasting?" in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 37, Vancouver, BC, Canada, 2024.

[5] Y. Yang, M. Wang, J. Wang, P. Li, and M. Zhou, "Multi-agent deep reinforcement learning for integrated demand forecasting and inventory optimization in sensor-enabled retail supply chains," *Sensors*, vol. 25, no. 8, Art. no. 2428, 2025, doi: 10.3390/s25082428.

[6] J. M. Bates and C. W. J. Granger, "The combination of forecasts," *Oper. Res. Q.*, vol. 20, no. 4, pp. 451–468, 1969, doi: 10.1057/jors.1969.103.

[7] A. Timmermann, "Forecast combinations," in *Handbook of Economic Forecasting*, vol. 1, G. Elliott, C. W. J. Granger, and A. Timmermann, Eds. Amsterdam, The Netherlands: Elsevier, 2006, pp. 135–196.

[8] X. Wang, R. J. Hyndman, F. Li, and Y. Kang, "Forecast combinations: An over 50-year review," *Int. J. Forecast.*, vol. 39, no. 4, pp. 1518–1547, 2023, doi: 10.1016/j.ijforecast.2022.11.005.

[9] V. Zarnowitz and L. A. Lambros, "Consensus and uncertainty in economic prediction," *J. Polit. Econ.*, vol. 95, no. 3, pp. 591–621, 1987, doi: 10.1086/261473.

[10] J. S. Whitaker and A. F. Loughe, "The relationship between ensemble spread and ensemble mean skill," *Mon. Weather Rev.*, vol. 126, no. 12, pp. 3292–3302, 1998, doi: 10.1175/1520-0493(1998)126<3292:TRBESA>2.0.CO;2.

[11] A. Madaan et al., "Self-Refine: Iterative refinement with self-feedback," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 36, New Orleans, LA, USA, 2023.

[12] L. Chen, M. Zaharia, and J. Zou, "FrugalGPT: How to use large language models while reducing cost and improving performance," arXiv preprint arXiv:2305.05176, May 2023.

[13] I. Ong et al., "RouteLLM: Learning to route LLMs from preference data," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Singapore, 2025.

[14] N. Gruver, M. Finzi, S. Qiu, and A. G. Wilson, "Large language models are zero-shot time series forecasters," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 36, New Orleans, LA, USA, 2023.

[15] M. Jin et al., "Time-LLM: Time series forecasting by reprogramming large language models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Vienna, Austria, 2024.

[16] E. A. Silver, D. F. Pyke, and D. J. Thomas, *Inventory and Production Management in Supply Chains*, 4th ed. Boca Raton, FL, USA: CRC Press, 2017.

[17] P. Zippenfenig, "Open-Meteo.com Weather API," Zenodo, 2023. [Software]. doi: 10.5281/zenodo.7970649.

[18] A. A. Syntetos, K. Nikolopoulos, and J. E. Boylan, "Judging the judges through accuracy-implication metrics: The case of inventory forecasting," *Int. J. Forecast.*, vol. 26, no. 1, pp. 134–143, 2010, doi: 10.1016/j.ijforecast.2009.05.016.

[19] N. Kourentzes, J. R. Trapero, and D. K. Barrow, "Optimising forecasting models for inventory planning," *Int. J. Prod. Econ.*, vol. 225, Art. no. 107597, 2020, doi: 10.1016/j.ijpe.2019.107597.

[20] Rohlik Group. "Rohlik sales forecasting challenge v2." Kaggle. https://www.kaggle.com/competitions/rohlik-sales-forecasting-challenge-v2 (accessed Oct. 12, 2026).

[21] J. Huang et al., "Large language models cannot self-correct reasoning yet," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Vienna, Austria, 2024.

[22] M. Sharma et al., "Towards understanding sycophancy in language models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Vienna, Austria, 2024.

[23] A. Grattafiori et al., "The Llama 3 herd of models," arXiv preprint arXiv:2407.21783, Jul. 2024.

[24] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *J. Bus. Econ. Stat.*, vol. 13, no. 3, pp. 253–263, 1995.
