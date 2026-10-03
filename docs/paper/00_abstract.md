# When does disagreement-triggered LLM deliberation help demand forecasting? A controlled simulation and a real-data study

## Abstract

Calling a large language model (LLM) on every forecast is costly. We test whether disagreement
among specialised forecasting agents can decide when an LLM should deliberate. In RetailSense, a
LightGBM Historical agent and residual-adjusting Calendar, Weather and Trend agents forecast daily
item demand (Trend is omitted on Rohlik, which has no search-interest data). A confidence-weighted
disagreement score D triggers an LLM critic and one bounded revision round when D >= tau (the 80th
percentile of validation D). The LLM never writes the final forecast or the order quantity; it may
propose a confidence multiplier and a revised value inside each agent's band, which code clips and
combines. All systems share one (s, S) inventory simulator. The data are sales of 60 items in a
simulated Mumbai kirana store (generated with known drivers, real historical weather and a real
festival calendar), and real daily sales of 60 items from a Czech online grocer (Rohlik).

On India test data, D is weakly associated with error (Spearman rho = 0.072, AUROC = 0.539).
On India test, deliberation slightly lowered error on triggered cases (MAE 12.661 to 12.633,
Wilcoxon p = 0.007), but this did not replicate on India validation (p = 0.32) or Rohlik validation
(p = 0.26). On India test, selective deliberation matched always-on accuracy with 73% fewer LLM
calls, though always-on had lower inventory cost ($199,868 against $206,313, not significant).
Selective triggering is not reliably better than random triggering at the same rate: the DM test is
not significant (p = 0.08), and random triggering had lower WAPE in all three backtests. One
all-features LightGBM is more accurate (WAPE 0.3688 against 0.3775), and ETS has the lowest
inventory cost. On Rohlik validation, D is also weakly associated with error (rho = 0.099,
AUROC = 0.570) but adds nothing beyond the Historical band width and item volume. There, only
always-on deliberation helped (WAPE 0.1619 to 0.1602); selective (0.1618) was no better than random
(0.1615). Rohlik test: [ROHLIK TEST: pending final run]. Limitations: agent forecasts are highly
correlated (failing our 0.95 gate), so D largely reflects adjustment size; results use one model
seed, one 8B local LLM, and observed weather as a perfect forecast. Disagreement is a weak risk
signal, and an 8B LLM critic triggered by it did not beat chance.

## References

No references are cited in the abstract.
