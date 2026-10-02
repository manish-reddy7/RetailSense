# 5 Results

All numbers come from the single frozen test run (`reports/india/final/`, MLflow run
`14e1f0e0999f451897d3e61011a56d49`, config hash `acbb90af8345`) and are summarised in
`reports/RESULTS.md`. tau = 0.076898 and prompt v1 were frozen on validation before the test split
was read. Differences are RetailSense minus the comparison system, with 95% paired bootstrap
intervals over items.

## 5.1 Forecast accuracy

The single LightGBM with all features is the most accurate system on test, with an h = 1 WAPE of
0.3688 against 0.3775 for RetailSense (Table 1). Its advantage is significant (difference +0.0087
[0.0015, 0.0161], Diebold-Mariano p = 0.003) and it is also the most accurate system in all three
backtest windows. RetailSense is significantly more accurate than ETS (-0.0129 [-0.0232, -0.0042],
DM p = 0.005) and than seasonal naive. Against the Historical agent alone the difference is not
significant (-0.0018 [-0.0036, 0.0000], DM p = 0.22). All multi-agent systems (5-9) lie within
0.0005 WAPE of each other. Splitting the evidence into agents therefore costs accuracy relative to
one model that sees every feature.

Table 1 (source: `table_forecast_metrics.csv`): h = 1 WAPE seasonal naive 0.5530, ETS 0.3904,
LightGBM all features 0.3688, Historical 0.3793, multi-agent mean 0.3778, multi-agent confidence
0.3777, random trigger 0.3778, always-on 0.3773, RetailSense 0.3775.

## 5.2 H1: does disagreement predict error?

D is weakly associated with error: Spearman rho = 0.072 [0.039, 0.108] and AUROC = 0.539
[0.505, 0.565] for flagging the worst 20% of errors (`table_mechanism.csv`,
`reports/figures/india/final/h1_deciles.png`). Both intervals exclude chance, but the effect is
small, and in the backtests the AUROC is 0.526, 0.512 and 0.516. On validation, D added nothing
beyond the Historical agent's band width and item volume. H1 is supported only weakly.

## 5.3 H2: does deliberation help the triggered cases?

Of the 892 triggered test points (trigger rate 26.5%), 30.3% improved and 24.6% got worse after
deliberation, and the mean absolute error fell from 12.661 to 12.633 (Wilcoxon p = 0.007;
`reports/figures/india/final/before_after.png`). No case fell back to the fast path, and agents
held their position in 58.7% of revisions, so the revisions are not simple compliance with the
critic. The same comparison on validation was not significant (p = 0.32). H2 is supported on
test, with a small effect that was not seen on validation.

## 5.4 H3: selective vs always-on deliberation

RetailSense and always-on deliberation are indistinguishable in accuracy (+0.0002 [-0.0003,
0.0008], DM p = 0.58), but RetailSense uses 1,331 rather than 5,018 LLM calls per 1,000 forecasts
(-73%) and 6.9 s rather than 20.7 s mean latency per forecast (`table_latency_llm.csv`; latency
includes queueing behind four parallel requests on one GPU). Always-on has a lower inventory cost
($199,868 vs $206,313), but the per-item difference is not significant (+$107 [-6, 266]). H3 is
supported for accuracy and cost.

## 5.5 H4: disagreement vs a random trigger

Triggering on D is not reliably better than triggering at random at the same rate. On test the
WAPE difference is -0.0003 [-0.0006, -0.00001], but DM p = 0.08 and the item-level Wilcoxon
p = 0.07, and the inventory cost difference is -$1.3 per item [-23, 20]. In all three backtests
the random trigger has the lower WAPE. Along the cost-accuracy curve (`table_cost_accuracy.csv`,
`reports/figures/india/final/cost_accuracy.png`) selective triggering is about two random-draw
standard deviations better at trigger rates of 10-20% but worse at 40-80%. H4 is not supported.

## 5.6 H5: from forecasts to inventory

ETS has the lowest total test cost ($138,621), followed by LightGBM with all features ($183,250);
RetailSense costs $206,313 (`table_inventory.csv`). ETS is significantly cheaper than RetailSense
(+$1,128 per item [220, 2,211]), even though RetailSense is more accurate than ETS. The ranking by
cost differs from the ranking by WAPE because the policy also uses the bands, and lost-sales cost
drives the differences ($63,722 for ETS vs $136,574 for RetailSense). With a 7-day lead time
seasonal naive is cheapest and RetailSense is the cheapest multi-agent system
(`table_lead_time.csv`). H5 is not supported for RetailSense.

## 5.7 Robustness and ablations

Across the three backtest windows (`table_backtest.csv`) LightGBM with all features is always the
most accurate; the cheapest system is ETS in one window and a multi-agent consensus in the other
two. The ablations (`table_ablations.csv`) show that calibrating tau matters (a fixed tau of 0.3
triggers on 0.3% of points), that normalising D matters (the unnormalised spread has AUROC
0.465), and that revised values add a little over re-weighting alone (WAPE 0.37752 vs 0.37768).
Removing the Weather or Trend agent improves WAPE (0.37552 and 0.37650 vs 0.37772 without
deliberation), while removing Calendar worsens it (0.38199). Adding noise to the weather inputs
raises the trigger rate from 26.5% to 36.5% and changes a third of the triggered set (Jaccard
0.64).

## 5.8 Case study

The case was chosen by a fixed rule, the highest-D triggered point (`case_study.json`,
`reports/figures/india/final/case_study.png`): item sevai_2 on d_1907 with D = 0.602. The
consensus was 3.35 units against an actual 2.0. The critic described the conflict only in general
terms (the agents use different types of evidence), the revisions left the forecast at 3.35, and
the policy acted on that number. With the 8B local model, the critic rarely names which specific
evidence is wrong, which is consistent with the small effects in Sections 5.3-5.5.
