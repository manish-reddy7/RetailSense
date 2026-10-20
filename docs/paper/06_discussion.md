# 6 Discussion

## 6.1 Why disagreement carries little signal

Disagreement among forecasters is known to be only a partial proxy for uncertainty
[zarnowitz1987], and even a well-built ensemble can show a modest spread-skill relationship
[whitaker1998]. RetailSense adds a structural reason. The context agents (three on India, two on
Rohlik, where the Trend agent is absent) are residual adjusters: each forecasts the Historical agent's baseline plus a small learned adjustment. Their
forecasts are therefore highly correlated with the baseline and with each other. On Rohlik
validation the pairwise agent forecast correlation is 0.994-0.997, and India also fails the
planned 0.95 gate. D then measures the size of the adjustments rather than a genuine conflict
between independent views of demand.

The adjustments themselves are weak because the context agents do not see item identity. They
cannot route a signal to the items it affects, so they apply roughly the same adjustment to every
item. Because the India data are simulated, the planted effects are known and this can be checked
directly. On validation the Weather agent added about +2 units to soap and atta, whose true weather
effect is 0, and +2.3 to tea, whose true effect is -1.4. The Trend agent added a flat +1.3 where
the true ice-cream interest shock was +11.7. The ablations agree: removing the Weather or Trend
agent improves test WAPE (0.37552 and 0.37650 against 0.37772 without deliberation), while only
Calendar helps (removing it gives 0.38199). D also added nothing on validation beyond the
Historical agent's band width and item volume. On India test, D is weakly related to error
(Spearman rho = 0.072, AUROC = 0.539).

## 6.2 Why the critic helped little

Self-critique and multi-agent debate can improve LLM reasoning [madaan2023; du2023], but without
reliable external feedback self-correction often fails and can make answers worse
[huang2024selfcorrect]. Our critic is an 8B local model (llama3.1 via Ollama). It usually
described conflicts in general terms, for example that the agents use different types of
evidence, and rarely named which evidence was wrong. The case study shows this: for the
highest-D point (sevai_2, d_1907, D = 0.602) the critique was generic and the final forecast stayed
at the consensus of 3.35 against an actual 2.0. The deliberation outcomes reflect this. On India
test, 30.3% of 892 triggered points improved and 24.6% got worse (MAE 12.661 to 12.633, Wilcoxon
p = 0.007), but this did not replicate on India validation (p = 0.32). On Rohlik validation, 24.4%
improved and 29.4% got worse (MAE 51.21 to 51.01, p = 0.26). Hold rates of 58.7% (India test) and 55.3% (Rohlik
validation) indicate that agents were not simply complying with the critic, which is the failure
mode that sycophancy research warns about [sharma2023sycophancy]. Hold rates suggest the prompt was
not coercive. Whether the limit is the critic's strength or the weak, shared evidence of the agents
cannot be separated with a single 8B model. This fits wider evidence that LLM components often add little
to numeric forecasting accuracy [tan2024]. RetailSense deliberately keeps the LLM away from the
demand number itself, unlike LLM-as-forecaster designs [gruver2023; jin2024timellm], so a weak
critic can only make small, bounded changes.

## 6.3 What differs on real data

On Rohlik validation, D is a stronger risk signal than on India: Spearman rho = 0.099
[0.053, 0.135] and AUROC = 0.570 [0.535, 0.595]. The top decile of D has mean relative error 0.4366
against 0.2143 in the bottom decile. Even so, D adds almost nothing beyond the Historical band
width (AUROC 0.7057 alone, 0.7058 with D). Within volume tertiles, D is close to uninformative
(rho -0.028 to 0.049; AUROC 0.49-0.53), and adding D to band width plus volume leaves AUROC at
0.7124. Most of D's pooled signal therefore reflects item volume rather than real conflict between
agents. On a 1,500-case stratified validation sample, always-on deliberation lowered WAPE from
0.1619 to 0.1602 (about 1%). This gap was not tested for significance and comes from one seed.
India showed no comparable gain. Selective triggering did not capture
it: selective reached 0.1618 against 0.1615 (sd 0.0006) for a random trigger at the same 20% rate.
On the cost-accuracy curve, selective is worse than the random mean at every rate from 0.05 to
0.30, including the 20% operating point. It is better mainly at rates of 0.65-0.85 (for example
0.1598 vs 0.1607, sd 0.0006, at 0.75), with mixed results in between. Real data
therefore sharpen the picture: disagreement marks riskier forecasts somewhat better, but the
critic does not repair those forecasts selectively. The Rohlik test split confirms this. D was
slightly more informative than on India (rho = 0.119, AUROC = 0.579) and clearly more so in the
three backtest windows (rho 0.28-0.44, AUROC 0.69-0.78). Even so, the triggered cases improved and
worsened about equally (27.6% vs 26.6%, Wilcoxon p = 1.00), and selective triggering matched a
random trigger (DM p = 0.57). On test, always-on deliberation was no longer better than selective
(WAPE 0.1741 vs 0.1745, DM p = 0.59).

## 6.4 Why accuracy and cost rankings differ

The ranking by WAPE does not carry over to inventory cost. ETS is less accurate than RetailSense
on India test but has the lowest total cost ($138,621 against $206,313). The (s, S) policy uses the
bands as well as the point forecast [silver2017], and lost-sales cost dominates the differences
($63,722 for ETS against $136,574 for RetailSense). This agrees with earlier work showing that
accuracy measures do not map directly onto inventory performance [syntetos2010] and that the most
accurate model is not always the best for inventory [kourentzes2020]. On Rohlik validation the
cheapest system was the Historical agent ($193,385), while the most accurate, LightGBM with all
features, cost $422,135. Evaluating forecasts through a common simulator is therefore necessary,
not optional.

## 6.5 Practical implications

Forecast combination is a strong default [bates1969; timmermann2006; wang2023], and a single
LightGBM with all features remains the most accurate system on India test (WAPE 0.3688 against
0.3775 for RetailSense). Splitting evidence across agents costs accuracy here. The clearest
practical result is cost: on India test, selective deliberation showed no detectable accuracy
difference from always-on (+0.0002 [-0.0003, 0.0008], DM p = 0.58) with 73% fewer LLM calls (1,331
vs 5,018 per 1,000 forecasts). Always-on had lower total inventory cost ($199,868 vs $206,313),
although the per-item difference was not significant (+$107 [-6, 266]). This is in the spirit of cascades and routers that invoke an expensive model only when
needed [chen2023frugalgpt; ong2024routellm]. But the saving is only worth having if deliberation
helps, and the random-trigger control shows no reliable advantage for D over chance (India test DM
p = 0.08, item-level Wilcoxon p = 0.07; random was better in all three backtests and on Rohlik
validation; H4 not supported). Selective deliberation could pay off when the agents hold genuinely independent
evidence, so that D measures real conflict; when context agents can target specific items; and
when the critic is strong enough to identify which evidence is wrong. On Rohlik validation, where
always-on helped, a trigger that selects the cases deliberation can fix, rather than the cases
that look risky, would be required.

## 6.6 Threats to validity

- **Simulated India sales.** The India data come from a documented generator with real Open-Meteo
  weather and a real Indian festival calendar, but the sales are simulated with known drivers.
  India results describe method behaviour, not a real store. Rohlik provides real sales, and its
  test results agree with India on every hypothesis except H2, where the small India effect did
  not replicate.
- **Single seed in the frozen run.** The test run used one LightGBM seed (42). A post-hoc check
  refit systems 1-6 on validation under four more seeds (no LLM, so systems 7-9 were not rerun;
  `reports/*/posthoc/val_seeds.csv`). The seed standard deviation of h = 1 WAPE is 0.0004-0.0006 on
  India and 0.0007-0.0023 on Rohlik, so gaps between systems 5-9 (below 0.0005) are within seed
  noise. The single all-features LightGBM beat the confidence-weighted multi-agent consensus under
  every seed (gap 0.0097 +/- 0.0006 India, 0.0098 +/- 0.0026 Rohlik). H1 is seed-sensitive on
  India: rho ranged 0.027-0.064 (mean 0.044) and AUROC 0.505-0.535, and seed 42 gave the highest
  value of both, so the reported India H1 is at the favourable end. On Rohlik rho was 0.085-0.104.
- **One LLM.** Only an 8B local model was tested. A stronger critic could change H2-H4.
- **Weather inputs.** The study feeds the Open-Meteo archive value for the target day, i.e. weather
  as a perfect forecast. A post-hoc comparison with Open-Meteo's archive of *issued* short-lead
  forecasts (`reports/*/posthoc/val_issued_weather.csv`) bounds the effect. For Mumbai the two
  archives return identical values on every day of the last training year, validation and test,
  so the India inputs equal forecasts that were available at h = 1; because the India demand was
  generated from this same series, the Weather agent still sees its true driver without error,
  which is a property of the simulation. For Prague the issued forecasts differ (MAE 0.9 C tmax,
  1.1 C tmin, 0.6 mm precipitation on validation), yet replacing the validation inputs with them,
  models unchanged, moved h = 1 WAPE by -0.0001 for the Weather agent and -0.0004 for the
  all-features control. The archive holds the shortest-lead forecast only, so for h > 1 the real
  penalty is larger; the noise ablation (trigger rate 26.5% to 36.5%, Jaccard 0.64) covers that
  direction.
- **Censored demand.** Sales are capped on stock-out days and treated as demand: 1.2% of India
  test item-days (generator flag) and 5.0% of Rohlik test item-days (availability < 1). Scoring
  without those days changes h = 1 WAPE by at most 0.016 and leaves the test ranking of systems
  1-6 unchanged on both datasets; on Rohlik validation ETS and the Historical agent, 0.0007 apart,
  swap places (`reports/*/posthoc/censoring.csv`).
- **Fixed inventory parameters.** L = 3, R = 7, z = 1.65 and the costs ($5 per order, 25% a year
  holding, lost sales at 1x price) are illustrative, and z = 1.65 is not cost-optimal when a lost
  sale costs far more than a day of holding. In the validation sensitivity runs a 0.99 service
  level lowers every system's India cost by 26-33%, but ETS remains the cheapest system
  (`reports/india/phase4_sensitivity.csv`), so the cost ranking is not an artefact of z.
- **Context agents cannot target items.** A residual agent sees the baseline and its own evidence
  but no item identity, so it applies about the same adjustment to every item (Trend adds a flat
  +1.3 while the planted ice-cream effect is +11.7). Confidence weights are uninformative (Spearman
  with error -0.09 to -0.12; the most confident agent is the most accurate on 33% of points).
- **Generated interest signal.** The India search-interest index is produced by the generator and
  is built to lead demand, so the Trend agent is tested on a signal designed to be useful.
- **Validation window.** The India validation window (December-March) has 0 rainy days and no
  major festival, so tau and all design choices were made on a period with little weather or
  festival evidence.
- **Missing comparisons.** There is no deep-learning or foundation-model baseline, and no ablation
  that revises without the critic's conflict analysis. A post-hoc validation check of other
  gradient-boosting learners (XGBoost, scikit-learn HGB) changed h = 1 WAPE by at most 3%, with no
  learner best on both datasets (`reports/*/posthoc/val_learners.csv`).
- **Process.** The go/pivot decision required after the early H1 smoke test (rho = 0.064 < 0.15)
  was recorded retrospectively, after both test runs.
- **Rohlik item selection.** Items were chosen by having a sales record on every day from d_1000,
  which uses presence (not sales) in the evaluation period. The Trend agent is absent, so Rohlik
  uses three agents.
- **Latency.** Latency includes queueing behind four parallel requests on one laptop GPU, and a
  replay from cache measures cache latency. Only accuracy, LLM calls and tokens are reproducible.
- **Validation-test mismatch.** The India test H2 effect (p = 0.007) was not significant on
  validation (p = 0.32), and H2 was not significant on Rohlik validation either.
- **Failed independence gate.** Pairwise agent forecast correlation exceeds the planned 0.95 gate
  on both datasets (Rohlik 0.994-0.997), so D largely measures adjustment size. This is a design
  failure, not a minor caveat.
- **Run details.** The Rohlik deliberation arms used a 1,500-case stratified validation sample.
  The India final run was resumed after two GPU out-of-memory crashes by replaying 2,350 cases from
  cache under the same config hash.

## References

- [bates1969] Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. Operational Research Quarterly, 20(4), 451-468. doi:10.1057/jors.1969.103. https://api.crossref.org/works/10.1057/jors.1969.103
- [chen2023frugalgpt] Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to use large language models while reducing cost and improving performance. arXiv:2305.05176. https://arxiv.org/abs/2305.05176
- [du2023] Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. arXiv:2305.14325. https://arxiv.org/abs/2305.14325
- [gruver2023] Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). Large language models are zero-shot time series forecasters. NeurIPS 2023. arXiv:2310.07820. https://arxiv.org/abs/2310.07820
- [huang2024selfcorrect] Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D. (2024). Large language models cannot self-correct reasoning yet. ICLR 2024. arXiv:2310.01798. https://arxiv.org/abs/2310.01798
- [jin2024timellm] Jin, M., Wang, S., Ma, L., et al. (2024). Time-LLM: Time series forecasting by reprogramming large language models. ICLR 2024. arXiv:2310.01728. https://arxiv.org/abs/2310.01728
- [kourentzes2020] Kourentzes, N., Trapero, J. R., & Barrow, D. K. (2020). Optimising forecasting models for inventory planning. International Journal of Production Economics, 225, 107597. doi:10.1016/j.ijpe.2019.107597. https://api.crossref.org/works/10.1016/j.ijpe.2019.107597
- [madaan2023] Madaan, A., Tandon, N., Gupta, P., et al. (2023). Self-Refine: Iterative refinement with self-feedback. NeurIPS 2023. arXiv:2303.17651. https://arxiv.org/abs/2303.17651
- [ong2024routellm] Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W., & Stoica, I. (2024). RouteLLM: Learning to route LLMs with preference data. arXiv:2406.18665. https://arxiv.org/abs/2406.18665
- [sharma2023sycophancy] Sharma, M., Tong, M., Korbak, T., et al. (2023). Towards understanding sycophancy in language models. arXiv:2310.13548. https://arxiv.org/abs/2310.13548
- [silver2017] Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017). Inventory and Production Management in Supply Chains (4th ed.). CRC Press. https://www.routledge.com/Inventory-and-Production-Management-in-Supply-Chains/Silver-Pyke-Thomas/p/book/9781466558618
- [syntetos2010] Syntetos, A. A., Nikolopoulos, K., & Boylan, J. E. (2010). Judging the judges through accuracy-implication metrics: The case of inventory forecasting. International Journal of Forecasting, 26(1), 134-143. doi:10.1016/j.ijforecast.2009.05.016. https://api.crossref.org/works/10.1016/j.ijforecast.2009.05.016
- [tan2024] Tan, M., Merrill, M. A., Gupta, V., Althoff, T., & Hartvigsen, T. (2024). Are language models actually useful for time series forecasting? NeurIPS 2024. arXiv:2406.16964. https://arxiv.org/abs/2406.16964
- [timmermann2006] Timmermann, A. (2006). Forecast combinations. In G. Elliott, C. W. J. Granger & A. Timmermann (Eds.), Handbook of Economic Forecasting, Vol. 1, pp. 135-196. Elsevier. https://ideas.repec.org/h/eee/ecofch/1-04.html
- [wang2023] Wang, X., Hyndman, R. J., Li, F., & Kang, Y. (2023). Forecast combinations: An over 50-year review. International Journal of Forecasting, 39(4), 1518-1547. doi:10.1016/j.ijforecast.2022.11.005. https://api.crossref.org/works/10.1016/j.ijforecast.2022.11.005
- [whitaker1998] Whitaker, J. S., & Loughe, A. F. (1998). The relationship between ensemble spread and ensemble mean skill. Monthly Weather Review, 126(12), 3292-3302. https://api.crossref.org/works/10.1175/1520-0493(1998)126%3C3292:TRBESA%3E2.0.CO;2
- [zarnowitz1987] Zarnowitz, V., & Lambros, L. A. (1987). Consensus and uncertainty in economic prediction. Journal of Political Economy, 95(3), 591-621. doi:10.1086/261473. https://ideas.repec.org/a/ucp/jpolec/v95y1987i3p591-621.html
