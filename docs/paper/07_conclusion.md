# 7 Conclusion and Future Work

## 7.1 Answers to the hypotheses

RetailSense was evaluated on a simulated Indian kirana store and on real sales from the Rohlik
online grocer. The India sales come from a documented generator that uses real weather and a real
festival calendar. They are not a real store's sales. India results below are from the single
frozen test run. Rohlik results are from validation, because the Rohlik test run is still in
progress.

- **H1 (D predicts error): weakly supported.** India test: Spearman rho = 0.072, AUROC = 0.539.
  Rohlik validation: rho = 0.099, AUROC = 0.570. On India validation and Rohlik validation, the
  Historical agent's band width and item volume predicted error better than D (Rohlik AUROC 0.706
  and 0.685 vs 0.570), and D added almost nothing beyond them (Rohlik band + volume 0.7124 vs
  band + volume + D 0.7124).
  Rohlik test: [ROHLIK TEST: pending final run].
- **H2 (deliberation helps triggered cases): supported on India test only, with a small effect.**
  30.3% of cases improved and 24.6% worsened (Wilcoxon p = 0.007); mean absolute error moved
  12.661 → 12.633 on 892 triggered points. This is a single-seed result that did not replicate on
  India or Rohlik validation, so it should be treated as tentative. The result was not significant
  on India validation (p = 0.32) or on Rohlik validation (p = 0.26).
- **H3 (selective matches always-on at lower cost): supported for accuracy and LLM cost, not for
  inventory.** On India test, RetailSense used 1,331 rather than 5,018 LLM calls per 1,000
  forecasts with no significant WAPE difference (DM p = 0.58). On Rohlik validation, selective did not match always-on:
  always-on had WAPE 0.1602 vs 0.1618 for selective, using 4,040 vs 811 LLM calls per 1,000
  forecasts; no significance test was run on these arms.
- **H4 (D beats a random trigger): not supported.** On India test, selective was slightly better
  than random (WAPE diff −0.0003, bootstrap CI [−0.0006, −0.00001]), but neither the DM test
  (p = 0.08) nor the item Wilcoxon (p = 0.07) was significant. Random had the lower WAPE in all
  three India backtests and on Rohlik validation (0.1615 vs 0.1618).
- **H5 (better forecasts give better inventory): not supported for RetailSense.** ETS had the
  lowest India test cost ($138,621, vs $206,313 for RetailSense).

On the India test, a single LightGBM with all features (WAPE 0.3688) was more accurate than every
multi-agent system (best 0.3773). On Rohlik validation, LightGBM had the best single-model WAPE
(0.162); the deliberation arms were scored on a 1,500-case subset and cannot be compared directly. Rohlik test results for H2-H5: [ROHLIK TEST: pending final run].

## 7.2 Main lesson

Disagreement can only signal risk if the agents disagree for informative reasons. Here, the
context agents are a shared baseline plus a small adjustment, and their forecasts are correlated
above the 0.95 gate (0.994-0.997 on Rohlik). D therefore measures the size of the adjustment rather
than real conflict between evidence sources. This fits earlier findings that disagreement is a
weak proxy for uncertainty [zarnowitz1987] and that spread predicts skill mainly at the extremes
[whitaker1998]. The critic's mixed effect on triggered cases is also consistent with the limits of
LLM self-correction [huang2024selfcorrect]. The gap between forecast accuracy and inventory cost
supports evaluating forecasts through a simulator [syntetos2010, kourentzes2020].

## 7.3 Future work

1. **Item-aware context agents.** Context agents currently lack item identity, so they apply
   about the same adjustment to every item. Giving them item identity would let them direct each
   signal to the items it affects, and could give D real information.
2. **Stronger critics.** All reported results use one LLM, an 8B local model (llama3.1 via
   Ollama); an earlier pilot with gemini-2.5-flash-lite was dropped after a 100% hold rate. A stronger LLM might name specific
   wrong evidence and change the H2-H4 results.
3. **Multi-seed runs and more datasets.** All results use a single LightGBM seed. Differences below
   about 0.001 WAPE need multi-seed evidence and replication on further real datasets.
4. **Weather forecasts available at prediction time.** Observed weather was used as a perfect
   forecast. Using archived issued forecasts would give a realistic test of the Weather agent.
5. **Learning when to deliberate.** A fixed threshold on D was not reliably better than a random
   trigger at the same rate: small gains at some rates were not significant and did not replicate
   across the backtests or Rohlik. A learned
   router that uses band width, volume and D, in the spirit of cost-aware LLM cascades
   [chen2023frugalgpt, ong2024routellm], is the natural next step.

## References

- [chen2023frugalgpt] Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to use large language models while reducing cost and improving performance. arXiv:2305.05176. https://arxiv.org/abs/2305.05176
- [huang2024selfcorrect] Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D. (2024). Large language models cannot self-correct reasoning yet. ICLR 2024. arXiv:2310.01798. https://arxiv.org/abs/2310.01798
- [kourentzes2020] Kourentzes, N., Trapero, J. R., & Barrow, D. K. (2020). Optimising forecasting models for inventory planning. International Journal of Production Economics, 225, 107597. doi:10.1016/j.ijpe.2019.107597. https://api.crossref.org/works/10.1016/j.ijpe.2019.107597
- [ong2024routellm] Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W., & Stoica, I. (2024). RouteLLM: Learning to route LLMs with preference data. arXiv:2406.18665. https://arxiv.org/abs/2406.18665
- [syntetos2010] Syntetos, A. A., Nikolopoulos, K., & Boylan, J. E. (2010). Judging the judges through accuracy-implication metrics: The case of inventory forecasting. International Journal of Forecasting, 26(1), 134-143. doi:10.1016/j.ijforecast.2009.05.016. https://api.crossref.org/works/10.1016/j.ijforecast.2009.05.016
- [whitaker1998] Whitaker, J. S., & Loughe, A. F. (1998). The relationship between ensemble spread and ensemble mean skill. Monthly Weather Review, 126(12), 3292-3302. https://api.crossref.org/works/10.1175/1520-0493(1998)126%3C3292:TRBESA%3E2.0.CO;2
- [zarnowitz1987] Zarnowitz, V., & Lambros, L. A. (1987). Consensus and uncertainty in economic prediction. Journal of Political Economy, 95(3), 591-621. doi:10.1086/261473. https://ideas.repec.org/a/ucp/jpolec/v95y1987i3p591-621.html
