# 1 Introduction

A store manager who restocks shelves every week has to decide how much of each item to order before
demand is known. Ordering too little loses sales. Ordering too much ties up cash and shelf space.
The forecast behind that decision draws on several kinds of evidence: recent sales, the calendar
(weekdays, festivals and public holidays), the weather and shifts in customer interest. Machine-
learning models such as LightGBM [ke2017lightgbm] combine these signals well, and gradient-boosted
models dominated the large M5 retail benchmark [makridakis2022m5]. Large language models (LLMs) may
add a different ability: in principle they can read conflicting evidence and reason about which
source to trust [du2023].
Each LLM call, however, costs time and money. Calling the model on every item and every day is
expensive, and the value of doing so is not obvious [tan2024].

This paper asks whether an LLM can be used *selectively*: only on the forecasts where it is most
likely to help. The idea comes from forecast combination and ensemble forecasting. Combining
forecasts usually beats a single forecast [bates1969; timmermann2006; wang2023], and the spread
between forecasters is a signal of uncertainty, although an imperfect one [zarnowitz1987;
whitaker1998]. If several forecasters disagree, the case may be risky, and a closer look may be
worth its cost.

## 1.1 The approach

RetailSense builds a small set of specialised agents, each of which sees one kind of evidence. A
Historical agent forecasts from lagged sales and price. Calendar, Weather and Trend agents adjust
that baseline using only their own signal. Each agent returns a daily forecast for the next seven
days, an uncertainty band and a confidence. The orchestrator computes a normalised disagreement
score D, a confidence-weighted coefficient of variation of the agents' next-day forecasts (the
denominator is floored at 1 unit). If D is
below a threshold tau, the system returns the confidence-weighted consensus. If D is at or above
tau, an LLM critic explains the conflict and each agent gets one bounded revision. This design
follows critique-then-revise and multi-agent debate methods [madaan2023; du2023], but limits them
to a single round. It is also a form of selective LLM use, close to cascades and routers
[chen2023frugalgpt; ong2024routellm].

The LLM never produces the final forecast or an order quantity. Unlike LLM-as-forecaster methods
[gruver2023; jin2024timellm], it may only propose a confidence multiplier and a revised value for
each agent's forecast, and code bounds and combines them. Code clips the multiplier to [0.25, 2.0]
and the revised value to the agent's own 10-90% band, then computes the final forecast as a
confidence-weighted average. The forecast then feeds a
periodic-review (s, S) inventory policy [silver2017] that outputs RESTOCK, HOLD or REDUCE. The
order quantity is always computed by code.

## 1.2 Research question and hypotheses

The main research question is: *does disagreement-triggered deliberation among heterogeneous
forecasting agents improve demand forecasting and inventory restocking decisions compared with
statistical forecasting, single-agent forecasting, simple multi-agent consensus, and always-on
deliberation?* It is split into five hypotheses:

- **H1.** Normalised disagreement D is positively associated with forecast error.
- **H2.** Selective deliberation lowers error on triggered cases versus no deliberation.
- **H3.** Selective deliberation matches always-on deliberation at a fraction of the LLM cost.
- **H4.** Disagreement is a better trigger than chance.
- **H5.** Better forecasts translate into better inventory outcomes.

H4 is the central test of the idea. If a random trigger at the same rate does as well, the
disagreement signal adds nothing.

## 1.3 Contributions

1. **A frozen protocol with leakage controls.** Every feature row records when it became
   available, and tests assert that no feature is newer than the prediction time. One exception is
   weather: observed archive weather stands in for a perfect forecast, so weather information is
   optimistic; a noise ablation (sd 1.0) tests how sensitive the results are to this. tau is set as
   the 80th percentile of validation D, and tau and the prompt are frozen before the test split is
   read. Each dataset's test split is read by a single final run (the India run was resumed once
   after a GPU crash, replaying completed cases from the cache with no new LLM calls; the Rohlik
   final run is [ROHLIK TEST: pending final run]).
2. **A simulator with logged ground-truth drivers.** The primary dataset is a simulated Indian
   kirana (neighbourhood grocery) chain. Its sales are generated, not observed. They are built from
   real Open-Meteo weather and a real Indian festival calendar, and every multiplier behind each
   simulated sale is logged, so agent behaviour can be checked against the true drivers.
3. **An evaluation through one inventory simulator.** Nine systems, from seasonal naive to always-on
   deliberation, pass through the same (s, S) simulator, because forecast accuracy and inventory
   cost do not always agree [syntetos2010; kourentzes2020]. A random-trigger control at the same
   rate isolates the value of D.
4. **A real-data study.** The protocol is applied to real daily sales from Rohlik, a Czech online
   grocer, with the same code and prompt, but with three agents (no search-interest data, so no
   Trend agent) and a shorter ETS fit window.

## 1.4 Summary of findings

The results are mostly negative, and we report them as such. On the India test split, RetailSense
does not beat a single LightGBM that sees all features (h = 1 WAPE 0.3775 vs 0.3688). D is only
weakly associated with error (Spearman rho = 0.072, AUROC = 0.539), so H1 is weakly supported.
On triggered cases, deliberation improved 30.3% of forecasts and worsened 24.6% (Wilcoxon
p = 0.007), but this small effect was not significant on validation. RetailSense matches
always-on accuracy with 1,331 rather than 5,018 LLM calls per 1,000 forecasts (-73%), which
supports H3 on accuracy and LLM cost, though not on inventory cost (always-on $199,868 vs
RetailSense $206,313; difference not significant). H4 is not supported: the disagreement trigger is not reliably
better than a random trigger at the same rate. H5 is not supported for RetailSense: ETS gives the
lowest inventory cost ($138,621 vs $206,313).

On Rohlik validation data, D is somewhat more informative (rho = 0.099, AUROC = 0.570), but it
adds almost nothing beyond the Historical agent's band width and item volume. On 1,500 stratified
validation cases, always-on deliberation lowers WAPE from 0.1619 to 0.1602 (single run, not
significance-tested), while selective deliberation (0.1618) is slightly worse than a random trigger
at the same rate (0.1615, sd 0.0006). Rohlik test results are [ROHLIK TEST: pending final run].
Taken together, disagreement between these agents is a weak risk signal, and a small 8B LLM critic
does not use it to improve forecasts selectively. These results come with caveats: the agents'
forecasts are highly correlated (0.994-0.997 on Rohlik, and above the 0.95 design gate on India
too), so D mostly measures the size of small context adjustments; LightGBM was run with a single
seed; and only one 8B local LLM was tested. These findings are consistent with work showing that
LLM self-correction is often unreliable [huang2024selfcorrect], although our setting gives the
model some external evidence.

## 1.5 Structure of the paper

Section 2 reviews related work on forecast combination, disagreement as an uncertainty signal,
multi-agent LLM deliberation and selective LLM use. Section 3 describes the agents, the
disagreement score, the critic and revision, and the inventory policy. Section 4 sets out the data,
the systems compared and the statistical tests. Section 5 reports the results for each hypothesis.
Section 6 discusses why the disagreement signal is weak and lists the study's limitations.
Section 7 concludes.

## References

- [bates1969] Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. Operational Research Quarterly, 20(4), 451-468. doi:10.1057/jors.1969.103. https://api.crossref.org/works/10.1057/jors.1969.103
- [chen2023frugalgpt] Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to use large language models while reducing cost and improving performance. arXiv:2305.05176. https://arxiv.org/abs/2305.05176
- [du2023] Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. arXiv:2305.14325. https://arxiv.org/abs/2305.14325
- [gruver2023] Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). Large language models are zero-shot time series forecasters. NeurIPS 2023. arXiv:2310.07820. https://arxiv.org/abs/2310.07820
- [huang2024selfcorrect] Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D. (2024). Large language models cannot self-correct reasoning yet. ICLR 2024. arXiv:2310.01798. https://arxiv.org/abs/2310.01798
- [jin2024timellm] Jin, M., Wang, S., Ma, L., et al. (2024). Time-LLM: Time series forecasting by reprogramming large language models. ICLR 2024. arXiv:2310.01728. https://arxiv.org/abs/2310.01728
- [ke2017lightgbm] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- [kourentzes2020] Kourentzes, N., Trapero, J. R., & Barrow, D. K. (2020). Optimising forecasting models for inventory planning. International Journal of Production Economics, 225, 107597. doi:10.1016/j.ijpe.2019.107597. https://api.crossref.org/works/10.1016/j.ijpe.2019.107597
- [madaan2023] Madaan, A., Tandon, N., Gupta, P., et al. (2023). Self-Refine: Iterative refinement with self-feedback. NeurIPS 2023. arXiv:2303.17651. https://arxiv.org/abs/2303.17651
- [makridakis2022m5] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting, 38(4), 1346-1364. doi:10.1016/j.ijforecast.2021.11.013. https://api.crossref.org/works/10.1016/j.ijforecast.2021.11.013
- [ong2024routellm] Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W., & Stoica, I. (2024). RouteLLM: Learning to route LLMs with preference data. arXiv:2406.18665. https://arxiv.org/abs/2406.18665
- [silver2017] Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017). Inventory and Production Management in Supply Chains (4th ed.). CRC Press. https://www.routledge.com/Inventory-and-Production-Management-in-Supply-Chains/Silver-Pyke-Thomas/p/book/9781466558618
- [syntetos2010] Syntetos, A. A., Nikolopoulos, K., & Boylan, J. E. (2010). Judging the judges through accuracy-implication metrics: The case of inventory forecasting. International Journal of Forecasting, 26(1), 134-143. doi:10.1016/j.ijforecast.2009.05.016. https://api.crossref.org/works/10.1016/j.ijforecast.2009.05.016
- [tan2024] Tan, M., Merrill, M. A., Gupta, V., Althoff, T., & Hartvigsen, T. (2024). Are language models actually useful for time series forecasting? NeurIPS 2024. arXiv:2406.16964. https://arxiv.org/abs/2406.16964
- [timmermann2006] Timmermann, A. (2006). Forecast combinations. In G. Elliott, C. W. J. Granger & A. Timmermann (Eds.), Handbook of Economic Forecasting, Vol. 1, pp. 135-196. Elsevier. https://ideas.repec.org/h/eee/ecofch/1-04.html
- [wang2023] Wang, X., Hyndman, R. J., Li, F., & Kang, Y. (2023). Forecast combinations: An over 50-year review. International Journal of Forecasting, 39(4), 1518-1547. doi:10.1016/j.ijforecast.2022.11.005. https://api.crossref.org/works/10.1016/j.ijforecast.2022.11.005
- [whitaker1998] Whitaker, J. S., & Loughe, A. F. (1998). The relationship between ensemble spread and ensemble mean skill. Monthly Weather Review, 126(12), 3292-3302. https://api.crossref.org/works/10.1175/1520-0493(1998)126%3C3292:TRBESA%3E2.0.CO;2
- [zarnowitz1987] Zarnowitz, V., & Lambros, L. A. (1987). Consensus and uncertainty in economic prediction. Journal of Political Economy, 95(3), 591-621. doi:10.1086/261473. https://ideas.repec.org/a/ucp/jpolec/v95y1987i3p591-621.html
