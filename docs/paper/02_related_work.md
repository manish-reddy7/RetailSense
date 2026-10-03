# 2 Related Work

RetailSense draws on seven lines of work. For each we state what it established and how this
paper relates to it or differs from it.

## 2.1 Forecast combination

Combining forecasts is an old and well-supported idea. Bates and Granger showed that a
combination weighted by past error can have a lower mean squared error than any of its inputs
[bates1969]. Later surveys explain why combinations tend to beat individual forecasts on average
and how combination weights are estimated [timmermann2006], and a recent review covers simple,
time-varying, nonlinear and cross-learning combinations of point and probabilistic forecasts
[wang2023]. The RetailSense fast path is a combination of this kind: agent forecasts are weighted
by a confidence derived from trailing error. The paper differs in what it asks of the combination.
The spread of the agent forecasts around the combination is also used as a signal to decide
whether further processing is worthwhile. Timmermann also notes that combining forecasts is mainly
useful when the underlying information sets cannot be pooled into one model [timmermann2006]. On
the India test split, the multi-agent systems do not beat one LightGBM trained on every feature
(h1 WAPE 0.3775 vs 0.3688).

## 2.2 Disagreement as an uncertainty signal

Zarnowitz and Lambros found that disagreement among forecasters is positively correlated with
uncertainty but understates it [zarnowitz1987]. In weather prediction, the spread-skill
literature shows that the correlation between ensemble spread and forecast error can be modest
even for a perfect ensemble, and that spread is informative mainly when it is extreme
[whitaker1998]. These results motivate treating the disagreement score D as a proxy for risk
rather than a measure of it, and they set a realistic expectation for H1. They also suggest why
a trigger at the 80th percentile of validation D may not select the cases that deliberation can improve. RetailSense
differs from both settings in that its agents are not independent forecasters or perturbed
ensemble members. The context agents are residual adjusters on a shared baseline, so their spread
measures the size of their adjustments. This is one reason the observed association between D and
error is weak. Pairwise agent forecast correlations exceed the 0.95 design gate (0.994-0.997 on
Rohlik validation, and India also fails), and D adds almost nothing to the historical band width
as a predictor of large errors (§5).

## 2.3 Multi-agent LLM debate, self-refinement and social compliance

Multi-agent debate, in which several LLM instances exchange answers over rounds, has been
reported to improve reasoning and factuality [du2023]. Self-Refine shows that a critique-then-revise
loop can improve outputs without retraining [madaan2023]. RetailSense borrows the structure of
both: a critic explains the conflict and each agent gets one bounded revision. Two findings
temper the expected benefit. Without external feedback, LLM self-correction often fails and can
make answers worse [huang2024selfcorrect]. RLHF-trained assistants also tend to change their
answers under social pressure, a behaviour described as sycophancy [sharma2023sycophancy]. A
deliberation step that makes agents move whenever they are challenged would therefore measure
compliance rather than evidence. This is why the design tracks the hold rate, the share of
revisions in which an agent keeps its position, as a gate, and treats holding as a normal answer.
It also explains why the deliberation output is constrained: the LLM can only rescale an agent's
weight within [0.25, 2.0] and move its value inside the agent's own band. The paper differs from
the debate literature in its task. Debate work targets question answering and reasoning, whereas
here deliberation acts on numeric forecasts produced by other models.

## 2.4 LLMs for time-series forecasting

A separate line of work uses LLMs as forecasters. Gruver et al. show that LLMs can forecast
numeric series zero-shot when the series is encoded as text [gruver2023], and Time-LLM
repurposes a frozen LLM as a forecaster by reprogramming its inputs [jin2024timellm]. Tan et al.
question the benefit: removing the LLM from such methods, or replacing it with a simple attention
layer, does not hurt accuracy [tan2024]. RetailSense takes the opposite design choice to the
LLM-as-forecaster work. The LLM never produces the demand number or the order quantity. It only
critiques forecasts made by gradient-boosted models and proposes bounded re-weightings and in-band
revisions; code clips these and computes the final value. The small or null LLM gains we observe are consistent with the scepticism of [tan2024].

## 2.5 Retail demand forecasting and gradient boosting

In the M5 competition, a large retail forecasting benchmark, LightGBM-based machine-learning
methods dominated [makridakis2022m5]. LightGBM itself is a gradient-boosted decision tree
implementation designed for speed on large tabular data [ke2017lightgbm]. These results justify
LightGBM as the Historical agent and as the model behind every context agent, and they justify a
single all-features LightGBM as a strong baseline. We evaluate on two datasets. The primary one is
generated by a documented simulator of an Indian kirana store, with real weather and a real
festival calendar as demand drivers; it is not a real store's sales. The second is real daily
sales from the Czech online grocer Rohlik. Neither is part of the M5 benchmark.

## 2.6 Forecast accuracy versus inventory performance

Standard accuracy measures do not translate directly into inventory efficiency [syntetos2010],
and the mapping from forecast error to inventory outcome is nonlinear, so the most accurate model
is not always the best for inventory [kourentzes2020]. For this reason every system in this paper,
baselines included, is passed through the same periodic-review (s, S) simulator, with safety stock
set by the standard textbook formula [silver2017]. H5 is tested on simulated inventory cost and
service, not on WAPE alone. Our results illustrate the point made in [kourentzes2020]: on the
India test split, ETS has the lowest total inventory cost although it is not the most accurate
system.

## 2.7 Cost-aware and selective LLM use

FrugalGPT studies cascades and selective invocation of LLMs to trade cost against accuracy
[chen2023frugalgpt], and RouteLLM learns routers that decide per query whether to call the more
expensive model [ong2024routellm]. Triggering deliberation only when D >= tau is a simple,
unlearned instance of the same idea: the expensive step runs only on cases whose D reaches a fixed
threshold, calibrated to trigger 20% of validation cases. The paper differs in how the router is evaluated. We compare it against a random
trigger at the same rate and trace a cost-accuracy curve over trigger rates, which isolates the
value of the signal from the value of deliberating more cases. Selective triggering was
not reliably better than random. On the India test split its WAPE was 0.0003 lower (bootstrap 95%
CI [-0.0006, -0.00001]), but the Diebold-Mariano and item-level Wilcoxon tests were not significant
(p = 0.08 and 0.07), and random had the lower WAPE in all three backtests. On Rohlik validation,
selective was slightly worse than random (WAPE 0.1618 vs 0.1615, sd 0.0006). The Rohlik test
result is pending the final run.

## References

- [bates1969] Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. Operational
  Research Quarterly, 20(4), 451-468. doi:10.1057/jors.1969.103.
  https://api.crossref.org/works/10.1057/jors.1969.103
- [chen2023frugalgpt] Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to use large language
  models while reducing cost and improving performance. arXiv:2305.05176.
  https://arxiv.org/abs/2305.05176
- [du2023] Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving
  factuality and reasoning in language models through multiagent debate. arXiv:2305.14325.
  https://arxiv.org/abs/2305.14325
- [gruver2023] Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). Large language models are
  zero-shot time series forecasters. NeurIPS 2023. arXiv:2310.07820.
  https://arxiv.org/abs/2310.07820
- [huang2024selfcorrect] Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., &
  Zhou, D. (2024). Large language models cannot self-correct reasoning yet. ICLR 2024.
  arXiv:2310.01798. https://arxiv.org/abs/2310.01798
- [jin2024timellm] Jin, M., Wang, S., Ma, L., et al. (2024). Time-LLM: Time series forecasting by
  reprogramming large language models. ICLR 2024. arXiv:2310.01728.
  https://arxiv.org/abs/2310.01728
- [ke2017lightgbm] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y.
  (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural
  Information Processing Systems 30 (NIPS 2017).
  https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- [kourentzes2020] Kourentzes, N., Trapero, J. R., & Barrow, D. K. (2020). Optimising forecasting
  models for inventory planning. International Journal of Production Economics, 225, 107597.
  doi:10.1016/j.ijpe.2019.107597. https://api.crossref.org/works/10.1016/j.ijpe.2019.107597
- [madaan2023] Madaan, A., Tandon, N., Gupta, P., et al. (2023). Self-Refine: Iterative refinement
  with self-feedback. arXiv:2303.17651 (NeurIPS 2023). https://arxiv.org/abs/2303.17651
- [makridakis2022m5] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy
  competition: Results, findings, and conclusions. International Journal of Forecasting, 38(4),
  1346-1364. doi:10.1016/j.ijforecast.2021.11.013.
  https://api.crossref.org/works/10.1016/j.ijforecast.2021.11.013
- [ong2024routellm] Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E.,
  Kadous, M. W., & Stoica, I. (2024). RouteLLM: Learning to route LLMs with preference data.
  arXiv:2406.18665. https://arxiv.org/abs/2406.18665
- [sharma2023sycophancy] Sharma, M., Tong, M., Korbak, T., et al. (2023). Towards understanding
  sycophancy in language models. arXiv:2310.13548. https://arxiv.org/abs/2310.13548
- [silver2017] Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017). Inventory and Production
  Management in Supply Chains (4th ed.). CRC Press.
  https://www.routledge.com/Inventory-and-Production-Management-in-Supply-Chains/Silver-Pyke-Thomas/p/book/9781466558618
- [syntetos2010] Syntetos, A. A., Nikolopoulos, K., & Boylan, J. E. (2010). Judging the judges
  through accuracy-implication metrics: The case of inventory forecasting. International Journal
  of Forecasting, 26(1), 134-143. doi:10.1016/j.ijforecast.2009.05.016.
  https://api.crossref.org/works/10.1016/j.ijforecast.2009.05.016
- [tan2024] Tan, M., Merrill, M. A., Gupta, V., Althoff, T., & Hartvigsen, T. (2024). Are language
  models actually useful for time series forecasting? NeurIPS 2024. arXiv:2406.16964.
  https://arxiv.org/abs/2406.16964
- [timmermann2006] Timmermann, A. (2006). Forecast combinations. In G. Elliott, C. W. J. Granger &
  A. Timmermann (Eds.), Handbook of Economic Forecasting, Vol. 1, pp. 135-196. Elsevier.
  https://ideas.repec.org/h/eee/ecofch/1-04.html
- [wang2023] Wang, X., Hyndman, R. J., Li, F., & Kang, Y. (2023). Forecast combinations: An over
  50-year review. International Journal of Forecasting, 39(4), 1518-1547.
  doi:10.1016/j.ijforecast.2022.11.005. https://api.crossref.org/works/10.1016/j.ijforecast.2022.11.005
- [whitaker1998] Whitaker, J. S., & Loughe, A. F. (1998). The relationship between ensemble spread
  and ensemble mean skill. Monthly Weather Review, 126(12), 3292-3302.
  https://api.crossref.org/works/10.1175/1520-0493(1998)126%3C3292:TRBESA%3E2.0.CO;2
- [zarnowitz1987] Zarnowitz, V., & Lambros, L. A. (1987). Consensus and uncertainty in economic
  prediction. Journal of Political Economy, 95(3), 591-621. doi:10.1086/261473.
  https://ideas.repec.org/a/ucp/jpolec/v95y1987i3p591-621.html
