"""Post-hoc checks declared after the frozen test run (2026-10-11). Nothing here feeds back into
the frozen study; no model is fit on, or chosen with, test data.

tails       VALIDATION. Share of actuals below `lower` / above `upper` per system; a 10-90% band
            should give 10% on each side. Upper-tail misses are what drive lost sales.
learners    VALIDATION. The all-features control refit with other learners: same features, one
            model per horizon, same early-stopping scheme (pick rounds on the last
            `early_stop_days` of train, refit on all of train). Settings mirror the LightGBM block
            and are NOT tuned per learner, so this compares learners, not tuning effort.
quadrants   validation and test, from the STORED outputs of the frozen run (architecture.md 4.4):
            D x worst-error quadrants with the frozen tau.
seeds       VALIDATION. Systems 1-6 and H1 under extra LightGBM seeds (configs/posthoc.yaml); no
            LLM, so systems 7-9 are not rerun.
weather     VALIDATION. Models fit as in the study; validation predicted with observed vs the
            weather forecast Open-Meteo actually issued (historical-forecast archive).
censoring   validation and test, stored outputs: h=1 WAPE with and without days on which sales
            were capped by stock (India: generator stock-out flag; Rohlik: availability < 1).
itemaware   VALIDATION. Context agents that also see item identity, so an adjustment can differ
            by item (the frozen agents apply about the same adjustment to every item). Otherwise
            fit as in the study, tag `itemaware`. No LLM. Compares the independence gate, accuracy
            and H1 with the frozen agents.
independent VALIDATION. Context agents that forecast demand DIRECTLY from item identity and
            their own evidence (no sales history, no shared baseline), so their disagreement is
            between independent views. Tag `independent`. No LLM. Same table as itemaware.
deliberation VALIDATION. The study's critic + one revision round on a stratified validation
            sample, for a given agent set (tag) and LLM model, written to posthoc/ only (not the
            study DB). Usage: deliberation TAG MODEL N, e.g. `deliberation independent
            llama3.1 600` or `deliberation phase2 gemma4:31b-cloud 300`.
attribution India, validation sample and stored test outputs, no new LLM calls. The simulator
            logs which multiplier drove demand. Each context agent sees one driver (Weather:
            weather; Calendar: festival x payday; Trend: interest shock). Does the critic rate the
            true driver's agent "strong" more often than the others, and does the revision give
            it more weight? Also scores the per-case files written by `deliberation`.
chronos     VALIDATION. Zero-shot foundation-model baseline (Chronos-Bolt, CHRONOS_MODEL): for
            each validation origin, each item's sales up to the origin (missing days NaN) -> 7-day
            median and 10/90% quantiles. Scored against the stored validation systems. Needs the
            `posthoc` extra (torch, chronos-forecasting).
sameCases   VALIDATION. Every `deliberation` run scored on its own cases against the frozen
            systems on the SAME cases (Historical agent, frozen consensus, all-features LightGBM,
            ETS), so a gain cannot come from a worse starting consensus.
inference   test, from STORED outputs (cached LLM answers replayed offline, nothing refit or
            tuned): equivalence tests (TOST, 90% item-bootstrap CI inside +/- margin) for the
            "matches" claims, minimum detectable effects, an item-clustered H2 test, and Holm
            corrections. Margins (EQ_MARGINS) are fixed post hoc, so several are reported.
router      VALIDATION. A learned trigger instead of D >= tau, from the stored always-on
            deliberation sample (phase3b, no new LLM calls). Inputs known before any LLM call:
            D, D_range, Historical band width, volume. Time-blocked CV: fit on four blocks of
            days, rank the fifth; compare with ranking by D, random, and the oracle (true gain)
            at equal rates.

Run: python -m retailsense.eval.posthoc [STEP ...]   (default: all steps above)
     RETAILSENSE_CONFIG selects the dataset.
"""

import json
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance

from retailsense.agents.calendar import calendar_agent
from retailsense.agents.historical import (
    FEATURES,
    HistoricalAgent,
    expanding_oof,
    fit_early_stopped,
    lgb_params,
)
from retailsense.agents.trend import trend_agent
from retailsense.agents.weather import weather_agent
from retailsense.config import ROOT, ExperimentCfg, load_experiment, load_inventory
from retailsense.data.features import ITEM_FEATURES
from retailsense.data.weather import apply_weather_error, fetch
from retailsense.eval import run_experiment
from retailsense.eval.deliberation_eval import build_cases, evaluate, run_all, stratified_sample
from retailsense.eval.final import Window
from retailsense.eval.h1_analysis import added_value, load_points, signal_stats
from retailsense.eval.inventory_eval import system_forecasts
from retailsense.eval.metrics import wape
from retailsense.eval.run_experiment import run_validation
from retailsense.eval.stats import diebold_mariano, paired_bootstrap
from retailsense.orchestrator.critic import agent_view
from retailsense.orchestrator.llm import LLM, offline_transport, ollama_transport
from retailsense.orchestrator.revise import Revision, apply_revision


# ---------------------------------------------------------------- tails
def tail_rates(e: pd.DataFrame) -> dict:
    """e has y, lower, upper. Shares below the band, above it, and inside it."""
    below, above = (e.y < e.lower).mean(), (e.y > e.upper).mean()
    return {"below_lower": below, "above_upper": above, "coverage": 1 - below - above, "n": len(e)}


def scored_systems(cfg: ExperimentCfg, sales: pd.DataFrame, tag: str, first: int, last: int):
    """Stored forecasts of systems 1-6 for targets in [first, last], joined to actuals."""
    actual = sales.set_index(["item_id", "d"]).sales
    for name, fc in system_forecasts(cfg, load_inventory(), sales, tag, cfg.data.train_end).items():
        e = fc.assign(d=fc.origin + fc.h)
        e = e[(e.d >= first) & (e.d <= last)]
        e = e.assign(y=actual.reindex(pd.MultiIndex.from_arrays([e.item_id, e.d])).to_numpy())
        yield name, e.dropna(subset=["y"])  # a day with no sales record is unknown, not zero


def tails(cfg: ExperimentCfg, sales: pd.DataFrame) -> pd.DataFrame:
    d = cfg.data
    rows = []
    for name, e in scored_systems(cfg, sales, "phase2", d.train_end + 1, d.val_end):
        for scope, part in (("h1", e[e.h == 1]), ("h1-7", e)):
            rows.append({"system": name, "scope": scope, **tail_rates(part)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- censored demand
def censored_days(cfg: ExperimentCfg) -> pd.Series:
    """(item_id, d) -> True where sales were capped by stock, so sales < demand.
    India: the generator's own stock-out flag. Rohlik: availability < 1 in the raw file."""
    if cfg.source is None:
        g = pd.read_parquet(cfg.path("raw") / "ground_truth.parquet")
        g = g[g.store_id == cfg.data.store]
        return g.set_index(["item_id", "d"]).stockout.astype(bool)
    src = cfg.source
    s = pd.read_csv(ROOT / src.dir / "sales_train.csv", parse_dates=["date"])
    s = s[s.warehouse == src.warehouse]
    names = pd.read_csv(ROOT / src.dir / "inventory.csv")
    names = names[names.warehouse == src.warehouse].set_index("unique_id").name
    s = s.assign(item_id=s.unique_id.map(names), d=(s.date - pd.Timestamp(src.start)).dt.days + 1)
    return s.set_index(["item_id", "d"]).availability.lt(1)


def censoring(cfg: ExperimentCfg, sales: pd.DataFrame) -> pd.DataFrame:
    """h=1 WAPE on all days vs only uncensored days, validation and stored test outputs."""
    d, cens = cfg.data, censored_days(cfg)
    rows = []
    for split, tag, first, last in (
        ("validation", "phase2", d.train_end + 1, d.val_end),
        ("test", "final_test", d.val_end + 1, d.test_end),
    ):
        for name, e in scored_systems(cfg, sales, tag, first, last):
            e = e[e.h == 1]
            c = cens.reindex(pd.MultiIndex.from_arrays([e.item_id, e.d]), fill_value=False)
            ok = ~c.to_numpy()
            rows.append({
                "split": split,
                "system": name,
                "censored_share": 1 - ok.mean(),
                "h1_WAPE_all": wape(e.y, e.yhat),
                "h1_WAPE_uncensored": wape(e.y[ok], e.yhat[ok]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- D x error quadrants
def quadrant_table(pts: pd.DataFrame, tau: float, share: float) -> pd.DataFrame:
    """architecture.md 4.4: D measures spread, not correctness. Rows: high/low D (D >= tau) x
    worst-`share` relative error. `same_sign` = every agent misses on the same side of y, i.e. a
    shared blind spot that no disagreement score can see."""
    f = pts.filter(like="f_")
    same = (f.lt(pts.y, axis=0).all(axis=1)) | (f.gt(pts.y, axis=0).all(axis=1))
    q = pts.assign(
        D_group=np.where(pts.D >= tau, "high_D", "low_D"),
        err_group=np.where(pts.rel_err >= pts.rel_err.quantile(1 - share), "high_err", "low_err"),
        same_sign=same,
    )
    t = q.groupby(["D_group", "err_group"]).agg(
        n=("D", "size"), mean_rel_err=("rel_err", "mean"), same_sign=("same_sign", "mean")
    )
    t["share_of_points"] = t.n / len(q)
    t["share_of_worst"] = t.n / (q.err_group == "high_err").sum()
    t.loc[(slice(None), "low_err"), "share_of_worst"] = np.nan
    return t


def quadrants(cfg: ExperimentCfg) -> pd.DataFrame:
    """Validation and test points of the frozen run. Reads stored outputs only."""
    d, tau = cfg.data, cfg.deliberation.tau
    parts = []
    for split, tag, first, last in (
        ("validation", "phase2", d.train_end + 1, d.val_end),
        ("test", "final_test", d.val_end + 1, d.test_end),
    ):
        pts = load_points(cfg, tag, first, last)
        parts.append(quadrant_table(pts, tau, cfg.analysis.worst_share).assign(split=split))
    return pd.concat(parts).reset_index().set_index(["split", "D_group", "err_group"])


# ---------------------------------------------------------------- model seeds
def seed_run(cfg: ExperimentCfg, seed: int) -> str:
    """Systems 1-6 refit on train with another LightGBM seed, scored on validation, under tag
    seed<k>. No LLM. run_validation also rewrites oof_baseline.parquet; the frozen copy is put
    back afterwards."""
    tag = f"seed{seed}"
    proc = cfg.path("processed")
    if not (proc / f"{tag}_agent_forecasts.parquet").exists():
        oof = proc / "oof_baseline.parquet"
        frozen = oof.read_bytes()
        try:
            run_validation(cfg.model_copy(update={"seed": seed}), 2, tag=tag, log=False)
        finally:
            oof.write_bytes(frozen)
    return tag


def seeds(cfg: ExperimentCfg, sales: pd.DataFrame) -> pd.DataFrame:
    d = cfg.data
    extra = yaml.safe_load((ROOT / "configs/posthoc.yaml").read_text())["seeds"]
    rows = []
    for seed in [cfg.seed, *extra]:
        tag = "phase2" if seed == cfg.seed else seed_run(cfg, seed)
        row = {"seed": seed}
        for name, e in scored_systems(cfg, sales, tag, d.train_end + 1, d.val_end):
            h1 = e[e.h == 1]
            row[name] = wape(h1.y, h1.yhat)
        pts = load_points(cfg, tag)
        worst = (pts.rel_err >= pts.rel_err.quantile(1 - cfg.analysis.worst_share)).to_numpy()
        row |= {f"H1_{k}": v for k, v in signal_stats(pts, "D", worst).items()}
        rows.append(row)
    t = pd.DataFrame(rows).set_index("seed")
    t["gap_multi_agent_vs_lgbm"] = t.multi_agent_confidence - t.lgbm_all_features
    return pd.concat([t, t.agg(["mean", "std"])])


# ---------------------------------------------------------------- item-aware context agents
def item_aware_factory(factory):
    """Wrap an agent factory so the agent also sees item identity. Item identity is an input, not
    evidence, so it is left out of the key factors (as the Historical agent does)."""
    def make(cfg: ExperimentCfg):
        agent = factory(cfg)
        agent.features = [*agent.features, *ITEM_FEATURES]
        fit = agent.fit

        def fit_hiding_item(oof):
            fit(oof)
            agent.importance = agent.importance.drop(ITEM_FEATURES, errors="ignore")
            return agent

        agent.fit = fit_hiding_item
        return agent
    return make


def control_without_duplicates(cfg: ExperimentCfg, features: list[str] | None = None):
    """run_validation builds the all-features control from FEATURES plus each agent's inputs;
    with item-aware agents that lists item_id twice. Drop the repeats so the control's inputs stay
    exactly the frozen ones."""
    if features is None:
        return HistoricalAgent(cfg)
    return HistoricalAgent(cfg, list(dict.fromkeys(features)))


class IndependentAgent:
    """A context agent that forecasts demand itself from item identity and its own evidence (no
    sales lags, no baseline input), with its own quantile band. It keeps ResidualAgent's
    interface, reporting adjustment = own forecast - Historical forecast, so run_validation's
    wiring (confidence track record, key factors) is unchanged."""

    def __init__(self, base):
        self.agent_id, self.cfg = base.agent_id, base.cfg
        self.evidence = base.features[1:]
        self.features = ["baseline", *self.evidence]  # run_validation reads features[1:]
        self.inputs = [*ITEM_FEATURES, *self.evidence]
        self.model = HistoricalAgent(self.cfg, self.inputs)

    def _point(self, df):
        params = lgb_params(self.cfg, "historical", "tweedie")
        return fit_early_stopped(df, self.inputs, "sales", params, self.cfg)

    def fit(self, oof: pd.DataFrame) -> "IndependentAgent":
        self.model.fit(oof)
        self.importance = self.model.importance.drop(ITEM_FEATURES, errors="ignore")
        return self

    def apply(self, base: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
        own = self.model.predict(feats)
        out = base[["item_id", "d", "h"]].copy()
        for col in ("forecast", "lower", "upper"):
            out[col] = own[col].to_numpy()
        out["adjustment"] = out.forecast.to_numpy() - base.forecast.to_numpy()
        return out

    def oof_adjustment(self, oof: pd.DataFrame) -> pd.DataFrame:
        """The agent's own out-of-sample record before validation (pooled over horizons; the
        evidence is target-day information, so it does not depend on h)."""
        out = expanding_oof(
            oof.drop(columns=["fold", "fold_train_end"]),
            self.cfg.agents.oof_folds,
            self._point,
            lambda m, d: np.clip(m.predict(d[self.inputs]), 0, None) - d.baseline.to_numpy(),
        )
        return out.rename(columns={"oof_pred": "adjustment"})

    def key_factors(self, row: pd.Series, adjustment: float) -> list[str]:
        top = list(self.importance.index[: self.cfg.lgbm.key_factors])
        facts = [f"{f} = {row[f]}" if pd.notna(row[f]) else f"{f} none" for f in top]
        return [f"{self.agent_id} forecast {row.forecast:.2f}", *facts]


def independent_factory(factory):
    return lambda cfg: IndependentAgent(factory(cfg))


def variant_run(cfg: ExperimentCfg, tag: str, wrap) -> str:
    """Systems 1-6 on validation with every context agent wrapped by `wrap`, under `tag`. The
    frozen oof_baseline.parquet is put back afterwards, as in seed_run."""
    proc = cfg.path("processed")
    if not (proc / f"{tag}_agent_forecasts.parquet").exists():
        oof = proc / "oof_baseline.parquet"
        frozen = oof.read_bytes()
        try:
            with mock.patch.multiple(
                run_experiment,
                calendar_agent=wrap(calendar_agent),
                weather_agent=wrap(weather_agent),
                trend_agent=wrap(trend_agent),
                HistoricalAgent=control_without_duplicates,
            ):
                run_validation(cfg, 2, tag=tag, log=False)
        finally:
            oof.write_bytes(frozen)
    return tag


def item_aware_run(cfg: ExperimentCfg) -> str:
    return variant_run(cfg, "itemaware", item_aware_factory)


def gate_and_h1(cfg: ExperimentCfg, sales: pd.DataFrame, tag: str) -> dict:
    """Independence gate, per-agent and per-system h=1 WAPE, H1 and D's added value, one tag."""
    d = cfg.data
    pts = load_points(cfg, tag)
    f = pts.filter(like="f_").rename(columns=lambda c: c[2:])
    corr = f.corr().to_numpy()[np.triu_indices(f.shape[1], 1)]
    adj = f.drop(columns="historical").sub(f.historical, axis=0)
    worst = (pts.rel_err >= pts.rel_err.quantile(1 - cfg.analysis.worst_share)).to_numpy()
    av = added_value(pts, worst).auroc_worst
    row = {
        "max_pairwise_corr": corr.max(),
        "min_pairwise_corr": corr.min(),
        "mean_abs_adjustment": adj.abs().mean().mean(),
        **{f"H1_{k}": v for k, v in signal_stats(pts, "D", worst).items()},
        "auroc_band_volume": av["band + volume"],
        "auroc_band_volume_D": av["band + volume + D"],
        **{f"agent_{a}": wape(pts.y, f[a]) for a in f},
    }
    for name, e in scored_systems(cfg, sales, tag, d.train_end + 1, d.val_end):
        h1 = e[e.h == 1]
        row[name] = wape(h1.y, h1.yhat)
    return row


def hist_error_row(cfg: ExperimentCfg, tag: str) -> dict:
    """Stored outputs of one agent set: within-item agent correlation (pooled correlation after
    removing each item's mean; range over agent pairs) and D's added value as a flag for the
    HISTORICAL agent's worst errors (band + volume -> + D)."""
    pts = load_points(cfg, tag)
    rel = (pts.y - pts.f_historical).abs() / pts.y.clip(lower=1)
    worst = (rel >= rel.quantile(1 - cfg.analysis.worst_share)).to_numpy()
    av = added_value(pts, worst).auroc_worst
    f = pts.set_index("item_id").filter(like="f_")
    within = (f - f.groupby(level=0).transform("mean")).corr().to_numpy()
    within = within[np.triu_indices(f.shape[1], 1)]
    return {"within_item_corr_min": within.min(), "within_item_corr_max": within.max(),
            "hist_auroc_band_volume": av["band + volume"],
            "hist_auroc_band_volume_D": av["band + volume + D"]}


def item_aware(cfg: ExperimentCfg, sales: pd.DataFrame) -> pd.DataFrame:
    rows = {"frozen": gate_and_h1(cfg, sales, "phase2")}
    rows["item_aware"] = gate_and_h1(cfg, sales, item_aware_run(cfg))
    return pd.DataFrame(rows)


def independent(cfg: ExperimentCfg, sales: pd.DataFrame) -> pd.DataFrame:
    rows = {"frozen": gate_and_h1(cfg, sales, "phase2")}
    rows["independent"] = gate_and_h1(cfg, sales, variant_run(cfg, "independent",
                                                               independent_factory))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- post-hoc deliberation runs
FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def schema_in_prompt(transport):
    """Cloud models behind Ollama ignore `format` (no constrained decoding), which the local
    model had. Give them the same JSON schema as text after the system prompt and strip a
    markdown fence from the reply. The v1 prompt text itself is unchanged."""
    def call(system: str, user: str, schema):
        spec = json.dumps(schema.model_json_schema())
        rule = "Return ONLY a JSON object that validates against this JSON Schema:"
        system = f"{system}\n\n{rule}\n{spec}"
        text, *rest = transport(system, user, schema)
        return (FENCE.sub("", text), *rest)
    return call


def deliberation(cfg: ExperimentCfg, tag: str, model: str, n: int) -> tuple:
    """The frozen critic + revision prompt (v1) on a stratified validation sample of agent set
    `tag`, with LLM `model`. tau is the 80th percentile of THIS agent set's validation D (the
    documented calibration). The sample is drawn as the study drew its 1,500 cases, then n of
    them at random, so for tag phase2 every case is one the frozen llama3.1 run also saw."""
    dcfg = cfg.deliberation.model_copy(update={"model": model})
    c = cfg.model_copy(update={"deliberation": dcfg})
    pts = load_points(c, tag)
    tau = float(pts.D.quantile(c.disagreement.tau_quantile))
    sample = stratified_sample(pts, c.deliberation.sample_size, c.seed)
    sample = sample.sample(min(n, len(sample)), random_state=c.seed).reset_index(drop=True)
    cases = build_cases(c, sample, tag)
    print(f"{len(cases)} cases, agents {tag}, model {model}, tau {tau:.4f}", flush=True)
    llm = LLM(c, transport=schema_in_prompt(ollama_transport(c))) if "-cloud" in model else None
    fast = run_all(c, cases, tau, trigger=False, llm=llm)
    deli = run_all(c, cases, tau, trigger=True, llm=llm)
    arms, curve, h2 = evaluate(c, sample, deli, fast, tau)
    per_case = case_judgements(deli)
    meta = pd.Series({**h2, "tau": tau, "n": len(cases), "model": model, "tag": tag})
    finals = sample[["item_id", "d", "y", "D", "consensus"]].assign(
        final=[s["final"] for s in deli], triggered=(sample.D >= tau).to_numpy()
    )
    return arms, curve, meta, per_case, finals


def no_critic_case(llm: LLM, case: dict, m_bounds) -> tuple[float | None, list[dict]]:
    """Ablation V-10: one bounded revision round WITHOUT the critic. Each agent sees the other
    agents' views instead of a critic report (prompt revise_nocritic_v1); clipping and the
    final combination are the study's own apply_revision. None: an agent's answer was invalid."""
    views = {a.agent_id: agent_view(a) for a in case["agents"]}

    def check(r: Revision) -> None:
        bad = set(r.cited_evidence) - set(views)
        if bad:
            raise ValueError(f"cited evidence is not an agent in this case: {sorted(bad)}")
        if not np.isfinite([r.confidence_multiplier, r.revised_forecast]).all():
            raise ValueError("non-finite number")

    revs = {}
    for a in case["agents"]:
        payload = {"dataset": case["dataset"], "you": views[a.agent_id],
                   "other_agents": [v for k, v in views.items() if k != a.agent_id]}
        res = llm.call("revise_nocritic", payload, Revision, check)
        if res.value is None:
            return None, []
        revs[a.agent_id] = res.value
    return apply_revision(case["agents"], revs, m_bounds)


def no_critic(cfg: ExperimentCfg, n: int) -> pd.DataFrame:
    """V-10 on the n validation cases of `deliberation phase2 <frozen model> n`: the with-critic
    arm replays from the LLM cache; the no-critic arm differs only in the missing critic."""
    *_, meta, _, fin = deliberation(cfg, "phase2", cfg.deliberation.model, n)
    cases = build_cases(cfg, fin, "phase2")
    llm = LLM(cfg)
    with ThreadPoolExecutor(cfg.deliberation.workers) as pool:
        out = list(pool.map(lambda c: no_critic_case(llm, c, cfg.deliberation.m_bounds), cases))
    nc = np.array([np.nan if f is None else f for f, _ in out])
    ok = ~np.isnan(nc)
    nc = np.where(ok, nc, fin.consensus)  # invalid -> fast consensus, as the study falls back
    rows = [r for _, rs in out for r in rs]
    trig = fin.triggered.to_numpy()
    arms = {
        "none": fin.consensus.to_numpy(),
        "critic_always_on": fin.final.to_numpy(),
        "nocritic_always_on": nc,
        "critic_selective": np.where(trig, fin.final, fin.consensus),
        "nocritic_selective": np.where(trig, nc, fin.consensus),
    }
    y, base = fin.y.to_numpy(), np.abs(fin.y - fin.consensus).to_numpy()
    t = pd.DataFrame({
        k: {"wape": wape(y, v), "improved": np.mean(np.abs(y - v) < base - 1e-9),
            "worsened": np.mean(np.abs(y - v) > base + 1e-9)}
        for k, v in arms.items()
    }).T
    t["unchanged"] = 1 - t.improved - t.worsened
    diff, boot = item_boot(fin.item_id, np.abs(y - nc), np.abs(y - fin.final), y,
                           cfg.analysis.bootstrap, cfg.seed)
    t.attrs.update({
        "n": len(fin), "tau": meta["tau"], "fallback_rate": 1 - ok.mean(),
        "hold_rate": np.mean([r["held"] for r in rows]),
        "clip_rate": np.mean([r["clipped"] for r in rows]),
        "nocritic_minus_critic": diff, "ci_lo": np.percentile(boot, 2.5),
        "ci_hi": np.percentile(boot, 97.5),
    })
    return t


def case_judgements(states: list[dict]) -> pd.DataFrame:
    """One row per (case, agent): the critic's strength judgement and the revision's multiplier."""
    rows = []
    for s in states:
        case = s["case"]
        strength = {e.agent: e.strength for e in s["critic"].evidence_assessment} if s.get(
            "critic") else {}
        for r in s.get("revisions") or []:
            rows.append({
                "item_id": case["item_id"],
                "d": case["origin"] + 1,
                "agent": r["agent_id"],
                "strength": strength.get(r["agent_id"]),
                "m": r["m"],
                "held": r["held"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- critic vs the true driver (India)
DRIVERS = {"weather": ["weather"], "calendar": ["festival", "payday"], "trend": ["shock"]}
DRIVER_MIN = np.log(1.1)  # a driver counts if it moves demand by at least 10%


def true_drivers(cfg: ExperimentCfg) -> pd.DataFrame:
    """(item_id, d) -> the context agent whose logged multiplier moves demand most, or 'none'."""
    g = pd.read_parquet(cfg.path("raw") / "ground_truth.parquet")
    g = g[g.store_id == cfg.data.store].astype({"item_id": str})
    size = pd.DataFrame({a: np.log(g[cols].prod(axis=1)).abs() for a, cols in DRIVERS.items()})
    driver = size.idxmax(axis=1).where(size.max(axis=1) >= DRIVER_MIN, "none")
    return g[["item_id", "d"]].assign(driver=driver.to_numpy(), driver_size=size.max(axis=1))


def db_judgements(cfg: ExperimentCfg, experiment: str) -> pd.DataFrame:
    """case_judgements for a deliberation run stored in the study DB (read only)."""
    con = sqlite3.connect(cfg.path("db"))
    runs = pd.read_sql(
        "select run_id, item_id, prediction_time from forecast_runs where experiment_id = ?",
        con, params=(experiment,))
    crit = pd.read_sql("select run_id, report_json from critic_reports", con)
    preds = pd.read_sql(
        "select run_id, agent_id as agent, confidence_multiplier as m from agent_predictions", con)
    con.close()
    dates = pd.read_parquet(cfg.path("processed") / "features.parquet", columns=["d", "date", "h"])
    dates = dates[dates.h == 1].drop_duplicates("d")
    dates = dates.set_index(dates.date.dt.strftime("%Y-%m-%d")).d
    runs = runs.assign(d=runs.prediction_time.map(dates))
    strength = {
        (r.run_id, e["agent"]): e["strength"]
        for r in crit.itertuples()
        for e in json.loads(r.report_json).get("evidence_assessment", [])
    }
    j = preds.merge(runs, on="run_id")
    j["strength"] = [strength.get((a, b)) for a, b in zip(j.run_id, j.agent, strict=True)]
    return j[["item_id", "d", "agent", "strength", "m"]]


def attribution_table(j: pd.DataFrame, drivers: pd.DataFrame) -> pd.Series:
    """For cases with a true driver: share of driver-agent vs other-context-agent judgements that
    are 'strong', mean multiplier m for each, and how often the driver agent is the single most
    up-weighted context agent (chance = 1/3 with three context agents)."""
    j = j[j.agent.isin(list(DRIVERS))].merge(drivers, on=["item_id", "d"])
    j = j[j.driver != "none"]
    is_driver = j.agent == j.driver
    top = j.loc[j.groupby(["item_id", "d"]).m.idxmax()]
    tied = j.groupby(["item_id", "d"]).m.transform(lambda m: (m == m.max()).sum() > 1)
    top = top[~tied.loc[top.index]]
    return pd.Series({
        "cases_with_driver": j[["item_id", "d"]].drop_duplicates().shape[0],
        "strong_share_driver_agent": (j.strength[is_driver] == "strong").mean(),
        "strong_share_other_agents": (j.strength[~is_driver] == "strong").mean(),
        "weak_share_driver_agent": (j.strength[is_driver] == "weak").mean(),
        "weak_share_other_agents": (j.strength[~is_driver] == "weak").mean(),
        "mean_m_driver_agent": j.m[is_driver].mean(),
        "mean_m_other_agents": j.m[~is_driver].mean(),
        "driver_is_most_upweighted": (top.agent == top.driver).mean(),
        "cases_with_a_single_most_upweighted": len(top),
        **paired_case_ci(j.assign(v=(j.strength == "strong").astype(float)), is_driver, "strong"),
        **paired_case_ci(j.assign(v=j.m), is_driver, "m"),
    })


def paired_case_ci(j: pd.DataFrame, is_driver: pd.Series, name: str, n: int = 1000) -> dict:
    """Per case: value of the driver's agent minus the mean of the other context agents; mean
    over cases with a 95% bootstrap CI (cases resampled)."""
    key = ["item_id", "d"]
    drv = j[is_driver].groupby(key).v.mean()
    oth = j[~is_driver].groupby(key).v.mean()
    diff = (drv - oth).dropna().to_numpy()
    idx = np.random.default_rng(0).integers(0, len(diff), size=(n, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(1), [2.5, 97.5])
    return {f"{name}_diff_driver_minus_others": diff.mean(), f"{name}_ci_lo": lo,
            f"{name}_ci_hi": hi}


def attribution(cfg: ExperimentCfg, out) -> pd.DataFrame:
    drivers = true_drivers(cfg)
    cols = {
        "llama3.1 validation (1,500, study DB)": db_judgements(cfg, "phase3b"),
        "llama3.1 test (3,360, study DB)": db_judgements(cfg, "final_test"),
    }
    for f in sorted(out.glob("val_deliberation_phase2_*_cases.parquet")):
        cols[f.stem.removeprefix("val_deliberation_phase2_").removesuffix("_cases")] = (
            pd.read_parquet(f))
    return pd.DataFrame({k: attribution_table(v, drivers) for k, v in cols.items()})


# ---------------------------------------------------------------- foundation-model baseline
CHRONOS_MODEL = "amazon/chronos-bolt-small"
CHRONOS_CONTEXT = 2048  # Chronos-Bolt's maximum context length


def chronos_forecasts(cfg: ExperimentCfg, sales: pd.DataFrame) -> pd.DataFrame:
    """sales: item_id, d, sales for days <= val_end. Context ends at each origin: no leakage."""
    import torch  # noqa: PLC0415
    from chronos import BaseChronosPipeline  # noqa: PLC0415

    pipe = BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL, device_map="cpu", torch_dtype=torch.float32
    )
    d, horizon = cfg.data, max(cfg.agents.horizons)
    wide = sales.pivot(index="d", columns="item_id", values="sales").sort_index()
    wide = wide.reindex(range(int(wide.index.min()), d.val_end + 1))
    parts = []
    for origin in range(d.train_end, d.val_end):
        ctx = wide.loc[:origin].tail(CHRONOS_CONTEXT)
        series = []
        for item in ctx.columns:
            v = ctx[item].to_numpy(float)
            series.append(torch.tensor(v[np.argmax(~np.isnan(v)):]))  # drop leading NaN
        q, _ = pipe.predict_quantiles(series, prediction_length=horizon,
                                      quantile_levels=[0.1, 0.5, 0.9])
        q = q.clamp(min=0).numpy()  # (items, horizon, 3)
        for h in range(1, horizon + 1):
            parts.append(pd.DataFrame({
                "item_id": ctx.columns, "origin": origin, "h": h, "d": origin + h,
                "yhat": q[:, h - 1, 1], "lower": q[:, h - 1, 0], "upper": q[:, h - 1, 2],
            }))
    return pd.concat(parts, ignore_index=True)


def chronos_table(cfg: ExperimentCfg, sales: pd.DataFrame) -> pd.DataFrame:
    d, a = cfg.data, cfg.analysis
    fc = chronos_forecasts(cfg, sales)
    actual = sales.set_index(["item_id", "d"]).sales
    fc = fc[(fc.d > d.train_end) & (fc.d <= d.val_end)]
    fc = fc.assign(y=actual.reindex(pd.MultiIndex.from_arrays([fc.item_id, fc.d])).to_numpy())
    systems = dict(scored_systems(cfg, sales, "phase2", d.train_end + 1, d.val_end))
    systems["chronos_bolt_small"] = fc.dropna(subset=["y"])
    keys = ["item_id", "origin", "h"]
    rows = {}
    for name, e in systems.items():
        h1 = e[e.h == 1]
        rows[name] = {
            "h1_WAPE": wape(h1.y, h1.yhat),
            "h1-7_WAPE": wape(e.y, e.yhat),
            "h1_coverage_10_90": ((h1.y >= h1.lower) & (h1.y <= h1.upper)).mean()
            if h1.lower.notna().any() else np.nan,
        }
    ch = systems["chronos_bolt_small"]
    ch = ch[ch.h == 1].set_index(keys).sort_index()
    for ref in ("lgbm_all_features", "multi_agent_confidence", "ets"):
        r = systems[ref]
        r = r[r.h == 1].set_index(keys).sort_index()
        both = ch.index.intersection(r.index)
        c, b = ch.loc[both], r.loc[both]
        ec, eb = (c.y - c.yhat).abs(), (b.y - b.yhat).abs()
        bt = paired_bootstrap(both.get_level_values(0), ec, eb, c.y, a.bootstrap, cfg.seed)
        day = both.get_level_values("origin")
        dm = diebold_mariano(ec.groupby(day).mean(), eb.groupby(day).mean())
        rows["chronos_bolt_small"] |= {
            f"diff_vs_{ref}": bt["diff"], f"ci_lo_vs_{ref}": bt["ci_lo"],
            f"ci_hi_vs_{ref}": bt["ci_hi"], f"DM_p_vs_{ref}": dm["p"],
        }
    return pd.DataFrame(rows).T


def same_cases(cfg: ExperimentCfg, out, sales: pd.DataFrame) -> pd.DataFrame:
    d, a = cfg.data, cfg.analysis
    frozen = {
        name: e[e.h == 1].set_index(["item_id", "d"]).yhat
        for name, e in scored_systems(cfg, sales, "phase2", d.train_end + 1, d.val_end)
        if name in ("historical", "multi_agent_confidence", "lgbm_all_features", "ets")
    }
    rows = {}
    for f in sorted(out.glob("val_deliberation_*_forecasts.parquet")):
        run = f.stem.removeprefix("val_deliberation_").removesuffix("_forecasts")
        r = pd.read_parquet(f).astype({"item_id": str}).set_index(["item_id", "d"])
        own = {
            "own: no deliberation": r.consensus,
            "own: always-on": r.final,
            "own: selective": r.final.where(r.triggered, r.consensus),
        }
        allf = {**own, **{f"frozen: {k}": v.reindex(r.index) for k, v in frozen.items()}}
        row = {k: wape(r.y, v) for k, v in allf.items()}
        ea = (r.y - r.final).abs()
        for ref in ("frozen: historical", "frozen: multi_agent_confidence",
                    "frozen: lgbm_all_features"):
            eb = (r.y - allf[ref]).abs()
            bt = paired_bootstrap(r.index.get_level_values(0), ea, eb, r.y, a.bootstrap, cfg.seed)
            row[f"always-on minus {ref} [CI]"] = (
                f"{bt['diff']:+.4f} [{bt['ci_lo']:+.4f}, {bt['ci_hi']:+.4f}]")
        rows[run] = row | {"cases": len(r)}
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- inference on stored test outputs
EQ_MARGINS = (0.005, 0.01, 0.02)  # equivalence margins, relative to the no-deliberation WAPE
PAIRS = {  # claim -> (system a, system b); diff = WAPE(a) - WAPE(b)
    "H3 selective vs always-on": ("retailsense", "always_on"),
    "H4 selective vs random": ("retailsense", "random_trigger"),
    "always-on vs none": ("always_on", "multi_agent_confidence"),
    "selective vs none": ("retailsense", "multi_agent_confidence"),
}


def item_boot(items, ea, eb, y, n: int, seed: int) -> tuple[float, np.ndarray]:
    """WAPE(a) - WAPE(b) and its item-bootstrap draws (same scheme as stats.paired_bootstrap)."""
    g = np.unique(np.asarray(items), return_inverse=True)[1]
    k = g.max() + 1
    sa, sb, sy = (np.bincount(g, np.asarray(x, float), k) for x in (ea, eb, y))
    idx = np.random.default_rng(seed).integers(0, k, size=(n, k))
    return (sa.sum() - sb.sum()) / sy.sum(), (sa[idx].sum(1) - sb[idx].sum(1)) / sy[idx].sum(1)


def equivalence_row(items, ea, eb, y, ref_wape: float, seed_sd: float, n: int, seed: int) -> dict:
    """TOST by the CI method: equivalent at margin m if the 90% CI lies inside +/- m * ref_wape.
    MDE: the smallest true difference detected with 80% power at two-sided alpha 0.05."""
    diff, boot = item_boot(items, ea, eb, y, n, seed)
    lo90, hi90 = np.percentile(boot, [5, 95])
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    se = boot.std(ddof=1)
    mde = (norm.ppf(0.975) + norm.ppf(0.8)) * se
    row = {
        "diff": diff,
        "ci95_lo": lo95,
        "ci95_hi": hi95,
        "ci90_lo": lo90,
        "ci90_hi": hi90,
        "se": se,
        "MDE_80pct_power": mde,
        "MDE_in_seed_sd": mde / seed_sd,
        "diff_in_seed_sd": diff / seed_sd,
    }
    for m in EQ_MARGINS:
        row[f"equivalent_at_{m:.1%}"] = bool(-m * ref_wape < lo90 and hi90 < m * ref_wape)
    return row


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    order = p.sort_values()
    adj = (order * np.arange(len(order), 0, -1)).cummax().clip(upper=1)
    return adj.reindex(p.index)


def test_window(cfg: ExperimentCfg) -> Window:
    """The frozen test window rebuilt from stored forecasts and the LLM cache (offline: a cache
    miss raises instead of calling the API)."""
    d = cfg.data
    llm = LLM(cfg, transport=offline_transport(cfg))
    tau = cfg.deliberation.tau
    return Window(
        cfg, load_inventory(), llm, "final_test", d.train_end, d.val_end, d.test_end, tau, True
    )


def clustered_h2(w: Window, n: int, seed: int) -> dict:
    """H2 with cases clustered by item: resample items, mean gain over their triggered cases."""
    t = w.sel & w.ok
    p, r = w.pts[t], w.res[t]
    before, after = (p.y - r.consensus).abs().to_numpy(), (p.y - r.final).abs().to_numpy()
    gain = before - after
    g = np.unique(p.item_id, return_inverse=True)[1]
    k = g.max() + 1
    s, c = np.bincount(g, gain, k), np.bincount(g, minlength=k)
    idx = np.random.default_rng(seed).integers(0, k, size=(n, k))
    boot = s[idx].sum(1) / c[idx].sum(1)
    return {
        "wilcoxon_p": wilcoxon(before, after).pvalue,
        "clustered_p": min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())),
        "mean_gain": gain.mean(),
        "ci_lo": np.percentile(boot, 2.5),
        "ci_hi": np.percentile(boot, 97.5),
        "MDE_80pct_power": (norm.ppf(0.975) + norm.ppf(0.8)) * boot.std(ddof=1),
    }


def inference(cfg: ExperimentCfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = cfg.analysis
    w = test_window(cfg)
    h1 = {k: v[v.h == 1].reset_index(drop=True) for k, v in w.evals.items()}
    base = h1["multi_agent_confidence"]
    for v in h1.values():
        if not (v[["item_id", "d"]].to_numpy() == base[["item_id", "d"]].to_numpy()).all():
            raise RuntimeError("systems are not aligned point by point")
    ref = wape(base.y, base.yhat)
    seeds = pd.read_csv(ROOT / "reports" / cfg.dataset / "posthoc" / "val_seeds.csv", index_col=0)
    seed_sd = float(seeds.loc["std", "multi_agent_confidence"])
    rows = {}
    for claim, (sa, sb) in PAIRS.items():
        ea, eb = (h1[sa].y - h1[sa].yhat).abs(), (h1[sb].y - h1[sb].yhat).abs()
        rows[claim] = equivalence_row(
            base.item_id, ea, eb, base.y, ref, seed_sd, a.bootstrap, cfg.seed
        )
    eq = pd.DataFrame(rows).T

    h2 = clustered_h2(w, a.bootstrap, cfg.seed)
    sig = pd.read_csv(
        ROOT / "reports" / cfg.dataset / "final" / "table_significance.csv", index_col=0
    )  # the frozen run's DM tests (on daily mean absolute error)
    worst = (w.pts.rel_err >= w.pts.rel_err.quantile(1 - a.worst_share)).to_numpy()
    confirm = pd.DataFrame(
        {
            "H1 Spearman(D, error)": {
                "p": spearmanr(w.pts.D, w.pts.rel_err).pvalue,
                "effect": signal_stats(w.pts, "D", worst)["auroc_worst"],
            },
            "H2 item-clustered": {"p": h2["clustered_p"], "effect": h2["mean_gain"]},
            "H4 DM selective vs random": {
                "p": sig.loc["random_trigger", "DM_p"],
                "effect": eq.loc["H4 selective vs random", "diff"],
            },
        }
    ).T
    confirm["p_holm"] = holm(confirm.p)
    systems = pd.DataFrame({"p": sig.DM_p, "effect": sig.WAPE_diff, "p_holm": holm(sig.DM_p)})
    systems.index = [f"DM RetailSense vs {s}" for s in systems.index]
    tests = pd.concat([confirm.assign(family="confirmatory"), systems.assign(family="systems")])
    eq.attrs.update(ref_wape=ref, seed_sd=seed_sd, h2=h2)
    return eq, tests


# ---------------------------------------------------------------- learned trigger
ROUTER_INPUTS = ["D", "D_range", "hist_band", "low_volume"]


def deliberation_sample(cfg: ExperimentCfg) -> pd.DataFrame:
    """Validation points that the always-on arm deliberated, with consensus and deliberated value.
    gain > 0: deliberation lowered the absolute error."""
    con = sqlite3.connect(cfg.path("db"))
    runs = pd.read_sql(
        "select item_id, prediction_time, final_forecast from forecast_runs "
        "where experiment_id = 'phase3b' and system = 'always_on' and horizon = 1",
        con,
    )
    con.close()
    dates = pd.read_parquet(cfg.path("processed") / "features.parquet", columns=["d", "date", "h"])
    dates = dates[dates.h == 1].drop_duplicates("d")
    dates = dates.set_index(dates.date.dt.strftime("%Y-%m-%d")).d
    runs = runs.assign(d=runs.prediction_time.map(dates)).drop(columns="prediction_time")
    pts = load_points(cfg).merge(runs, on=["item_id", "d"], validate="1:1")
    return pts.assign(gain=(pts.y - pts.consensus).abs() - (pts.y - pts.final_forecast).abs())


def routed_wape(s: pd.DataFrame, pick: np.ndarray) -> float:
    return wape(s.y, np.where(pick, s.final_forecast, s.consensus))


def router_cv(s: pd.DataFrame, rates: list[float], folds: int, seed: int, draws: int = 200,
              inputs: list[str] = ROUTER_INPUTS):
    """Time-blocked CV. Within each held-out block, trigger the top `rate` share by the score."""
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB  # noqa: PLC0415

    s = s.sort_values("d").reset_index(drop=True)
    block = pd.qcut(s.d.rank(method="dense"), folds, labels=False).to_numpy()
    score = np.empty(len(s))
    for k in range(folds):
        fit, held = block != k, block == k
        m = HGB(max_iter=100, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=40,
                random_state=seed).fit(s.loc[fit, inputs], s.gain[fit])
        score[held] = m.predict(s.loc[held, inputs])
    rng = np.random.default_rng(seed)
    rows = []
    for rate in rates:
        def top(v, rate=rate):
            pick = np.zeros(len(s), bool)
            for k in range(folds):
                idx = np.flatnonzero(block == k)
                n = int(round(rate * len(idx)))
                pick[idx[np.argsort(-v[idx], kind="stable")[:n]]] = True
            return pick
        rand = [routed_wape(s, top(rng.random(len(s)))) for _ in range(draws)]
        rows.append({
            "rate": rate,
            "learned": routed_wape(s, top(score)),
            "by_D": routed_wape(s, top(s.D.to_numpy())),
            "random_mean": np.mean(rand),
            "random_sd": np.std(rand),
            "oracle": routed_wape(s, top(s.gain.to_numpy())),
        })
    t = pd.DataFrame(rows).set_index("rate")
    t.attrs["spearman_score_gain"] = pd.Series(score).corr(s.gain, method="spearman")
    t.attrs["spearman_D_gain"] = s.D.corr(s.gain, method="spearman")
    return t


def router(cfg: ExperimentCfg) -> pd.DataFrame:
    s = deliberation_sample(cfg)
    t = router_cv(s, [0.1, 0.2, 0.3, 0.5], folds=5, seed=cfg.seed)
    t.loc["none"] = routed_wape(s, np.zeros(len(s), bool))
    t.loc["always_on"] = routed_wape(s, np.ones(len(s), bool))
    print(
        f"n = {len(s)}; deliberation helped {np.mean(s.gain > 0):.1%}, "
        f"hurt {np.mean(s.gain < 0):.1%}; "
        f"Spearman(gain, learned score) = {t.attrs['spearman_score_gain']:.3f}, "
        f"Spearman(gain, D) = {t.attrs['spearman_D_gain']:.3f}"
    )
    return t


def signal_triggers(s: pd.DataFrame, rate: float, worst_share: float, seed: int,
                    draws: int = 1000) -> pd.DataFrame:
    """Each pre-call signal as a trigger on the always-on sample: does it flag large errors
    (AUROC, worst share), does it predict the deliberation gain (Spearman), and does deliberating
    its top `rate` beat a random trigger with the same number of calls (z vs random draws)."""
    worst = (s.rel_err >= s.rel_err.quantile(1 - worst_share)).to_numpy()
    n = int(round(rate * len(s)))
    rng = np.random.default_rng(seed)

    def pick(v):
        p = np.zeros(len(s), bool)
        p[np.argsort(-np.asarray(v), kind="stable")[:n]] = True
        return p

    rand = np.array([routed_wape(s, pick(rng.random(len(s)))) for _ in range(draws)])
    rows = {}
    for sig in [*ROUTER_INPUTS, "hist_err"]:
        p, w = pick(s[sig]), routed_wape(s, pick(s[sig]))
        rows[sig] = {
            **signal_stats(s, sig, worst),
            "spearman_gain": s[sig].corr(s.gain, method="spearman"),
            "wape": w,
            "z_vs_random": (w - rand.mean()) / rand.std(),  # < 0: better than random
            "improved": np.mean(s.gain[p] > 0),
            "worsened": np.mean(s.gain[p] < 0),
            "unchanged": np.mean(s.gain[p] == 0),
        }
    t = pd.DataFrame(rows).T
    t.loc["random"] = {"wape": rand.mean(), "z_vs_random": 0.0}
    t.loc["none"] = {"wape": routed_wape(s, np.zeros(len(s), bool))}
    t.loc["always_on"] = {"wape": routed_wape(s, np.ones(len(s), bool)),
                          "improved": np.mean(s.gain > 0), "worsened": np.mean(s.gain < 0),
                          "unchanged": np.mean(s.gain == 0)}
    return t


def signals(cfg: ExperimentCfg) -> pd.DataFrame:
    """Validation only. Adds the Historical agent's trailing error (from its confidence
    c = 1 / (1 + trailing 28-day WAPE), known before any call) to the router's candidates."""
    s = deliberation_sample(cfg).assign(hist_err=lambda x: 1 / x.c_historical - 1)
    rate = round(1 - cfg.disagreement.tau_quantile, 6)  # the study's call budget
    t = signal_triggers(s, rate, cfg.analysis.worst_share, cfg.seed)
    cv = router_cv(s, [rate], folds=5, seed=cfg.seed, inputs=[*ROUTER_INPUTS, "hist_err"])
    t.loc["learned_all", ["wape", "spearman_gain"]] = (
        cv.learned.iloc[0], cv.attrs["spearman_score_gain"])
    t.loc["learned_all", "z_vs_random"] = (cv.learned.iloc[0] - cv.random_mean.iloc[0]) / (
        cv.random_sd.iloc[0])
    t.loc["oracle", "wape"] = cv.oracle.iloc[0]
    t.attrs["rate"] = rate
    return t


# ---------------------------------------------------------------- issued weather forecasts
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def issued_weather(cfg: ExperimentCfg, feats: pd.DataFrame) -> pd.DataFrame:
    """The frozen study feeds OBSERVED target-day weather (a perfect forecast). Here the models are
    fit exactly as in the study (observed history), then validation is predicted twice: with
    observed weather and with the forecast Open-Meteo actually issued for each date. The issued
    series is the archive's short-lead forecast, so for h > 1 the real penalty is larger."""
    d, proc = cfg.data, cfg.path("processed")
    train = feats[(feats.d <= d.train_end) & feats.launched & feats.sales.notna()]
    origin = feats.d - feats.h
    rows = feats[(origin >= d.train_end) & (feats.d > d.train_end) & (feats.d <= d.val_end)]
    rows = rows[rows.sales.notna()].reset_index(drop=True)

    span = {
        "fetch_start": f"{rows.date.min():%Y-%m-%d}",
        "fetch_end": f"{rows.date.max():%Y-%m-%d}",
    }
    wcfg = cfg.model_copy(update={"weather": cfg.weather.model_copy(update=span)})
    issued = fetch(wcfg, url=HIST_FORECAST_URL).set_index("date")
    obs = pd.read_parquet(proc / "weather.parquet").set_index("date")[list(issued.columns)]
    err = issued - obs.reindex(issued.index)
    variants = {"observed": rows, "issued": apply_weather_error(rows, err, cfg)}

    hist = HistoricalAgent(cfg).fit(train)
    base = hist.predict(rows)
    agent = weather_agent(cfg).fit(hist.oof_baseline(train))
    cols = control_features(cfg, feats)
    control = {h: fit_lgbm(g, cols, cfg) for h, g in train.groupby("h")}

    out = []
    for name, x in variants.items():
        wa = agent.apply(base, x).forecast.to_numpy()
        ctl = np.empty(len(x))
        for h, model in control.items():
            m = (x.h == h).to_numpy()
            ctl[m] = np.clip(model.predict(x.loc[m, cols]), 0, None)
        for scope, m in (("h1", (x.h == 1).to_numpy()), ("h1-7", np.ones(len(x), bool))):
            y = x.sales.to_numpy()[m]
            out.append({
                "weather_input": name,
                "scope": scope,
                "historical_plus_weather_WAPE": wape(y, wa[m]),
                "lgbm_all_features_WAPE": wape(y, ctl[m]),
                "historical_WAPE": wape(y, base.forecast.to_numpy()[m]),
            })
    e = err.dropna()
    print(
        f"issued - observed, validation dates: MAE tmax {e.tmax.abs().mean():.2f} C, "
        f"tmin {e.tmin.abs().mean():.2f} C, precip {e.precip.abs().mean():.2f} mm; "
        f"{len(err) - len(e)} of {len(err)} dates missing an issued forecast"
    )
    return pd.DataFrame(out)


# ---------------------------------------------------------------- learners
def _hold_split(df: pd.DataFrame, cfg: ExperimentCfg):
    cut = df.d.max() - cfg.lgbm.early_stop_days
    return df[df.d <= cut], df[df.d > cut]


def fit_lgbm(df, feats, cfg):
    return fit_early_stopped(df, feats, "sales", lgb_params(cfg, "historical", "tweedie"), cfg)


def fit_xgb(df, feats, cfg):
    import xgboost as xgb  # noqa: PLC0415  (`posthoc` extra)

    p = cfg.lgbm.historical
    kw = dict(
        objective="reg:tweedie",
        tweedie_variance_power=p["tweedie_variance_power"],
        learning_rate=p["learning_rate"],
        max_leaves=p["num_leaves"],
        grow_policy="lossguide",
        max_depth=0,
        subsample=p["bagging_fraction"],
        colsample_bytree=p["feature_fraction"],
        reg_lambda=p["lambda_l2"],
        tree_method="hist",
        enable_categorical=True,
        random_state=cfg.seed,
        n_jobs=cfg.lgbm.num_threads,
    )
    fit, hold = _hold_split(df, cfg)
    probe = xgb.XGBRegressor(
        n_estimators=cfg.lgbm.max_rounds, early_stopping_rounds=cfg.lgbm.early_stopping_rounds, **kw
    ).fit(fit[feats], fit.sales, eval_set=[(hold[feats], hold.sales)], verbose=False)
    return xgb.XGBRegressor(n_estimators=probe.best_iteration + 1, **kw).fit(df[feats], df.sales)


def fit_hgb(df, feats, cfg, step: int = 100):
    """sklearn has no time-ordered early stopping, so grow in steps on the same hold-out."""
    p = cfg.lgbm.historical
    kw = dict(
        loss="poisson",
        learning_rate=p["learning_rate"],
        max_leaf_nodes=p["num_leaves"],
        min_samples_leaf=p["min_data_in_leaf"],
        l2_regularization=p["lambda_l2"],
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=cfg.seed,
    )
    fit, hold = _hold_split(df, cfg)
    m = HistGradientBoostingRegressor(max_iter=step, warm_start=True, **kw)
    best, best_it = np.inf, step
    for it in range(step, cfg.lgbm.max_rounds + 1, step):
        m.set_params(max_iter=it).fit(fit[feats], fit.sales)
        dev = mean_poisson_deviance(hold.sales, np.clip(m.predict(hold[feats]), 1e-9, None))
        if dev < best:
            best, best_it = dev, it
        elif it - best_it >= cfg.lgbm.early_stopping_rounds:
            break
    return HistGradientBoostingRegressor(max_iter=best_it, **kw).fit(df[feats], df.sales)


LEARNERS = {"lightgbm": fit_lgbm, "xgboost": fit_xgb, "sklearn_hgb": fit_hgb}


def control_features(cfg: ExperimentCfg, feats: pd.DataFrame) -> list[str]:
    """The lgbm_all_features control's inputs, built as run_validation builds them."""
    agents = [calendar_agent(cfg), weather_agent(cfg), trend_agent(cfg)]
    extra = [f for ag in agents for f in ag.features[1:]]
    trend = trend_agent(cfg).features[1:]
    if any(f not in feats for f in trend):  # Rohlik: no interest data, as in run_validation
        extra = [f for f in extra if f not in trend]
    return FEATURES + extra


def algorithms(cfg: ExperimentCfg, feats: pd.DataFrame) -> pd.DataFrame:
    d = cfg.data
    cols = control_features(cfg, feats)
    train = feats[(feats.d <= d.train_end) & feats.launched & feats.sales.notna()]
    origin = feats.d - feats.h
    val = feats[(origin >= d.train_end) & (feats.d > d.train_end) & (feats.d <= d.val_end)]
    val = val[val.sales.notna()]
    preds = []
    for name, fit in LEARNERS.items():
        for h, g in train.groupby("h"):
            print(f"  {name} h={h} ...", flush=True)
            m = fit(g, cols, cfg)
            v = val[val.h == h]
            preds.append(v[["item_id", "d", "h", "sales"]].assign(
                learner=name, yhat=np.clip(m.predict(v[cols]), 0, None)
            ))
    p = pd.concat(preds).astype({"item_id": str})
    stored = pd.read_parquet(cfg.path("processed") / "phase2_all_forecasts.parquet")
    stored = stored[stored.system == "lgbm_all_features"][["item_id", "d", "h", "yhat"]]
    chk = p[p.learner == "lightgbm"].merge(stored, on=["item_id", "d", "h"], suffixes=("", "_s"))
    gap = (chk.yhat - chk.yhat_s).abs().max()
    print(f"lightgbm refit vs stored control: max |diff| = {gap:.2e}")
    return p


def learner_table(p: pd.DataFrame, cfg: ExperimentCfg, ref: str = "lightgbm") -> pd.DataFrame:
    rows = []
    base = p[p.learner == ref].sort_values(["item_id", "d", "h"]).reset_index(drop=True)
    for name, e in p.groupby("learner"):
        e = e.sort_values(["item_id", "d", "h"]).reset_index(drop=True)
        h1, b1 = e[e.h == 1], base[base.h == 1]
        ea, eb = (h1.sales - h1.yhat).abs(), (b1.sales - b1.yhat).abs()
        ci = paired_bootstrap(h1.item_id, ea, eb, h1.sales, cfg.analysis.bootstrap, cfg.seed)
        rows.append({
            "learner": name,
            "h1_WAPE": wape(h1.sales, h1.yhat),
            "h1-7_WAPE": wape(e.sales, e.yhat),
            f"h1_diff_vs_{ref}": ci["diff"],
            "ci_lo": ci["ci_lo"],
            "ci_hi": ci["ci_hi"],
            "DM_p": np.nan if name == ref else diebold_mariano(ea, eb)["p"],
        })
    return pd.DataFrame(rows).set_index("learner")


def main(steps: list[str]) -> None:
    cfg = load_experiment()
    out = ROOT / "reports" / cfg.dataset / "posthoc"
    out.mkdir(parents=True, exist_ok=True)
    deliberation_args = []
    if "deliberation" in steps:  # its three arguments are not step names
        i = steps.index("deliberation")
        deliberation_args, steps = steps[i + 1 : i + 4], [*steps[: i + 1], *steps[i + 4 :]]
    if "quadrants" in steps:
        q = quadrants(cfg)
        q.to_csv(out / "d_error_quadrants.csv")
        print(q.to_string())
    if "censoring" in steps:  # scores stored test outputs, like `quadrants`
        f = pd.read_parquet(
            cfg.path("processed") / "features.parquet", columns=["item_id", "d", "h", "sales"]
        )
        sales = f[f.h == 1].drop(columns="h").astype({"item_id": str})
        c = censoring(cfg, sales)
        c.to_csv(out / "censoring.csv", index=False)
        print(c.to_string(index=False))
    if "weather" in steps:
        f = pd.read_parquet(cfg.path("processed") / "features.parquet")
        w = issued_weather(cfg, f[f.d <= cfg.data.val_end])  # never load test days
        w.to_csv(out / "val_issued_weather.csv", index=False)
        print(w.to_string(index=False))
    if "seeds" in steps:
        f = pd.read_parquet(
            cfg.path("processed") / "features.parquet", columns=["item_id", "d", "h", "sales"]
        )
        f = f[(f.h == 1) & (f.d <= cfg.data.val_end)]
        s = seeds(cfg, f.drop(columns="h").astype({"item_id": str}))
        s.to_csv(out / "val_seeds.csv")
        print(s.to_string())
    if "itemaware" in steps:
        f = pd.read_parquet(
            cfg.path("processed") / "features.parquet", columns=["item_id", "d", "h", "sales"]
        )
        f = f[(f.h == 1) & (f.d <= cfg.data.val_end)]
        a = item_aware(cfg, f.drop(columns="h").astype({"item_id": str}))
        a.to_csv(out / "val_item_aware.csv")
        print(a.to_string())
    if "independent" in steps:
        f = pd.read_parquet(
            cfg.path("processed") / "features.parquet", columns=["item_id", "d", "h", "sales"]
        )
        f = f[(f.h == 1) & (f.d <= cfg.data.val_end)]
        a = independent(cfg, f.drop(columns="h").astype({"item_id": str}))
        a.to_csv(out / "val_independent.csv")
        print(a.to_string())
    if "deliberation" in steps:
        tag, model, n = deliberation_args[0], deliberation_args[1], int(deliberation_args[2])
        arms, curve, h2, per_case, finals = deliberation(cfg, tag, model, n)
        name = f"val_deliberation_{tag}_{model.replace(':', '-').replace('.', '_')}"
        arms.to_csv(out / f"{name}_arms.csv")
        curve.to_csv(out / f"{name}_curve.csv")
        h2.to_csv(out / f"{name}_mechanism.csv")
        per_case.to_parquet(out / f"{name}_cases.parquet", index=False)
        finals.to_parquet(out / f"{name}_forecasts.parquet", index=False)
        print(arms.round(4).to_string())
        print(h2.to_string())
    if "chronos" in steps:
        f = pd.read_parquet(
            cfg.path("processed") / "features.parquet", columns=["item_id", "d", "h", "sales"]
        )
        f = f[(f.h == 1) & (f.d <= cfg.data.val_end)]  # never load test days
        c = chronos_table(cfg, f.drop(columns="h").astype({"item_id": str}))
        c.to_csv(out / "val_chronos.csv")
        print(c.to_string())
    if "sameCases" in steps:
        f = pd.read_parquet(
            cfg.path("processed") / "features.parquet", columns=["item_id", "d", "h", "sales"]
        )
        f = f[(f.h == 1) & (f.d <= cfg.data.val_end)]
        s = same_cases(cfg, out, f.drop(columns="h").astype({"item_id": str}))
        s.to_csv(out / "val_deliberation_same_cases.csv")
        print(s.to_string())
    if "attribution" in steps:
        a = attribution(cfg, out)
        a.to_csv(out / "critic_attribution.csv")
        print(a.to_string())
    if "inference" in steps:
        eq, tests = inference(cfg)
        eq.to_csv(out / "test_equivalence.csv")
        tests.to_csv(out / "test_inference.csv")
        pd.Series(eq.attrs["h2"]).to_csv(out / "test_h2_clustered.csv")
        ref, sd = eq.attrs["ref_wape"], eq.attrs["seed_sd"]
        print(f"no-deliberation h1 WAPE {ref:.4f}, seed sd {sd:.5f}")
        print(eq.T.to_string())
        print(pd.Series(eq.attrs["h2"]).to_string())
        print(tests.to_string())
    if "router" in steps:
        r = router(cfg)
        r.to_csv(out / "val_router.csv")
        print(r.to_string())
    if "nocritic" in steps:  # V-10; argument: number of cases
        n = int(steps[steps.index("nocritic") + 1])
        t = no_critic(cfg, n)
        t.to_csv(out / f"val_nocritic_{n}.csv")
        pd.Series(t.attrs).to_csv(out / f"val_nocritic_{n}_meta.csv")
        print(t.round(4).to_string())
        print(pd.Series(t.attrs).to_string())
    if "histerr" in steps:  # stored agent sets only; nothing refit
        tags = [t for t in ["phase2", "itemaware", "independent"]
                if (cfg.path("processed") / f"{t}_agent_forecasts.parquet").exists()]
        h = pd.DataFrame({t: hist_error_row(cfg, t) for t in tags})
        h.to_csv(out / "val_hist_error.csv")
        print(h.round(4).to_string())
    if "signals" in steps:
        g = signals(cfg)
        g.to_csv(out / "val_trigger_signals.csv")
        print(f"trigger rate {g.attrs['rate']}")
        print(g.round(4).to_string())
    if not {"tails", "learners"} & set(steps):
        return
    feats = pd.read_parquet(cfg.path("processed") / "features.parquet")
    feats = feats[feats.d <= cfg.data.val_end]  # never load test days
    if "tails" in steps:
        sales = feats[feats.h == 1][["item_id", "d", "sales"]].astype({"item_id": str})
        t = tails(cfg, sales)
        t.to_csv(out / "val_band_tails.csv", index=False)
        print(t.to_string(index=False))
    if "learners" in steps:
        a = learner_table(algorithms(cfg, feats), cfg)
        a.to_csv(out / "val_learners.csv")
        print(a.to_string())


if __name__ == "__main__":
    steps = ["tails", "learners", "quadrants", "censoring", "seeds", "weather", "itemaware"]
    main(sys.argv[1:] or [*steps, "router", "inference"])
