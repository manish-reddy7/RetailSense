"""Export stored results to JSON for the Next.js web app (web/). Reads the SQLite run store,
data/processed agent forecasts and reports/ tables; never fits or runs a model.

Run: python -m retailsense.web_export        (RETAILSENSE_CONFIG picks the dataset)
Writes web/public/data/<dataset>/: meta.json, tables.json, runs/<experiment>/<item>.json
"""

import json
import math
import shutil
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from retailsense.config import ROOT, load_experiment, load_inventory
from retailsense.data.load import load_calendar

# experiment id -> (db suffix, agent-forecast tag, inventory run-id prefix, label)
EXPERIMENTS = {
    "final_test": ("", "final_test", "final_test-retailsense", "Final TEST run"),
    "dryrun_val": (
        "_dryrun",
        "dryrun_val",
        "dryrun_val-retailsense",
        "Dry run on validation (stub LLM: every agent holds, not a result)",
    ),
    "phase3b": ("", "phase2", None, "Phase 3b validation sample"),
    "phase3b_pilot": ("", "phase2", None, "Phase 3b validation pilot (50 cases)"),
}


def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _write(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), separators=(",", ":")), encoding="utf-8")


def tables(cfg) -> dict:
    """Every CSV in reports/<dataset>/ (validation) and its final/ and dryrun/ subfolders."""
    base = ROOT / "reports" / cfg.dataset
    out = {}
    for kind, folder in (
        ("validation", base),
        ("dryrun", base / "dryrun"),
        ("final", base / "final"),
    ):
        files = sorted(folder.glob("*.csv")) if folder.exists() else []
        if files:
            out[kind] = {f.stem: pd.read_csv(f).to_dict("records") for f in files}
    return out


def export_run(cfg, exp, db, tag, inv_prefix, dates, out) -> dict:
    con = sqlite3.connect(db)
    runs = pd.read_sql_query(
        "SELECT * FROM forecast_runs WHERE experiment_id=?", con, params=(exp,)
    )
    agents = pd.read_sql_query(
        "SELECT a.* FROM agent_predictions a JOIN forecast_runs f USING (run_id) "
        "WHERE f.experiment_id=?",
        con,
        params=(exp,),
    )
    critic = (
        pd.read_sql_query(
            "SELECT c.run_id, c.report_json FROM critic_reports c "
            "JOIN forecast_runs f USING (run_id) "
            "WHERE f.experiment_id=?",
            con,
            params=(exp,),
        )
        .set_index("run_id")
        .report_json
    )
    inv = pd.DataFrame()
    if inv_prefix:
        inv = pd.read_sql_query(
            "SELECT * FROM inventory_state WHERE run_id LIKE ?", con, params=(f"{inv_prefix}-%",)
        ).set_index("run_id")
    con.close()
    af_path = cfg.path("processed") / f"{tag}_agent_forecasts.parquet"
    bands = pd.DataFrame()
    if af_path.exists():
        bands = pd.read_parquet(af_path, filters=[("h", "==", 1)]).astype({"item_id": str})
        bands = bands.set_index(["item_id", "origin", "agent"])
    runs["d"] = runs.prediction_time.map(dates)
    by_run = {k: g for k, g in agents.groupby("run_id")}
    points = []
    for item, g in runs.sort_values("d").groupby("item_id"):
        cases = []
        for r in g.itertuples():
            a = by_run[r.run_id]
            cons = float(
                (a.initial_forecast * a.initial_confidence).sum() / a.initial_confidence.sum()
            )
            triggered = bool(r.D >= r.tau)
            ags = []
            for x in a.itertuples():
                b = bands.loc[(item, r.d - 1, x.agent_id)] if len(bands) else None
                ags.append(
                    {
                        "id": x.agent_id,
                        "forecast": x.initial_forecast,
                        "lower": None if b is None else float(b.lower),
                        "upper": None if b is None else float(b.upper),
                        "confidence": x.initial_confidence,
                        "revised": x.revised_forecast,
                        "m": x.confidence_multiplier,
                        "rationale": x.rationale,
                        "key_factors": json.loads(x.key_factors),
                        "limitations": json.loads(x.limitations),
                    }
                )
            st = f"{inv_prefix}-{item}-{r.d - 1}"
            s = inv.loc[st].to_dict() if len(inv) and st in inv.index else None
            cases.append(
                {
                    "date": r.prediction_time,
                    "d": int(r.d),
                    "D": r.D,
                    "tau": r.tau,
                    "triggered": triggered,
                    "consensus": cons,
                    "deliberated": r.final_forecast,
                    "final": r.final_forecast if triggered else cons,
                    "actual": r.actual,
                    "fallback": r.fallback,
                    "agents": ags,
                    "critic": json.loads(critic[r.run_id]) if r.run_id in critic.index else None,
                    "inventory": s,
                }
            )
            points.append((r.D, r.actual, cons))
        _write(out / "runs" / exp / f"{item}.json", cases)
    p = pd.DataFrame(points, columns=["D", "y", "cons"])
    p["rel"] = (p.y - p.cons).abs() / p.y.clip(lower=1)
    p["dec"] = pd.qcut(p.D.rank(method="first"), min(10, len(p)), labels=False) + 1
    deciles = [
        {
            "decile": int(k),
            "mean_rel_err": float(q.rel.mean()),
            "wape": float((q.y - q.cons).abs().sum() / max(q.y.sum(), 1)),
            "n": len(q),
        }
        for k, q in p.groupby("dec")
    ]
    return {
        "items": sorted(runs.item_id.unique().tolist()),
        "n_cases": len(runs),
        "tau": float(runs.tau.max()),
        "deciles": deciles,
    }


def main() -> None:
    cfg = load_experiment()
    out = ROOT / "web" / "public" / "data" / cfg.dataset
    if out.exists():
        shutil.rmtree(out)
    cal = load_calendar(cfg).assign(d=lambda c: c.d.str.removeprefix("d_").astype(int))
    dates = dict(zip(pd.to_datetime(cal.date).dt.strftime("%Y-%m-%d"), cal.d, strict=True))
    exps = {}
    for exp, (suffix, tag, inv_prefix, label) in EXPERIMENTS.items():
        db = cfg.path("db").with_name(f"{cfg.path('db').stem}{suffix}.db")
        if not db.exists():
            continue
        with sqlite3.connect(db) as con:
            n = con.execute(
                "SELECT COUNT(*) FROM forecast_runs WHERE experiment_id=?", (exp,)
            ).fetchone()[0]
        if n:
            print(f"exporting {exp} ({n} cases) ...", flush=True)
            exps[exp] = {"label": label, **export_run(cfg, exp, db, tag, inv_prefix, dates, out)}
    lat, lon = cfg.weather.city[cfg.data.state]
    with sqlite3.connect(cfg.path("db")) as con:
        agents = [r[0] for r in con.execute("SELECT DISTINCT agent_id FROM agent_predictions")]
    real = cfg.source is not None
    meta = {
        "dataset": cfg.dataset,
        "label": f"{cfg.data.store} · Rohlik online grocer, Prague (real sales)"
        if real
        else f"{cfg.data.store} · kirana store, Mumbai (simulated sales)",
        "real_data": real,
        "agents": sorted(agents, key=["historical", "calendar", "weather", "trend"].index),
        "store": cfg.data.store,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "model": cfg.deliberation.model,
        "prompt_version": cfg.deliberation.prompt_version,
        "tau_frozen": cfg.deliberation.tau,
        "experiments": exps,
        "inventory": {"case_pack": load_inventory().case_pack},
        "live": {
            "lat": lat,
            "lon": lon,
            "timezone": cfg.weather.timezone,
            "rainy_mm": cfg.weather.rainy_mm,
        },
    }
    _write(out / "meta.json", meta)
    _write(out / "tables.json", tables(cfg))
    idx = ROOT / "web" / "public" / "data" / "index.json"
    known = json.loads(idx.read_text()) if idx.exists() else []
    _write(idx, sorted(set(known) | {cfg.dataset}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
