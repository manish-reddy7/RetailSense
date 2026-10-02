"""SQLite run store (blueprint 10.1). Initial and revised agent forecasts are stored separately."""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id TEXT PRIMARY KEY, experiment_id TEXT, system TEXT, item_id TEXT, store_id TEXT,
    prediction_time TEXT, horizon INTEGER, actual REAL, final_forecast REAL, lower REAL,
    upper REAL, D REAL, tau REAL, deliberation_triggered INTEGER, latency_ms REAL,
    fallback TEXT);
CREATE TABLE IF NOT EXISTS agent_predictions (
    run_id TEXT, agent_id TEXT, initial_forecast REAL, initial_confidence REAL,
    revised_forecast REAL, confidence_multiplier REAL, key_factors TEXT, limitations TEXT,
    rationale TEXT, PRIMARY KEY (run_id, agent_id));
CREATE TABLE IF NOT EXISTS critic_reports (
    run_id TEXT PRIMARY KEY, report_json TEXT, model_version TEXT, prompt_version TEXT,
    tokens_in INTEGER, tokens_out INTEGER, cache_hit INTEGER);
CREATE TABLE IF NOT EXISTS inventory_state (
    run_id TEXT, on_hand REAL, on_order REAL, safety_stock REAL, reorder_point REAL,
    order_up_to REAL, action TEXT, order_qty REAL, sold REAL, lost REAL);
CREATE TABLE IF NOT EXISTS evaluation_results (
    experiment_id TEXT, system TEXT, MAE REAL, RMSE REAL, WAPE REAL, sMAPE REAL, coverage REAL,
    stockout_rate REAL, fill_rate REAL, overstock REAL, total_cost REAL, trigger_rate REAL,
    llm_calls REAL, mean_latency REAL, PRIMARY KEY (experiment_id, system));
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def log_case(
    con: sqlite3.Connection,
    run_id: str,
    experiment_id: str,
    system: str,
    s: dict,
    actual: float,
    prompt_version: str,
    model_version: str,
) -> None:
    """One graph run: forecast row, one row per agent (initial vs revised), critic report."""
    c = s["case"]
    lat = sum(s["timings"].values())
    con.execute(
        "INSERT OR REPLACE INTO forecast_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            run_id,
            experiment_id,
            system,
            c["item_id"],
            c["store_id"],
            str(c["target_date"]),
            1,
            actual,
            s["final"],
            None,
            None,
            s["D"],
            s.get("tau"),
            int(s["route"] == "deliberate"),
            lat,
            s.get("fallback"),
        ),
    )
    rev = {r["agent_id"]: r for r in s.get("revisions") or []}
    for a in s["agents"]:
        r = rev.get(a.agent_id, {})
        con.execute(
            "INSERT OR REPLACE INTO agent_predictions VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                a.agent_id,
                a.forecast[0],
                a.confidence,
                r.get("y"),
                r.get("m"),
                json.dumps(a.key_factors),
                json.dumps(a.limitations),
                r.get("rationale"),
            ),
        )
    if s.get("critic") is not None:
        con.execute(
            "INSERT OR REPLACE INTO critic_reports VALUES (?,?,?,?,?,?,?)",
            (
                run_id,
                s["critic"].model_dump_json(),
                model_version,
                prompt_version,
                s["tokens_in"],
                s["tokens_out"],
                int(s["llm_calls"] == 0),
            ),
        )


def log_evaluation(con: sqlite3.Connection, experiment_id: str, rows: list[dict]) -> None:
    cols = [
        "system",
        "MAE",
        "RMSE",
        "WAPE",
        "sMAPE",
        "coverage",
        "stockout_rate",
        "fill_rate",
        "overstock",
        "total_cost",
        "trigger_rate",
        "llm_calls",
        "mean_latency",
    ]
    for r in rows:
        con.execute(
            f"INSERT OR REPLACE INTO evaluation_results VALUES (?,{','.join('?' * len(cols))})",
            (experiment_id, *[r.get(k) for k in cols]),
        )
