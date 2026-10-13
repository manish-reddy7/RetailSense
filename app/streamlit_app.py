"""RetailSense dashboard (blueprint 10.3): Forecast, Explain, Evaluate, Live.

Reads stored results only - SQLite run store, data/processed outputs and reports/. It never
fits or re-runs a model. Run: streamlit run app/streamlit_app.py
"""

import json
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import requests_cache
import streamlit as st

from retailsense.config import ROOT, load_experiment, load_inventory
from retailsense.data.load import load_calendar

cfg, inv = load_experiment(), load_inventory()
DATA = f"{cfg.data.store} kirana store, India"
REPORTS = ROOT / "reports" / cfg.dataset
FIGS = ROOT / "reports" / "figures" / cfg.dataset
# experiment id in SQLite -> (database, agent-forecast parquet tag, inventory run-id prefix, label)
SOURCES = {
    "final_test": (cfg.path("db"), "final_test", "final_test-retailsense", "Final TEST run"),
    "dryrun_val": (
        cfg.path("db").with_name(f"{cfg.path('db').stem}_dryrun.db"),
        "dryrun_val",
        "dryrun_val-retailsense",
        "Dry run on VALIDATION (stub LLM - every agent holds; not a result)",
    ),
    "phase3b": (cfg.path("db"), "phase2", None, "Phase 3b validation sample"),
    "phase3b_pilot": (cfg.path("db"), "phase2", None, "Phase 3b validation pilot (50 cases)"),
}

st.set_page_config(page_title="RetailSense", layout="wide")


@st.cache_data
def query(db: str, sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(db) as con:
        return pd.read_sql_query(sql, con, params=params)


@st.cache_data
def calendar() -> pd.DataFrame:
    c = load_calendar(cfg).assign(d=lambda x: x.d.str.removeprefix("d_").astype(int))
    return c.assign(date=pd.to_datetime(c.date).dt.date.astype(str))[["d", "date"]]


@st.cache_data
def agent_bands(tag: str, item: str, origin: int) -> pd.DataFrame:
    p = cfg.path("processed") / f"{tag}_agent_forecasts.parquet"
    if not p.exists():
        return pd.DataFrame()
    af = pd.read_parquet(p, filters=[("origin", "==", origin), ("h", "==", 1)])
    return af[af.item_id.astype(str) == item]


def available() -> dict:
    out = {}
    for exp, (db, *_rest, label) in SOURCES.items():
        if db.exists() and len(
            query(str(db), "SELECT 1 FROM forecast_runs WHERE experiment_id=? LIMIT 1", (exp,))
        ):
            out[exp] = label
    return out


st.sidebar.title("RetailSense")
st.sidebar.caption(DATA)
view = st.sidebar.radio("View", ["Forecast", "Explain", "Evaluate", "Live"])
srcs = available()
if not srcs and view != "Live":
    st.warning("No stored runs yet. Run the experiments first (see README).")
    st.stop()
exp = st.sidebar.selectbox("Run", list(srcs), format_func=srcs.get) if srcs else None
if exp:
    db, tag, inv_prefix, label = SOURCES[exp]
    db = str(db)
    tau = query(db, "SELECT MAX(tau) t FROM forecast_runs WHERE experiment_id=?", (exp,)).t[0]
    items = query(
        db, "SELECT DISTINCT item_id FROM forecast_runs WHERE experiment_id=? ORDER BY 1", (exp,)
    ).item_id.tolist()
    item = st.sidebar.selectbox("Item", items)
    dates = query(
        db,
        "SELECT prediction_time FROM forecast_runs WHERE experiment_id=? AND item_id=? ORDER BY 1",
        (exp, item),
    ).prediction_time.tolist()
    day = st.sidebar.selectbox("Target date", dates, index=len(dates) - 1)
    run = query(
        db,
        "SELECT * FROM forecast_runs WHERE experiment_id=? AND item_id=? AND prediction_time=?",
        (exp, item, day),
    ).iloc[0]
    agents = query(db, "SELECT * FROM agent_predictions WHERE run_id=?", (run.run_id,))
    critic = query(db, "SELECT * FROM critic_reports WHERE run_id=?", (run.run_id,))
    target_d = int(calendar().set_index("date").d[day])
    consensus = float(
        (agents.initial_forecast * agents.initial_confidence).sum()
        / agents.initial_confidence.sum()
    )
    triggered = run.D >= tau
    final = run.final_forecast if triggered else consensus
    state = (
        query(
            db,
            "SELECT * FROM inventory_state WHERE run_id=?",
            (f"{inv_prefix}-{item}-{target_d - 1}",),
        )
        if inv_prefix
        else pd.DataFrame()
    )


def badge(ok: bool, yes: str, no: str) -> None:
    (st.error if ok else st.success)(yes if ok else no)


if view == "Forecast":
    st.header(f"Forecast - {item}, {day}")
    st.caption(f"{label}. The forecast was made at the end of the previous day.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final forecast (units)", f"{final:.2f}")
    c2.metric("Disagreement D", f"{run.D:.3f}", f"tau {tau:.3f}", delta_color="off")
    c3.metric("Actual (revealed later)", f"{run.actual:.0f}")
    if len(state):
        s = state.iloc[0]
        c4.metric("Stock on hand (end of day)", f"{s.on_hand:.0f}")
    badge(
        triggered,
        "DELIBERATED: D >= tau, the critic and one revision round ran",
        "FAST PATH: D < tau, confidence-weighted consensus",
    )
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=run.D,
            title={"text": "D vs tau"},
            gauge={
                "axis": {"range": [0, max(1.0, run.D * 1.2)]},
                "threshold": {"value": tau, "line": {"color": "red", "width": 3}},
            },
        )
    )
    gauge.update_layout(height=250, margin=dict(t=40, b=0))
    bands = agent_bands(tag, item, target_d - 1)
    fig = go.Figure()
    for r in bands.itertuples():
        fig.add_trace(
            go.Scatter(
                x=[r.agent],
                y=[r.forecast],
                mode="markers",
                marker_size=12,
                name=r.agent,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[r.upper - r.forecast],
                    arrayminus=[r.forecast - r.lower],
                ),
            )
        )
    fig.add_hline(y=final, line_dash="dash", annotation_text="final")
    fig.add_hline(y=run.actual, line_color="black", annotation_text="actual")
    fig.update_layout(title="Agent forecasts with 10-90% bands", height=380, showlegend=False)
    a, b = st.columns([2, 1])
    a.plotly_chart(fig, width="stretch")
    b.plotly_chart(gauge, width="stretch")
    if len(state):
        s = state.iloc[0]
        st.subheader(
            f"Recommended action: {s.action}" + (f" {s.order_qty:.0f} units" if s.order_qty else "")
        )
        st.caption("RetailSense system in the shared (s, S) simulator; quantity is code-computed.")
    else:
        st.info("No inventory simulation stored for this run (validation deliberation sample).")

elif view == "Explain":
    st.header(f"Explain - {item}, {day}")
    st.caption(label)
    st.subheader("Key factors per agent")
    for r in agents.itertuples():
        with st.expander(
            f"{r.agent_id}: {r.initial_forecast:.2f} (confidence {r.initial_confidence:.2f})"
        ):
            st.write(json.loads(r.key_factors))
            lim = json.loads(r.limitations)
            if lim:
                st.caption("Limitations: " + "; ".join(lim))
    st.subheader("Critic conflict report")
    if len(critic):
        rep = json.loads(critic.report_json[0])
        st.write(f"**{rep['primary_conflict']}**")
        st.table(pd.DataFrame(rep["evidence_assessment"]))
        st.caption(f"Model {critic.model_version[0]}, prompt {critic.prompt_version[0]}")
        if not triggered:
            st.info(
                "D < tau, so RetailSense did NOT use this report. It comes from the "
                "always-on arm, stored for comparison."
            )
    else:
        st.write("No critic report (fast path or fallback).", run.fallback or "")
    st.subheader("Initial vs revised forecasts")
    st.dataframe(
        agents[
            [
                "agent_id",
                "initial_forecast",
                "revised_forecast",
                "confidence_multiplier",
                "rationale",
            ]
        ],
        hide_index=True,
    )
    st.caption(
        f"Consensus {consensus:.2f} -> deliberated {run.final_forecast:.2f}. The LLM "
        "proposes a multiplier and a value inside the agent's band; code computes the number."
    )
    if len(state):
        s = state.iloc[0]
        ip = s.on_hand + s.on_order
        st.subheader("Why this action")
        st.write(
            f"Inventory position IP = on hand {s.on_hand:.0f} + on order {s.on_order:.0f} = "
            f"**{ip:.0f}**. Reorder point s = {s.reorder_point:.1f} (safety stock "
            f"{s.safety_stock:.1f}), order-up-to S = {s.order_up_to:.1f}. "
            + (
                f"IP <= s, so order S - IP rounded up to case packs of {inv.case_pack}: "
                f"**{s.order_qty:.0f}**."
                if s.action == "RESTOCK"
                else f"IP > s, so no order ({s.action})."
            )
        )

elif view == "Evaluate":
    st.header("Evaluate")
    kind = st.radio("Results", ["final", "dryrun", "validation (phases 3b-4)"], horizontal=True)
    if kind == "dryrun":
        st.warning(
            "Dry run: validation window, stub LLM where every agent holds. Pipeline "
            "check only - these are not results."
        )
    if kind == "validation (phases 3b-4)":
        files = {
            "Inventory (validation, systems 1-6)": REPORTS / "phase4_inventory.csv",
            "Deliberation arms (validation sample)": REPORTS / "phase3b_arms.csv",
            "H1 summary (validation)": REPORTS / "h1_summary.csv",
        }
        figs = [
            FIGS / "h1_deciles_and_sweep.png",
            FIGS / "cost_accuracy.png",
            FIGS / "inventory_sensitivity.png",
        ]
    else:
        d = REPORTS / kind
        files = {
            "Forecast metrics (all systems)": d / "table_forecast_metrics.csv",
            "Inventory outcomes": d / "table_inventory.csv",
            "Significance vs RetailSense": d / "table_significance.csv",
            "Latency and LLM calls": d / "table_latency_llm.csv",
            "Ablations": d / "table_ablations.csv",
            "Rolling-origin backtest": d / "table_backtest.csv",
        }
        figs = [
            FIGS / kind / f
            for f in ("h1_deciles.png", "cost_accuracy.png", "before_after.png", "case_study.png")
        ]
    for title, p in files.items():
        if p.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(p).round(4), hide_index=True)
    for f in figs:
        if f.exists():
            st.image(str(f))
    if not any(p.exists() for p in files.values()):
        st.info("No stored results for this selection yet.")

else:  # Live
    st.header("Live - DEMO MODE")
    st.warning(
        "Demo mode. Live signals are shown next to a REPLAYED item's stored forecasts from "
        f"{DATA}. The dashboard does not re-run models, has no store POS data, and nothing "
        "here is a real-time forecast for a real store."
    )
    session = requests_cache.CachedSession(
        cfg.path("cache") / "live", expire_after=3600, stale_if_error=True
    )
    lat, lon = cfg.weather.city[cfg.data.state]
    st.subheader(f"Today's Open-Meteo forecast ({cfg.data.state} representative city)")
    try:
        r = session.get(
            "https://api.open-meteo.com/v1/forecast",
            timeout=15,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": ",".join(cfg.weather.vars),
                "timezone": cfg.weather.timezone,
                "forecast_days": 7,
            },
        )
        r.raise_for_status()
        dly = pd.DataFrame(r.json()["daily"])
        dly["rainy_day"] = dly["precipitation_sum"] >= cfg.weather.rainy_mm
        st.dataframe(dly, hide_index=True)
        st.caption("From cache (API unavailable or recent)" if r.from_cache else "Fresh from API")
    except Exception as e:  # noqa: BLE001 - demo view: show the failure, never crash
        st.error(f"Weather unavailable and nothing cached: {e}")
    if exp:
        st.subheader(f"Replayed item {item}: stored agent forecasts for {day}")
        st.dataframe(
            agents[["agent_id", "initial_forecast", "initial_confidence"]], hide_index=True
        )
