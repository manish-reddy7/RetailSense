"""Deliberation with a MOCKED LLM - these tests never call the API."""

import json
from datetime import date

import pytest

from retailsense.agents.schema import AgentForecast
from retailsense.orchestrator.critic import CriticReport
from retailsense.orchestrator.graph import build_graph, run_case
from retailsense.orchestrator.llm import LLM
from retailsense.orchestrator.revise import Revision, apply_revision
from retailsense.store.db import connect, log_case

REPORT = {
    "primary_conflict": "Calendar expects a weekend lift that Historical does not.",
    "conflicting_agents": ["historical", "calendar"],
    "evidence_assessment": [
        {"agent": "historical", "strength": "strong", "reason": "stable 8-week pattern"},
        {"agent": "calendar", "strength": "moderate", "reason": "Saturday"},
        {"agent": "weather", "strength": "weak", "reason": "normal day"},
    ],
    "missing_or_stale_evidence": [],
    "questions_for_agents": [{"agent": "calendar", "question": "Is the lift item-specific?"}],
}


def agent(aid, f, lo, hi, c):
    return AgentForecast(
        agent_id=aid,
        item_id="milk_1",
        store_id="MUM_1",
        prediction_time=date(2016, 1, 2),
        horizon_days=[1],
        forecast=[f],
        lower=[lo],
        upper=[hi],
        confidence=c,
        key_factors=[f"{aid} kf"],
        limitations=[],
    )


def case():
    return {
        "dataset": "india",
        "item_id": "milk_1",
        "store_id": "MUM_1",
        "origin": 1800,
        "history_end_d": 1800,
        "target_date": date(2016, 1, 3),
        "history": [10.0] * 14,
        "agents": [
            agent("historical", 10, 6, 14, 0.6),
            agent("calendar", 16, 11, 21, 0.5),
            agent("weather", 10.5, 6.5, 14.5, 0.6),
        ],
    }


class FakeLLM:
    """Scripted transport: a list of responses per prompt, consumed in order."""

    def __init__(self, critic: list[str], revise: list[str]):
        self.queue = {"critic": list(critic), "revise": list(revise)}
        self.calls = {"critic": 0, "revise": 0}

    def __call__(self, system, user, schema):
        name = "critic" if schema is CriticReport else "revise"
        self.calls[name] += 1
        q = self.queue[name]
        text = q.pop(0) if len(q) > 1 else q[0]
        return text, 100, 20, "fake-model", 5.0


def rev(m=1.0, y=10.0, cite=("historical",)):
    return json.dumps(
        {
            "confidence_multiplier": m,
            "revised_forecast": y,
            "rationale": "because",
            "cited_evidence": list(cite),
        }
    )


@pytest.fixture
def cfg(cfg, tmp_path):
    paths = cfg.paths.model_copy(update={"cache": tmp_path})
    return cfg.model_copy(update={"paths": paths})


def graph(cfg, fake):
    return build_graph(cfg, LLM(cfg, transport=fake))


def test_critic_schema_has_no_numeric_field():
    def numeric(node):
        if isinstance(node, dict):
            if node.get("type") in ("number", "integer"):
                return True
            return any(numeric(v) for v in node.values())
        return isinstance(node, list) and any(numeric(v) for v in node)

    assert not numeric(CriticReport.model_json_schema())


def test_fast_path_makes_no_llm_calls(cfg):
    fake = FakeLLM([json.dumps(REPORT)], [rev()])
    s = run_case(graph(cfg, fake), case(), tau=99.0)
    assert s["route"] == "fast" and s["final"] == s["consensus"] and fake.calls["critic"] == 0
    assert set(s["timings"]) >= {"load_context", "run_agents", "disagreement", "consensus"}


def test_deliberation_final_is_code_computed(cfg):
    fake = FakeLLM([json.dumps(REPORT)], [rev(1.0, 10), rev(0.5, 14), rev(1.0, 10.5)])
    s = run_case(graph(cfg, fake), case(), tau=0.0)
    # weights c*m: 0.6, 0.25, 0.6 -> final = (0.6*10 + 0.25*14 + 0.6*10.5) / 1.45
    assert s["deliberated"] and s["final"] == pytest.approx((6 + 3.5 + 6.3) / 1.45)
    assert fake.calls == {"critic": 1, "revise": 3} and s["llm_calls"] == 4


def test_adversarial_llm_is_clipped_to_bounds_and_bands():
    agents = case()["agents"]
    revs = {
        a.agent_id: Revision(
            confidence_multiplier=50,
            revised_forecast=1e9,
            rationale="x",
            cited_evidence=["historical"],
        )
        for a in agents
    }
    final, rows = apply_revision(agents, revs, (0.25, 2.0))
    assert all(
        r["m"] == 2.0 and r["y"] == a.upper[0] and r["clipped"]
        for r, a in zip(rows, agents, strict=True)
    )
    assert max(a.lower[0] for a in agents) <= final <= max(a.upper[0] for a in agents)


def test_invalid_json_retries_once_then_falls_back(cfg):
    fake = FakeLLM(["not json", "still not json"], [rev()])
    s = run_case(graph(cfg, fake), case(), tau=0.0)
    assert fake.calls["critic"] == 2 and fake.calls["revise"] == 0
    assert s["final"] == s["consensus"] and not s["deliberated"]
    assert s["fallback"].startswith("critic invalid") and len(s["llm_errors"]) == 1


def test_retry_can_recover(cfg):
    fake = FakeLLM(["{broken", json.dumps(REPORT)], [rev()])
    s = run_case(graph(cfg, fake), case(), tau=0.0)
    assert s["deliberated"] and fake.calls["critic"] == 2


def test_uncited_evidence_is_invalid(cfg):
    fake = FakeLLM([json.dumps(REPORT)], [rev(cite=("invented_evidence",))])
    s = run_case(graph(cfg, fake), case(), tau=0.0)
    assert s["fallback"].startswith("revision invalid") and s["final"] == s["consensus"]


def test_critic_must_assess_every_agent(cfg):
    partial = {**REPORT, "evidence_assessment": REPORT["evidence_assessment"][:2]}
    s = run_case(graph(cfg, FakeLLM([json.dumps(partial)], [rev()])), case(), tau=0.0)
    assert s["fallback"].startswith("critic invalid")


def test_cache_prevents_repeat_calls(cfg):
    fake = FakeLLM([json.dumps(REPORT)], [rev()])
    g = graph(cfg, fake)
    run_case(g, case(), tau=0.0)
    s = run_case(g, case(), tau=0.0)
    assert fake.calls == {"critic": 1, "revise": 3} and s["llm_calls"] == 0


def test_invalid_agent_is_excluded_not_zeroed(cfg):
    c = case()
    bad = c["agents"][2].model_copy(update={"forecast": [99.0]})  # outside its own band
    c["agents"][2] = bad
    s = run_case(graph(cfg, FakeLLM([json.dumps(REPORT)], [rev()])), c, tau=99.0)
    assert s["excluded"] == ["weather"] and len(s["agents"]) == 2 and s["final"] > 0


def test_history_must_end_at_origin(cfg):
    c = {**case(), "history_end_d": 1801}  # one day of the future
    with pytest.raises(ValueError):
        run_case(graph(cfg, FakeLLM([json.dumps(REPORT)], [rev()])), c, tau=0.0)


def test_db_logs_initial_and_revised_separately(cfg, tmp_path):
    fake = FakeLLM([json.dumps(REPORT)], [rev(1.0, 10), rev(0.5, 14), rev(1.0, 10.5)])
    s = run_case(graph(cfg, fake), case(), tau=0.0)
    con = connect(tmp_path / "t.db")
    log_case(con, "r1", "exp", "selective", s, 12.0, "v1", "fake-model")
    rows = dict(con.execute("SELECT agent_id, initial_forecast FROM agent_predictions").fetchall())
    revised = dict(con.execute("SELECT agent_id, revised_forecast FROM agent_predictions"))
    assert rows["calendar"] == 16 and revised["calendar"] == 14
    assert con.execute("SELECT count(*) FROM critic_reports").fetchone()[0] == 1


def test_echoing_rounded_value_is_a_hold():
    a = agent("historical", 2.0834, 1.0, 4.0, 0.6)
    r = Revision(
        confidence_multiplier=1.0,
        revised_forecast=2.08,
        rationale="hold",
        cited_evidence=["historical"],
    )
    final, rows = apply_revision([a], {"historical": r}, (0.25, 2.0))
    assert rows[0]["held"] and rows[0]["y"] == 2.0834 and final == 2.0834


def test_prompt_text_change_invalidates_cache(cfg, monkeypatch):
    fake = FakeLLM([json.dumps(REPORT)], [rev()])
    g = graph(cfg, fake)
    run_case(g, case(), tau=0.0)
    import retailsense.orchestrator.llm as llm_mod

    original = llm_mod.load_prompt
    monkeypatch.setattr(llm_mod, "load_prompt", lambda n, v: original(n, v) + "\nedited")
    run_case(g, case(), tau=0.0)
    assert fake.calls["critic"] == 2


def test_cache_is_safe_under_concurrent_identical_calls(cfg, tmp_path):
    """Worker threads asking the same question must never read a half-written cache file."""
    import threading
    import time

    from retailsense.orchestrator.critic import CriticReport

    def slow(system, user, schema):
        time.sleep(0.01)
        return json.dumps(REPORT), 1, 1, "fake", 1.0

    paths = cfg.paths.model_copy(update={"cache": tmp_path})
    llm = LLM(cfg.model_copy(update={"paths": paths}), transport=slow)
    errors = []

    def call():
        try:
            assert llm.call("critic", {"same": 1}, CriticReport).value is not None
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_critic_retry_message_is_process_independent():
    """The retry prompt embeds this message; if it changed per process, replays would miss the
    LLM cache and give different answers."""
    import os
    import subprocess
    import sys

    code = (
        "from retailsense.orchestrator.critic import CriticReport, check_report\n"
        "r = CriticReport(primary_conflict='x', conflicting_agents=[], evidence_assessment=[],"
        " missing_or_stale_evidence=[], questions_for_agents=[])\n"
        "try:\n"
        "    check_report({'historical', 'calendar', 'weather', 'trend'})(r)\n"
        "except ValueError as e:\n"
        "    print(e)\n"
    )
    out = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3", "4")
    }
    assert len(out) == 1 and "['calendar', 'historical', 'trend', 'weather']" in out.pop()


def test_ollama_retries_server_errors(cfg, monkeypatch):
    """A CUDA OOM makes Ollama return 500 while it reloads; that must not end the run."""
    import requests

    from retailsense.orchestrator import llm as llm_mod

    class Resp:
        def __init__(self, code):
            self.status_code, self.ok, self.text = code, code == 200, "x"

        def json(self):
            return {"message": {"content": "{}"}, "prompt_eval_count": 1, "eval_count": 1}

    codes = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp(codes.pop(0)))
    call = llm_mod.ollama_transport(cfg)
    codes[:] = [500, 200]
    assert call("s", "u", CriticReport)[0] == "{}"
    codes[:] = [400]
    with pytest.raises(RuntimeError, match="Ollama 400"):
        call("s", "u", CriticReport)
