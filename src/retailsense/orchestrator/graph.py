"""LangGraph workflow (blueprint 7.3):

load_context -> run_agents -> disagreement -> route -> consensus ----------> inventory_policy
                                                   \\-> critic -> revise -/

Every node's wall time is recorded. Any invalid LLM output (after its one retry) makes the case
fall back to fast consensus, with the reason logged. inventory_policy is a Phase 4 placeholder.
"""

import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from retailsense.agents.schema import AgentForecast
from retailsense.config import ExperimentCfg
from retailsense.orchestrator.critic import run_critic
from retailsense.orchestrator.disagreement import consensus, d_range, disagreement, route
from retailsense.orchestrator.llm import LLM, LLMResult
from retailsense.orchestrator.revise import apply_revision, run_revision


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class State(TypedDict, total=False):
    case: dict  # dataset, item_id, store_id, origin, target_date, history, agents
    tau: float
    trigger: bool | None  # set by an experiment arm (random / always / never); None = use D vs tau
    agents: list[AgentForecast]
    excluded: list[str]
    D: float
    D_range: float
    consensus: float
    route: str
    critic: Any  # CriticReport | None
    revisions: list[dict]
    final: float
    deliberated: bool
    fallback: str | None
    inventory: Any
    timings: Annotated[dict, _merge]
    llm_calls: Annotated[int, operator.add]  # API calls actually made (0 when cached)
    llm_attempts: Annotated[int, operator.add]  # calls the case costs, cached or not (with retries)
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    llm_errors: Annotated[list, operator.add]


def _llm_stats(r: LLMResult) -> dict:
    return {
        "llm_calls": r.calls,
        "llm_attempts": r.attempts,
        "tokens_in": r.tokens_in,
        "tokens_out": r.tokens_out,
        "llm_errors": [r.error] if r.error else [],
    }


def build_graph(cfg: ExperimentCfg, llm: LLM):
    d = cfg.deliberation

    def timed(name, fn):
        def run(s: State) -> dict:
            t = time.perf_counter()
            out = fn(s) or {}
            return {**out, "timings": {name: 1000 * (time.perf_counter() - t)}}

        return run

    def load_context(s: State) -> dict:
        c = s["case"]
        if len(c["history"]) != d.history_days or c["history_end_d"] != c["origin"]:
            raise ValueError("history must be the last history_days revealed days up to origin")
        return {}

    def run_agents(s: State) -> dict:
        """Agents ran upstream (replay); re-validate at the orchestrator boundary. An invalid agent
        is excluded and flagged - never turned into a zero forecast."""
        ok, excluded = [], []
        for a in s["case"]["agents"]:
            try:
                ok.append(AgentForecast.model_validate(a.model_dump()))
            except ValidationError:
                excluded.append(getattr(a, "agent_id", "?"))
        return {"agents": ok, "excluded": excluded}

    def disagreement_node(s: State) -> dict:
        y = [a.forecast[0] for a in s["agents"]]
        c = [a.confidence for a in s["agents"]]
        dis = disagreement(y, c, cfg.disagreement.floor)
        trig = s.get("trigger")
        r = ("deliberate" if trig else "fast") if trig is not None else route(dis, s["tau"])
        if len(s["agents"]) < 2:
            r = "fast"
        return {
            "D": dis,
            "D_range": d_range(y, c, cfg.disagreement.floor),
            "consensus": consensus(y, c)[0],
            "route": r,
        }

    def consensus_node(s: State) -> dict:
        return {"final": s["consensus"], "deliberated": False, "fallback": None}

    def critic_node(s: State) -> dict:
        res = run_critic(llm, s["case"], s["D"])
        return {
            "critic": res.value,
            **_llm_stats(res),
            "fallback": None if res.value else f"critic invalid: {res.error}",
        }

    def revise_node(s: State) -> dict:
        if s.get("fallback"):
            return {"final": s["consensus"], "deliberated": False, "revisions": []}
        revs, stats = (
            {},
            {"llm_calls": 0, "llm_attempts": 0, "tokens_in": 0, "tokens_out": 0, "llm_errors": []},
        )
        for a in s["agents"]:
            res = run_revision(llm, a, s["critic"], s["case"]["dataset"])
            for k, v in _llm_stats(res).items():
                stats[k] = stats[k] + v
            if res.value is None:
                return {
                    "final": s["consensus"],
                    "deliberated": False,
                    "revisions": [],
                    **stats,
                    "fallback": f"revision invalid ({a.agent_id}): {res.error}",
                }
            revs[a.agent_id] = res.value
        final, rows = apply_revision(s["agents"], revs, d.m_bounds)
        for r in rows:
            r["rationale"] = revs[r["agent_id"]].rationale
        return {"final": final, "deliberated": True, "revisions": rows, **stats}

    def inventory_policy(s: State) -> dict:
        return {"inventory": None}  # Phase 4: (s, S) policy on the final forecast

    g = StateGraph(State)
    for name, fn in [
        ("load_context", load_context),
        ("run_agents", run_agents),
        ("disagreement", disagreement_node),
        ("consensus", consensus_node),
        ("critic", critic_node),
        ("revise", revise_node),
        ("inventory_policy", inventory_policy),
    ]:
        g.add_node(name, timed(name, fn))
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "run_agents")
    g.add_edge("run_agents", "disagreement")
    g.add_conditional_edges(
        "disagreement", lambda s: s["route"], {"fast": "consensus", "deliberate": "critic"}
    )
    g.add_edge("consensus", "inventory_policy")
    g.add_edge("critic", "revise")
    g.add_edge("revise", "inventory_policy")
    g.add_edge("inventory_policy", END)
    return g.compile()


def run_case(graph, case: dict, tau: float, trigger: bool | None = None) -> State:
    return graph.invoke(
        {
            "case": case,
            "tau": tau,
            "trigger": trigger,
            "timings": {},
            "llm_calls": 0,
            "llm_attempts": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "llm_errors": [],
        }
    )
