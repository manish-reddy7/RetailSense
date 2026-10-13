"""Bounded revision round (blueprint 7.5, Appendix A.2). One round only.

The LLM proposes a confidence multiplier and a revised value; CODE clips the multiplier to
m_bounds and the value to the agent's own [lower, upper] band, then computes the final number:
final = sum(w'_i y'_i), w'_i proportional to c_i * m_i. The LLM never writes the final forecast.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from retailsense.agents.schema import AgentForecast
from retailsense.orchestrator.critic import DECIMALS, CriticReport, agent_view
from retailsense.orchestrator.llm import LLM, LLMResult


class Revision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confidence_multiplier: float
    revised_forecast: float
    rationale: str
    cited_evidence: list[str] = Field(min_length=1)


def check_revision(report: CriticReport):
    def check(r: Revision) -> None:
        bad = set(r.cited_evidence) - report.evidence_keys()
        if bad:
            raise ValueError(f"cited evidence not in the critic report: {sorted(bad)}")
        if not np.isfinite([r.confidence_multiplier, r.revised_forecast]).all():
            raise ValueError("non-finite number")

    return check


def run_revision(llm: LLM, f: AgentForecast, report: CriticReport, dataset: str) -> LLMResult:
    question = next((q.question for q in report.questions_for_agents if q.agent == f.agent_id), "")
    payload = {
        "dataset": dataset,
        "you": agent_view(f),
        "critic_report": report.model_dump(),
        "question_for_you": question,
    }
    return llm.call("revise", payload, Revision, check_revision(report))


def apply_revision(
    agents: list[AgentForecast], revisions: dict[str, Revision], m_bounds: tuple[float, float]
) -> tuple[float, list[dict]]:
    """Clip, re-weight and combine at h=1. An agent without a revision keeps m=1 and its value."""
    lo_m, hi_m = m_bounds
    rows = []
    for a in agents:
        r = revisions.get(a.agent_id)
        m_raw = r.confidence_multiplier if r else 1.0
        y_raw = r.revised_forecast if r else a.forecast[0]
        if abs(y_raw - round(a.forecast[0], DECIMALS)) <= 0.5 * 10**-DECIMALS:
            y_raw = a.forecast[0]  # echoing the displayed (rounded) value is a hold, not a move
        m = min(max(m_raw, lo_m), hi_m)
        y = min(max(y_raw, a.lower[0]), a.upper[0])  # stay inside the agent's own band
        rows.append(
            {
                "agent_id": a.agent_id,
                "m_raw": m_raw,
                "m": m,
                "y_raw": y_raw,
                "y": y,
                "initial": a.forecast[0],
                "c": a.confidence,
                "clipped": (m != m_raw) or (y != y_raw),
                "held": m == 1.0 and abs(y - a.forecast[0]) < 1e-9,
            }
        )
    w = np.array([r["c"] * r["m"] for r in rows])
    w = w / w.sum()
    for r, wi in zip(rows, w, strict=True):
        r["w"] = float(wi)
    return float(np.dot(w, [r["y"] for r in rows])), rows
