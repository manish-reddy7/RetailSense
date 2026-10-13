"""Critic Agent (blueprint 7.4). Called only when D >= tau. Explains the conflict and judges each
agent's evidence. The schema has NO numeric field: the critic cannot output a demand number."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from retailsense.agents.schema import AgentForecast, AgentId
from retailsense.orchestrator.llm import LLM, LLMResult

Strength = Literal["strong", "moderate", "weak"]
DECIMALS = 2  # numbers shown to the LLM are rounded to this many decimals
EVIDENCE = {
    "historical": "recent sales history and price",
    "calendar": "weekday, month, holidays and festivals",
    "weather": "temperature and precipitation for the target day and their anomalies",
    "trend": "public search interest in related products",
}


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: AgentId
    strength: Strength
    reason: str


class AgentQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: AgentId
    question: str


class CriticReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary_conflict: str
    conflicting_agents: list[AgentId]
    evidence_assessment: list[EvidenceAssessment]
    missing_or_stale_evidence: list[AgentId]
    questions_for_agents: list[AgentQuestion]

    def evidence_keys(self) -> set[str]:
        return {a.agent for a in self.evidence_assessment} | {"primary_conflict"}


def check_report(agent_ids: set[str]):
    """Every agent assessed exactly once; nothing about agents that are not in the case."""

    def check(r: CriticReport) -> None:
        assessed = [a.agent for a in r.evidence_assessment]
        if sorted(assessed) != sorted(agent_ids):
            # sorted: a set's repr changes order per process, which would change the retry
            # prompt and miss the LLM cache on replay
            raise ValueError(
                f"assess each agent exactly once: got {assessed}, need {sorted(agent_ids)}"
            )
        named = set(r.conflicting_agents) | set(r.missing_or_stale_evidence)
        named |= {q.agent for q in r.questions_for_agents}
        if not named <= agent_ids:
            raise ValueError(f"unknown agents: {sorted(named - agent_ids)}")

    return check


def agent_view(f: AgentForecast) -> dict:
    """What any LLM sees about an agent: its h=1 forecast, band, confidence and evidence."""
    return {
        "agent_id": f.agent_id,
        "evidence": EVIDENCE[f.agent_id],
        "forecast": round(f.forecast[0], DECIMALS),
        "lower": round(f.lower[0], DECIMALS),
        "upper": round(f.upper[0], DECIMALS),
        "confidence": round(f.confidence, 3),
        "key_factors": f.key_factors,
        "limitations": f.limitations,
    }


def critic_payload(case: dict, d: float) -> dict:
    """tau is deliberately NOT shown, so a case's deliberation does not depend on the threshold."""
    return {
        "dataset": case["dataset"],
        "item_id": case["item_id"],
        "store_id": case["store_id"],
        "forecast_for": str(case["target_date"]),
        "recent_daily_sales_oldest_first": case["history"],
        "agents": [agent_view(f) for f in case["agents"]],
        "disagreement_D": round(d, 4),
    }


def run_critic(llm: LLM, case: dict, d: float) -> LLMResult:
    ids = {f.agent_id for f in case["agents"]}
    return llm.call("critic", critic_payload(case, d), CriticReport, check_report(ids))
