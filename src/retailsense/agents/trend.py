"""Trend Agent (blueprint 6.2): residual adjuster on interest features only."""

from retailsense.agents.residual import ResidualAgent
from retailsense.config import ExperimentCfg
from retailsense.data.trends import slug


def trend_features(cfg: ExperimentCfg) -> list[str]:
    kw = cfg.interest.keywords
    return [f"tr_{slug(k)}_{s}" for k in kw for s in ("rel52", "chg4", "z52")]


def trend_agent(cfg: ExperimentCfg) -> ResidualAgent:
    return ResidualAgent("trend", trend_features(cfg), cfg)
