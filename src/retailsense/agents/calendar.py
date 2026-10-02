"""Calendar Agent (blueprint 6.2): residual adjuster on calendar features only."""

from retailsense.agents.residual import ResidualAgent
from retailsense.config import ExperimentCfg
from retailsense.data.features import CALENDAR_FEATURES


def calendar_agent(cfg: ExperimentCfg) -> ResidualAgent:
    return ResidualAgent("calendar", CALENDAR_FEATURES, cfg)
