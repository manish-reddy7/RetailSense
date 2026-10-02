"""Weather Agent (blueprint 6.2): residual adjuster on weather features only."""

from retailsense.agents.residual import ResidualAgent
from retailsense.config import ExperimentCfg
from retailsense.data.weather import WEATHER_FEATURES


def weather_agent(cfg: ExperimentCfg) -> ResidualAgent:
    return ResidualAgent("weather", WEATHER_FEATURES, cfg)
