"""Shared agent output schema (blueprint 6.1)."""

import math
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentId = Literal["historical", "weather", "trend", "calendar"]


class AgentForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: AgentId
    item_id: str
    store_id: str
    prediction_time: date  # forecast made at end of this day
    horizon_days: list[int] = Field(min_length=1)
    forecast: list[float]  # units per day, same length as horizon_days
    lower: list[float]  # 10th percentile
    upper: list[float]  # 90th percentile
    confidence: float = Field(ge=0.0, le=1.0)
    key_factors: list[str]
    limitations: list[str]

    @model_validator(mode="after")
    def _check(self):
        n = len(self.horizon_days)
        if not (len(self.forecast) == len(self.lower) == len(self.upper) == n):
            raise ValueError("forecast, lower, upper must match horizon_days in length")
        if len(set(self.horizon_days)) != n or min(self.horizon_days) < 1:
            raise ValueError("horizon_days must be unique positive integers")
        for lo, f, hi in zip(self.lower, self.forecast, self.upper, strict=True):
            if any(math.isnan(x) or math.isinf(x) for x in (lo, f, hi)):
                raise ValueError("forecast values must be finite")
            if lo < 0:
                raise ValueError("forecast values must be non-negative")
            if not lo <= f <= hi:
                raise ValueError("need lower <= forecast <= upper")
        return self
