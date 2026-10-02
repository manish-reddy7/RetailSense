"""Pydantic config loader for configs/experiment.yaml and configs/inventory.yaml."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _day(v: str | int) -> int:
    """'d_1799' -> 1799."""
    return int(str(v).removeprefix("d_"))


class Paths(_Strict):
    raw: Path
    processed: Path
    cache: Path
    db: Path
    subset: Path


class DataCfg(_Strict):
    store: str
    state: str
    dept: str
    top_n: int
    rank_end: int
    train_end: int
    val_end: int
    test_end: int

    @field_validator("rank_end", "train_end", "val_end", "test_end", mode="before")
    @classmethod
    def _parse_day(cls, v):
        return _day(v)

    @model_validator(mode="after")
    def _ordered(self):
        if not self.rank_end <= self.train_end < self.val_end < self.test_end:
            raise ValueError("need rank_end <= train_end < val_end < test_end")
        return self


class WeatherCfg(_Strict):
    city: dict[str, tuple[float, float]]
    vars: list[str]
    timezone: str
    fetch_start: str
    fetch_end: str
    clim_years: int
    clim_half_window: int
    hot_anom_c: float
    rainy_mm: float
    change_days: int


class InterestCfg(_Strict):
    trends_dir: Path
    trends_geo: str
    keywords: dict[str, str]  # search term -> feature name
    trends_lt1: float  # value used for Google's "<1"


class AgentsCfg(_Strict):
    horizons: list[int]
    confidence_window: int
    confidence_min_days: int
    oof_folds: int


class DisagreementCfg(_Strict):
    method: str
    floor: float
    tau_quantile: float


class AnalysisCfg(_Strict):
    worst_share: float
    bootstrap: int
    trigger_rates: list[float]


class DeliberationCfg(_Strict):
    rounds: int
    m_bounds: tuple[float, float]
    provider: str
    model: str
    api_key_env: str
    temperature: float
    max_output_tokens: int
    context_tokens: int
    prompt_version: str
    history_days: int
    max_invalid_retries: int
    api_retries: int
    api_backoff_s: float
    rpm: int
    sample_size: int
    random_trigger_draws: int
    workers: int
    tau: float | None


class LgbmCfg(_Strict):
    num_threads: int
    early_stop_days: int
    early_stopping_rounds: int
    max_rounds: int
    quantiles: tuple[float, float]
    historical: dict[str, float | int]
    residual: dict[str, float | int]
    key_factors: int


class EtsCfg(_Strict):
    season: int
    fit_window: int
    refit_every: int
    band_alpha: float


class FinalCfg(_Strict):
    """Phase 5a (blueprint 9.4-9.5)."""

    backtest_windows: list[tuple[int, int]]
    fixed_tau: float
    weather_noise_sd: float
    curve_step: float
    curve_draws: int
    lead_times: list[int]

    @field_validator("backtest_windows", mode="before")
    @classmethod
    def _parse_windows(cls, v):
        return [(_day(a), _day(b)) for a, b in v]


class MlflowCfg(_Strict):
    tracking_uri: str


class ExperimentCfg(_Strict):
    dataset: str  # tags every MLflow experiment
    seed: int
    paths: Paths
    data: DataCfg
    weather: WeatherCfg
    interest: InterestCfg
    agents: AgentsCfg
    disagreement: DisagreementCfg
    analysis: AnalysisCfg
    deliberation: DeliberationCfg
    lgbm: LgbmCfg
    ets: EtsCfg
    mlflow: MlflowCfg
    final: FinalCfg | None = None

    def path(self, name: str) -> Path:
        return ROOT / getattr(self.paths, name)


class InventoryCfg(_Strict):
    lead_time: int
    coverage: int
    service_level: float
    z_table: dict[float, float]
    band_to_sigma: float
    holding_rate: float
    days_per_year: int
    order_cost: float
    lost_sales_mult: float
    case_pack: int
    naive_band_days: int
    sensitivity: dict[str, list[float]]

    @property
    def z(self) -> float:
        return self.z_table[self.service_level]


def env_value(key: str) -> str:
    """Secret from the environment, else from the gitignored .env file. Never from config."""
    env = ROOT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    dotenv = dict(ln.split("=", 1) for ln in lines if "=" in ln and not ln.startswith("#"))
    value = os.environ.get(key) or dotenv.get(key, "").strip().strip('"')
    if not value:
        raise RuntimeError(f"set {key} in .env")
    return value


def _read(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_experiment(path: Path | None = None) -> ExperimentCfg:
    """Default config, or the one named by the RETAILSENSE_CONFIG environment variable."""
    path = path or Path(os.environ.get("RETAILSENSE_CONFIG", ROOT / "configs/experiment.yaml"))
    return ExperimentCfg(**_read(ROOT / path))


def load_inventory(path: Path = ROOT / "configs/inventory.yaml") -> InventoryCfg:
    return InventoryCfg(**_read(path))
