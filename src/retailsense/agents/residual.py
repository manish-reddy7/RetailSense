"""Residual-adjuster agent (blueprint 6.3): forecast_i = b + g_i(b, x_i).

g_i is trained on out-of-fold baseline residuals (y - b_oof), pooled over horizons, using the
baseline plus the agent's own evidence only. The agent's band is the baseline band shifted by
its adjustment.
"""

import numpy as np
import pandas as pd

from retailsense.agents.historical import expanding_oof, fit_early_stopped, lgb_params
from retailsense.config import ExperimentCfg


class ResidualAgent:
    def __init__(self, agent_id: str, evidence: list[str], cfg: ExperimentCfg):
        self.agent_id = agent_id
        self.features = ["baseline", *evidence]
        self.cfg = cfg

    def _fit(self, oof: pd.DataFrame):
        df = oof.assign(residual=oof.sales - oof.baseline)
        params = lgb_params(self.cfg, "residual", "regression")
        return fit_early_stopped(df, self.features, "residual", params, self.cfg)

    def fit(self, oof: pd.DataFrame) -> "ResidualAgent":
        self.model = self._fit(oof)
        gain = self.model.feature_importance("gain")
        self.importance = pd.Series(gain, index=self.model.feature_name()).sort_values(
            ascending=False
        )
        return self

    def adjustment(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features])

    def oof_adjustment(self, oof: pd.DataFrame) -> pd.DataFrame:
        """Nested expanding-fold adjustments over the OOF period, so the agent's own track record
        before validation is out-of-sample (used for confidence, blueprint 6.4)."""
        out = expanding_oof(
            oof.drop(columns=["fold", "fold_train_end"]),
            self.cfg.agents.oof_folds,
            self._fit,
            lambda m, d: m.predict(d[self.features]),
        )
        return out.rename(columns={"oof_pred": "adjustment"})

    def apply(self, base: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
        """base: Historical output (item_id, d, forecast, lower, upper) aligned row-for-row
        with feats. Shifts forecast and band by the adjustment, clipped at 0."""
        adj = self.adjustment(feats.assign(baseline=base.forecast.to_numpy()))
        out = base[["item_id", "d", "h"]].copy()
        for col in ("forecast", "lower", "upper"):
            out[col] = np.clip(base[col].to_numpy() + adj, 0, None)
        out["adjustment"] = adj
        return out

    def key_factors(self, row: pd.Series, adjustment: float) -> list[str]:
        top = [f for f in self.importance.index if f != "baseline"][: self.cfg.lgbm.key_factors]
        facts = [f"{f} = {row[f]}" if pd.notna(row[f]) else f"{f} none" for f in top]
        return [f"{self.agent_id} adjustment {adjustment:+.2f}", *facts]
