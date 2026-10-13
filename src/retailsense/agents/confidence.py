"""Agent confidence (blueprint 6.4): c = 1 / (1 + WAPE over the last `window` revealed days)."""

import pandas as pd

from retailsense.eval.metrics import wape


def trailing_confidence(track: pd.DataFrame, origins, window: int, min_days: int) -> pd.DataFrame:
    """track: (item_id, d, y, yhat) - the agent's out-of-sample forecast for each day and the
    realised demand. At origin p only days in [p - window + 1, p] are used. Items with fewer
    than `min_days` such days fall back to the pooled (department-level) WAPE."""
    items = track.item_id.unique()
    parts = []
    for p in origins:
        w = track[(track.d > p - window) & (track.d <= p)].dropna(subset=["y", "yhat"])
        if w.empty:
            raise ValueError(f"no revealed forecasts before origin {p}")
        g = w.assign(err=(w.y - w.yhat).abs()).groupby("item_id", observed=True)
        per = g.agg(n=("y", "size"), err=("err", "sum"), ysum=("y", "sum")).reindex(items)
        item_wape = per.err / per.ysum.clip(lower=1)
        fallback = per.n.fillna(0) < min_days
        c = 1 / (1 + item_wape.where(~fallback, wape(w.y, w.yhat)))
        parts.append(
            pd.DataFrame(
                {
                    "item_id": items,
                    "origin": p,
                    "confidence": c.to_numpy(),
                    "fallback": fallback.to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
