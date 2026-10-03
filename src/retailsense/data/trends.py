"""Google Trends weekly interest (blueprint 4.2), from manually downloaded CSVs.

Files: data/raw/trends/<keyword with _ for spaces>__<start>_<end>.csv, one keyword per file,
geo US-CA, web search, downloaded once and never re-queried. The file's modification time is
recorded as its retrieval timestamp.

Trends scales each window to its own maximum, and that maximum can lie in the window's future.
So only SCALE-INVARIANT features are built (ratio to trailing mean, % change, z-score): multiplying
a window by any constant leaves them unchanged. The raw index level is never a feature.
A week (Sun-Sat) is available from the day after it ends.

Run: python -m retailsense.data.trends
"""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from retailsense.config import ROOT, ExperimentCfg, load_experiment

WEEKS = 52


def slug(keyword: str) -> str:
    return keyword.replace(" ", "_")


def read_trends_csv(path: Path, lt1: float) -> pd.Series:
    """Weekly values indexed by week start. Skips Google's header lines; '<1' -> lt1."""
    rows = []
    for rec in csv.reader(path.read_text(encoding="utf-8-sig").splitlines()):
        if len(rec) < 2:
            continue
        try:
            week = pd.Timestamp(datetime.strptime(rec[0].strip(), "%Y-%m-%d"))
        except ValueError:
            continue  # header line
        v = rec[1].strip()
        rows.append((week, lt1 if v == "<1" else float(v)))
    if not rows:
        raise ValueError(f"no weekly rows in {path}")
    s = pd.Series(dict(rows)).sort_index()
    if not (s.index.to_series().diff().dropna() == pd.Timedelta(days=7)).all():
        raise ValueError(f"{path} is not weekly data (window too long? keep it under 5 years)")
    return s


def stitch(windows: list[pd.Series]) -> pd.Series:
    """Chain windows in start order, rescaling each onto the previous via the overlap ratio."""
    windows = sorted(windows, key=lambda s: s.index[0])
    out = windows[0]
    for nxt in windows[1:]:
        both = out.index.intersection(nxt.index)
        both = both[(out[both] > 0) & (nxt[both] > 0)]
        if len(both) == 0:
            raise ValueError("trends windows do not overlap")
        k = out[both].sum() / nxt[both].sum()
        out = pd.concat([out, nxt[nxt.index > out.index[-1]] * k])
    return out


def weekly_features(s: pd.Series, name: str) -> pd.DataFrame:
    """Scale-invariant features per week. Trailing windows include the week itself."""
    mean = s.rolling(WEEKS, WEEKS).mean()
    std = s.rolling(WEEKS, WEEKS).std()
    prev = s.shift(4)
    return pd.DataFrame({
        f"tr_{name}_rel52": s / mean,
        f"tr_{name}_chg4": (s / prev - 1).where(prev > 0),
        f"tr_{name}_z52": ((s - mean) / std).where(std > 0),
    })


def trends_table(cfg: ExperimentCfg) -> pd.DataFrame:
    """One row per week: available_at (day after the week ends), features for every keyword."""
    d = ROOT / cfg.interest.trends_dir
    parts, missing = [], []
    for kw in cfg.interest.keywords:
        files = sorted(d.glob(f"{slug(kw)}__*.csv"))
        if not files:
            missing.append(kw)
            continue
        s = stitch([read_trends_csv(f, cfg.interest.trends_lt1) for f in files])
        parts.append(weekly_features(s, slug(kw)))
    if missing:
        raise FileNotFoundError(f"no Trends CSVs in {d} for: {missing}")
    out = pd.concat(parts, axis=1)  # weeks missing for one keyword stay NaN
    out.index.name = "week_start"
    out = out.reset_index()
    out["available_at"] = out.week_start + pd.Timedelta(days=7)  # Sunday after the Sat week end
    return out


def retrieval_log(cfg: ExperimentCfg) -> dict[str, str]:
    d = ROOT / cfg.interest.trends_dir
    return {
        f.name: pd.Timestamp(f.stat().st_mtime, unit="s").isoformat()
        for f in sorted(d.glob("*.csv"))
    }


def main() -> None:
    cfg = load_experiment()
    t = trends_table(cfg)
    path = cfg.path("processed") / "trends.parquet"
    t.to_parquet(path, index=False)
    print(f"wrote {path}: {len(t)} weeks {t.week_start.min():%Y-%m-%d}..{t.week_start.max():%Y-%m-%d}")
    print("retrieved:", retrieval_log(cfg))
    print("NaN share:", np.round(t.drop(columns=["week_start", "available_at"]).isna().mean().mean(), 4))


if __name__ == "__main__":
    main()
