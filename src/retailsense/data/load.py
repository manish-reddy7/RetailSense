"""Load the raw sales files, select the frozen subset (store/dept/top-N by train volume), write
configs/subset.yaml.

Run: python -m retailsense.data.load
"""

import numpy as np
import pandas as pd
import yaml

from retailsense.config import ExperimentCfg, load_experiment

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def day_cols(first: int, last: int) -> list[str]:
    return [f"d_{i}" for i in range(first, last + 1)]


def load_sales(cfg: ExperimentCfg) -> pd.DataFrame:
    """Wide sales for the configured store and dept (d_1 .. test_end)."""
    df = pd.read_csv(cfg.path("raw") / "sales_train_evaluation.csv")
    df = df[(df.store_id == cfg.data.store) & (df.dept_id == cfg.data.dept)]
    return df[ID_COLS + day_cols(1, cfg.data.test_end)].reset_index(drop=True)


def load_calendar(cfg: ExperimentCfg) -> pd.DataFrame:
    return pd.read_csv(cfg.path("raw") / "calendar.csv", parse_dates=["date"])


def load_prices(cfg: ExperimentCfg, items: list[str]) -> pd.DataFrame:
    df = pd.read_csv(cfg.path("raw") / "sell_prices.csv")
    return df[(df.store_id == cfg.data.store) & df.item_id.isin(items)].reset_index(drop=True)


def select_top_items(wide: pd.DataFrame, rank_end: int, top_n: int) -> list[str]:
    """Top-N items by total units over d_1..d_rank_end. Ties broken by item_id."""
    totals = wide.set_index("item_id")[day_cols(1, rank_end)].sum(axis=1)
    ranked = totals.reset_index(name="units").sort_values(
        ["units", "item_id"], ascending=[False, True]
    )
    return ranked.item_id.head(top_n).tolist()


def zero_share(wide: pd.DataFrame, first: int, last: int) -> float:
    return float((wide[day_cols(first, last)].to_numpy() == 0).mean())


def subset_summary(wide: pd.DataFrame, cfg: ExperimentCfg) -> dict:
    d = cfg.data
    splits = {"train": (1, d.train_end), "val": (d.train_end + 1, d.val_end),
              "test": (d.val_end + 1, d.test_end)}
    units = wide[day_cols(1, d.rank_end)].sum(axis=1)
    return {
        "n_items": len(wide),
        "zero_share": {k: round(zero_share(wide, *v), 4) for k, v in splits.items()},
        "train_units_per_item": {"min": int(units.min()), "median": float(np.median(units)),
                                 "max": int(units.max())},
    }


def write_subset(items: list[str], cfg: ExperimentCfg) -> None:
    d = cfg.data
    doc = {
        "frozen": "Do not change after Week 1 (blueprint 4.1).",
        "store": d.store,
        "dept": d.dept,
        "rule": f"top {d.top_n} items by total units over d_1..d_{d.rank_end} (train only)",
        "items": items,
    }
    cfg.path("subset").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def load_subset_items(cfg: ExperimentCfg) -> list[str]:
    return yaml.safe_load(cfg.path("subset").read_text(encoding="utf-8"))["items"]


def main() -> None:
    cfg = load_experiment()
    wide = load_sales(cfg)
    items = select_top_items(wide, cfg.data.rank_end, cfg.data.top_n)
    write_subset(items, cfg)
    print(yaml.safe_dump(subset_summary(wide[wide.item_id.isin(items)], cfg), sort_keys=False))


if __name__ == "__main__":
    main()
