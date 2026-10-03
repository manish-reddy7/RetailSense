"""Rohlik Sales Forecasting Challenge v2 (Kaggle) -> the raw file layout the pipeline reads.

Real daily sales of an online grocer, Aug 2020 - Jun 2024. One warehouse (config `source`).
Kaggle's own test period has no public sales, so only sales_train.csv is used and the
experiment cuts its own chronological splits from it.

A day without a sales record is MISSING (the product was not offered), never zero. Items are
kept only if they have a record on every day from `complete_from` to the end, so every scored
day and every baseline input is observed; earlier gaps stay NaN and are skipped in training.
Price = the weekly mean of sell_price_main (list price; discounts are not used).

Run: RETAILSENSE_CONFIG=configs/experiment_rohlik.yaml python -m retailsense.data.rohlik
"""

import pandas as pd

from retailsense.config import ROOT, load_experiment

RELIGIOUS = {"Christmas Eve", "1st Christmas Day", "2nd Christmas Day", "Good Friday",
             "Easter Monday"}
CULTURAL = {"International womens day"}


def convert(cfg) -> None:
    src = cfg.source
    sdir, out = ROOT / src.dir, cfg.path("raw")
    out.mkdir(parents=True, exist_ok=True)
    d0 = pd.Timestamp(src.start)
    if d0.day_name() != "Saturday":
        raise ValueError("d_1 must be a Saturday (weeks run Sat-Fri, wday 1 = Saturday)")
    store, state = cfg.data.store, cfg.data.state

    s = pd.read_csv(sdir / "sales_train.csv", parse_dates=["date"])
    s = s[s.warehouse == src.warehouse]
    s["d"] = (s.date - d0).dt.days + 1
    s = s[s.d >= 1]
    n = int(s.d.max())
    if n < cfg.data.test_end:
        raise ValueError(f"data ends at d_{n}, config test_end is d_{cfg.data.test_end}")
    inv = pd.read_csv(sdir / "inventory.csv")
    names = inv[inv.warehouse == src.warehouse].set_index("unique_id")
    s["item_id"] = s.unique_id.map(names.name)

    wide = s.pivot(index="item_id", columns="d", values="sales").reindex(
        columns=range(1, n + 1)
    )
    keep = wide.loc[:, src.complete_from :].notna().all(axis=1)
    wide = wide[keep]
    print(f"{keep.sum()} of {len(keep)} items have a record on every day d_{src.complete_from}+")
    cat = s.drop_duplicates("item_id").set_index("item_id").unique_id.map(
        names.L1_category_name_en
    )
    ids = pd.DataFrame(
        {
            "id": [f"{i}_{store}" for i in wide.index],
            "item_id": wide.index,
            "dept_id": cfg.data.dept,
            "cat_id": cat.reindex(wide.index).to_numpy(),
            "store_id": store,
            "state_id": state,
        }
    )
    wide.columns = [f"d_{c}" for c in wide.columns]
    pd.concat([ids, wide.reset_index(drop=True)], axis=1).to_csv(
        out / "sales_train_evaluation.csv", index=False
    )

    dates = pd.date_range(d0, periods=n)
    c = pd.read_csv(sdir / "calendar.csv", parse_dates=["date"])
    c = c[c.warehouse == src.warehouse].set_index("date").reindex(dates)
    hol = c.holiday_name.where(c.holiday_name.notna(), None)
    i = range(n)
    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": [11001 + k // 7 for k in i],
            "weekday": dates.day_name(),
            "wday": [k % 7 + 1 for k in i],
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{k + 1}" for k in i],
            "event_name_1": hol.to_numpy(),
            "event_type_1": [
                None if h is None else "Religious" if h in RELIGIOUS
                else "Cultural" if h in CULTURAL else "National"
                for h in hol
            ],
            "event_name_2": None,
            "event_type_2": None,
            f"snap_{state}": 0,  # no SNAP programme; column kept for the shared layout
        }
    )
    cal.to_csv(out / "calendar.csv", index=False)

    p = s[s.item_id.isin(wide.index)].assign(wm_yr_wk=lambda x: 11001 + (x.d - 1) // 7)
    p = p.groupby(["item_id", "wm_yr_wk"], as_index=False).sell_price_main.mean()
    p.insert(0, "store_id", store)
    p.rename(columns={"sell_price_main": "sell_price"}).to_csv(
        out / "sell_prices.csv", index=False
    )
    (out / "README.md").write_text(
        f"# Rohlik Prague_1 in the raw file layout\n\nConverted by `python -m "
        f"retailsense.data.rohlik` from the Kaggle Rohlik Sales Forecasting Challenge v2 "
        f"(sales_train.csv, real sales). Warehouse {src.warehouse}, d_1 = {src.start}, "
        f"{n} days, {len(wide)} items with a record on every day from d_{src.complete_from}. "
        "Missing days are left empty (not offered), never zero.\n",
        encoding="utf-8",
    )
    print(f"wrote {out}: {len(wide)} items x {n} days, {len(p)} weekly prices")


def main() -> None:
    convert(load_experiment())


if __name__ == "__main__":
    main()
