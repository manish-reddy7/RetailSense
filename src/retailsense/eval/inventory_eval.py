"""Phase 4 (blueprint 8, 9.1): every system through the SAME simulator, parameters and starting
rule on validation (d_{train_end+1}..d_{val_end}); plus one-at-a-time sensitivity runs.

Systems 1-6 of blueprint 9.1. Systems 7-9 (random trigger, always-on, RetailSense) are added
once full-validation deliberation is available (needs a paid LLM quota).

Run: python -m retailsense.eval.inventory_eval
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from retailsense.config import (  # noqa: E402,E501
    ROOT,
    ExperimentCfg,
    InventoryCfg,
    load_experiment,
    load_inventory,
)
from retailsense.data.load import load_calendar, load_prices  # noqa: E402
from retailsense.inventory.simulator import inventory_metrics, simulate_item  # noqa: E402
from retailsense.store.db import connect, log_evaluation  # noqa: E402
from retailsense.store.tracking import start_run  # noqa: E402

SYSTEMS = {  # blueprint 9.1 number -> name
    1: "seasonal_naive",
    2: "ets",
    3: "lgbm_all_features",
    4: "historical",
    5: "multi_agent_mean",
    6: "multi_agent_confidence",
}


def naive_band(inv: InventoryCfg, sales: pd.DataFrame, end: int) -> pd.DataFrame:
    """10-90% quantiles of the same-weekday-last-week error per item, from days <= end (the
    models' fit end) only."""
    s = sales[sales.d <= end].pivot(index="d", columns="item_id", values="sales")
    err = (s - s.shift(7)).loc[end - inv.naive_band_days + 1 : end]
    return pd.DataFrame({"q_lo": err.quantile(0.1), "q_hi": err.quantile(0.9)})


def system_forecasts(
    cfg: ExperimentCfg, inv: InventoryCfg, sales, tag: str = "phase2", fit_end: int | None = None
) -> dict[str, pd.DataFrame]:
    """(item_id, origin, h, yhat, lower, upper) per system 1-6, for every origin of a window."""
    proc = cfg.path("processed")
    allf = pd.read_parquet(proc / f"{tag}_all_forecasts.parquet").astype({"item_id": str})
    out = {
        k: allf[allf.system == k].drop(columns="system")
        for k in ("seasonal_naive", "ets", "lgbm_all_features", "historical")
    }
    b = naive_band(inv, sales, cfg.data.train_end if fit_end is None else fit_end)
    n = out["seasonal_naive"].join(b, on="item_id")
    out["seasonal_naive"] = n.assign(
        lower=np.minimum(n.yhat, np.clip(n.yhat + n.q_lo, 0, None)),
        upper=np.maximum(n.yhat, n.yhat + n.q_hi),
    ).drop(columns=["q_lo", "q_hi"])

    af = pd.read_parquet(proc / f"{tag}_agent_forecasts.parquet").astype({"item_id": str})
    af["w_eq"] = 1.0
    for name, wcol in (("multi_agent_mean", "w_eq"), ("multi_agent_confidence", "confidence")):
        w = af[wcol] / af.groupby(["item_id", "origin", "h"])[wcol].transform("sum")
        comb = af.assign(yhat=af.forecast * w, lower=af.lower * w, upper=af.upper * w)
        out[name] = comb.groupby(["item_id", "origin", "h"], as_index=False)[
            ["yhat", "lower", "upper"]
        ].sum()  # blueprint 7.6: weighted bands
    return out


def forecaster(fc: pd.DataFrame, horizons: list[int], last_day: int | None = None):
    """Near the end of the data (origin + h > last_day) a forecast has fewer horizons; the policy
    then repeats its last day, the same for every system."""
    table = {}
    for (item, origin), g in fc.groupby(["item_id", "origin"]):
        hs = [h for h in horizons if last_day is None or origin + h <= last_day]
        g = g.set_index("h").reindex(hs)
        if not g[["yhat", "lower", "upper"]].notna().all().all():
            raise ValueError(f"incomplete forecast for {item} at origin {origin}")
        table[(item, origin)] = (g.yhat.tolist(), g.lower.tolist(), g.upper.tolist())
    return lambda item: lambda origin: table.get((item, origin))


def run(
    cfg, inv, fcs, demand, price, days, systems=SYSTEMS, last_day=None
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, logs = [], {}
    items = sorted(demand)
    for num, name in systems.items():
        get = forecaster(fcs[name], cfg.agents.horizons, last_day)
        log = pd.concat(
            [
                simulate_item(days, demand[i], price[i], get(i), inv).assign(item_id=i)
                for i in items
            ],
            ignore_index=True,
        )
        logs[name] = log
        rows.append({"#": num, "system": name, **inventory_metrics(log)})
    return pd.DataFrame(rows).set_index("system"), logs


def sensitivity(cfg, inv, fcs, demand, price, days) -> pd.DataFrame:
    rows = []
    for param, values in inv.sensitivity.items():
        for v in values:
            alt = inv.model_copy(update={param: type(getattr(inv, param))(v)})
            table, _ = run(cfg, alt, fcs, demand, price, days)
            for system, r in table.iterrows():
                rows.append(
                    {
                        "param": param,
                        "value": v,
                        "system": system,
                        "total_cost": r.total_cost,
                        "fill_rate": r.fill_rate,
                        "stockout_rate": r.stockout_rate,
                    }
                )
            print(f"  sensitivity {param} = {v} done", flush=True)
    return pd.DataFrame(rows)


def sensitivity_figure(sens: pd.DataFrame, path, title: str) -> None:
    params = list(dict.fromkeys(sens.param))
    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 3.8), sharey=True)
    for ax, p in zip(axes, params, strict=True):
        for system, g in sens[sens.param == p].groupby("system", sort=False):
            ax.plot(g.value, g.total_cost / 1000, marker="o", label=system)
        ax.set(title=p, xlabel=p)
    axes[0].set_ylabel("total inventory cost ($ thousands)")
    axes[-1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    cfg, inv = load_experiment(), load_inventory()
    d = cfg.data
    days = list(range(d.train_end + 1, d.val_end + 1))
    feats = pd.read_parquet(
        cfg.path("processed") / "features.parquet", columns=["item_id", "d", "h", "sales"]
    )
    sales = feats[feats.h == 1].drop(columns="h").astype({"item_id": str})
    val = sales[sales.d.isin(days)]
    demand = {i: g.set_index("d").sales.to_dict() for i, g in val.groupby("item_id")}

    cal = load_calendar(cfg).assign(d=lambda c: c.d.str.removeprefix("d_").astype(int))
    prices = load_prices(cfg, sorted(demand)).merge(cal[["d", "wm_yr_wk"]], on="wm_yr_wk")
    prices = prices[prices.d.isin(days)]
    price = {i: g.set_index("d").sell_price.to_dict() for i, g in prices.groupby("item_id")}
    missing = [(i, t) for i in demand for t in days if t not in price.get(i, {})]
    if missing:
        raise ValueError(f"no sell price for {len(missing)} item-days, e.g. {missing[:3]}")

    fcs = system_forecasts(cfg, inv, sales)
    print(f"simulating {len(SYSTEMS)} systems x {len(demand)} items x {len(days)} days", flush=True)
    table, logs = run(cfg, inv, fcs, demand, price, days)
    sens = sensitivity(cfg, inv, fcs, demand, price, days)

    label = cfg.dataset
    out, figs = ROOT / "reports" / cfg.dataset, ROOT / "reports" / "figures" / cfg.dataset
    table.to_csv(out / "phase4_inventory.csv")
    sens.to_csv(out / "phase4_sensitivity.csv", index=False)
    sensitivity_figure(
        sens,
        figs / "inventory_sensitivity.png",
        f"Inventory cost sensitivity on validation - {label}",
    )

    con = connect(cfg.path("db"))
    con.execute("DELETE FROM inventory_state WHERE run_id LIKE 'phase4-%'")
    for name, log in logs.items():
        con.executemany(
            "INSERT INTO inventory_state VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    f"phase4-{name}-{r.item_id}-{r.d}",
                    r.on_hand,
                    r.on_order,
                    r.safety_stock,
                    r.reorder_point,
                    r.order_up_to,
                    r.action,
                    r.order_qty,
                    r.sold,
                    r.lost,
                )
                for r in log.itertuples()
            ],
        )
    log_evaluation(
        con, "phase4_validation", [{"system": s, **r} for s, r in table.to_dict("index").items()]
    )
    con.commit()
    with start_run(cfg, "phase4", "inventory_validation"):
        mlflow.log_params(
            {f"inv.{k}": v for k, v in inv.model_dump().items() if k != "sensitivity"}
        )
        mlflow.log_metrics(
            {f"{s}.{m}": float(v) for s, r in table.iterrows() for m, v in r.items() if m != "#"}
        )
        mlflow.log_text(table.round(4).to_string(), "inventory_table.txt")
        mlflow.log_artifact(str(figs / "inventory_sensitivity.png"))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table.round(3))
        print(
            sens.pivot_table(index=["param", "value"], columns="system", values="total_cost").round(
                0
            )
        )


if __name__ == "__main__":
    main()
