"""Daily inventory simulator (blueprint 8.1). Identical for every system; only forecasts differ.

Per item and day t: orders arriving on t are received, demand for t is revealed, sold =
min(on_hand, demand), lost = the rest. Then the forecast made at the END of day t (data up to t
only) drives the policy; an order placed on t arrives at the start of day t + lead_time.
Starting stock = S from the forecast made the day before the window.
"""

from collections.abc import Callable

import pandas as pd

from retailsense.config import InventoryCfg
from retailsense.inventory.costs import day_costs
from retailsense.inventory.policy import policy

# origin -> (forecast[7], lower[7], upper[7]) or None when no forecast exists for that origin
Forecaster = Callable[[int], tuple[list, list, list] | None]


def simulate_item(
    days: list[int], demand: dict, price: dict, forecast: Forecaster, inv: InventoryCfg
) -> pd.DataFrame:
    start = forecast(days[0] - 1)
    if start is None:
        raise ValueError("need a forecast made the day before the window to set starting stock")
    on_hand = float(policy(0, 0, *start, inv).S)
    pipeline: list[tuple[int, int]] = []  # (arrival day, qty)
    rows = []
    for t in days:
        on_hand += sum(q for a, q in pipeline if a == t)
        pipeline = [(a, q) for a, q in pipeline if a > t]
        dem = float(demand[t])
        sold = min(on_hand, dem)
        lost = dem - sold
        on_hand -= sold
        on_order = float(sum(q for _, q in pipeline))
        f = forecast(t)
        dec = policy(on_hand, on_order, *f, inv) if f is not None else None
        qty = dec.qty if dec else 0
        if qty > 0:
            pipeline.append((t + inv.lead_time, qty))
        rows.append(
            {
                "d": t,
                "demand": dem,
                "sold": sold,
                "lost": lost,
                "on_hand": on_hand,
                "on_order": on_order,
                "action": dec.action if dec else "NONE",
                "order_qty": qty,
                "reorder_point": dec.s if dec else None,
                "order_up_to": dec.S if dec else None,
                "safety_stock": dec.safety_stock if dec else None,
                "overstock": max(0.0, on_hand + on_order - dec.excess_level) if dec else 0.0,
                **day_costs(on_hand, qty, lost, price[t], inv),
            }
        )
    return pd.DataFrame(rows)


def inventory_metrics(log: pd.DataFrame) -> dict:
    """Blueprint 8.5, over all item-days in the log."""
    total = log.holding_cost.sum() + log.ordering_cost.sum() + log.lost_sales_cost.sum()
    return {
        "stockout_rate": float((log.lost > 0).mean()),
        "fill_rate": float(log.sold.sum() / max(log.demand.sum(), 1)),
        "lost_units": float(log.lost.sum()),
        "overstock": float(log.overstock.mean()),
        "reduce_flags": int((log.action == "REDUCE").sum()),
        "orders": int((log.order_qty > 0).sum()),
        "holding_cost": float(log.holding_cost.sum()),
        "ordering_cost": float(log.ordering_cost.sum()),
        "lost_sales_cost": float(log.lost_sales_cost.sum()),
        "total_cost": float(total),
    }
