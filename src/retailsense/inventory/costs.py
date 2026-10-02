"""Daily inventory costs (blueprint 8.2). All rates come from configs/inventory.yaml."""

from retailsense.config import InventoryCfg


def day_costs(on_hand_end: float, qty: int, lost: float, price: float, inv: InventoryCfg) -> dict:
    return {
        "holding_cost": on_hand_end * price * inv.holding_rate / inv.days_per_year,
        "ordering_cost": inv.order_cost if qty > 0 else 0.0,
        "lost_sales_cost": lost * price * inv.lost_sales_mult,
    }
