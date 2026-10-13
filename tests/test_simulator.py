import pytest

from retailsense.config import load_inventory
from retailsense.inventory.simulator import inventory_metrics, simulate_item

INV = load_inventory()


def const_forecast(level, sigma=0.0):
    half = sigma * INV.band_to_sigma / 2
    return lambda origin: ([level] * 7, [level - half] * 7, [level + half] * 7)


def test_perfect_constant_forecast_never_stocks_out():
    days = list(range(100, 160))
    log = simulate_item(
        days, {d: 10 for d in days}, {d: 2.0 for d in days}, const_forecast(10), INV
    )
    assert (log.lost == 0).all() and log.sold.sum() == 600


def test_orders_arrive_after_lead_time():
    days = list(range(1, 31))
    log = simulate_item(
        days, {d: 10 for d in days}, {d: 1.0 for d in days}, const_forecast(10), INV
    ).set_index("d")
    first = log[log.order_qty > 0].index[0]
    before = log.loc[first + INV.lead_time - 1, "on_hand"]
    arrive_day = log.loc[first + INV.lead_time]
    # on arrival day: stock = yesterday's closing + order - today's sales
    assert arrive_day.on_hand == pytest.approx(before + log.loc[first, "order_qty"] - 10)


def test_underforecast_causes_lost_sales_and_costs():
    days = list(range(1, 21))
    log = simulate_item(days, {d: 20 for d in days}, {d: 3.0 for d in days}, const_forecast(5), INV)
    m = inventory_metrics(log)
    assert m["lost_units"] > 0 and m["stockout_rate"] > 0 and m["fill_rate"] < 1
    assert m["lost_sales_cost"] == pytest.approx(m["lost_units"] * 3.0 * INV.lost_sales_mult)
    assert m["total_cost"] == pytest.approx(
        m["holding_cost"] + m["ordering_cost"] + m["lost_sales_cost"]
    )


def test_starting_stock_is_order_up_to_level():
    days = list(range(1, 3))
    log = simulate_item(days, {1: 0, 2: 0}, {1: 1.0, 2: 1.0}, const_forecast(10), INV)
    assert log.on_hand.iloc[0] == pytest.approx(10 * (INV.lead_time + INV.coverage))
