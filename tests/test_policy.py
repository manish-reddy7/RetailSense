import ast
import math
from pathlib import Path

import pytest

from retailsense.config import ROOT, load_inventory
from retailsense.inventory.policy import policy

INV = load_inventory()
HALF = 4 * 2.563 / 2  # sigma = 4 per day -> band width 4 * 2.563


def band(f):
    return [x - HALF for x in f], [x + HALF for x in f]


def test_worked_example_8_4():
    """Blueprint 8.4: next days 12, 13, 11 then 12/day (D10 = 120), sigma 4, on hand 30."""
    f = [12, 13, 11, 12, 12, 12, 12]
    d = policy(30, 0, f, *band(f), INV)
    assert d.action == "RESTOCK" and d.qty == 114
    assert d.s == pytest.approx(36 + 1.65 * math.sqrt(48))  # ~47.4; blueprint rounds to 48
    assert d.S == pytest.approx(120 + 1.65 * math.sqrt(160))  # ~140.9; blueprint rounds to 141


def test_hold_and_reduce():
    f = [12] * 7
    assert policy(100, 0, f, *band(f), INV).action == "HOLD"
    d = policy(400, 0, f, *band(f), INV)
    assert d.action == "REDUCE" and d.qty == 0


def test_on_order_counts_toward_position():
    f = [12] * 7
    assert policy(30, 0, f, *band(f), INV).action == "RESTOCK"
    assert policy(30, 60, f, *band(f), INV).action == "HOLD"


def test_case_pack_rounding():
    f = [12] * 7
    q = policy(0, 0, f, *band(f), INV).qty
    assert q % INV.case_pack == 0 and q > 0


def test_determinism():
    f = [5.5, 7.25, 6.0, 9.0, 4.0, 3.5, 8.0]
    assert len({policy(10, 6, f, *band(f), INV) for _ in range(50)}) == 1


@pytest.mark.parametrize("bad", [[float("nan")] * 7, [-1.0] * 7])
def test_rejects_invalid_forecast(bad):
    with pytest.raises(ValueError):
        policy(10, 0, bad, bad, bad, INV)


def test_llm_cannot_set_quantity():
    """No module in the inventory package imports anything from the LLM/deliberation layer, and
    the policy takes only numeric forecast inputs - an LLM can never set an order quantity."""
    llm_modules = {
        "retailsense.orchestrator.llm",
        "retailsense.orchestrator.critic",
        "retailsense.orchestrator.revise",
        "retailsense.orchestrator.graph",
    }
    for path in (ROOT / "src/retailsense/inventory").glob("*.py"):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not imported & llm_modules, path.name
    with pytest.raises((TypeError, ValueError)):
        policy(10, 0, ["order 500 units"] * 7, [0] * 7, [1] * 7, INV)
