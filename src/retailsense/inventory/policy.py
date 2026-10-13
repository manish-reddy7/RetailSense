"""(s, S) inventory policy (blueprint 8.3, Appendix A.3). Pure function of numbers: the forecast,
its 10-90% band and the stock position. No LLM output can reach it except through the final
forecast, which code computes and bounds.
"""

import math
from dataclasses import dataclass

import numpy as np

from retailsense.config import InventoryCfg


@dataclass(frozen=True)
class Decision:
    action: str  # RESTOCK / HOLD / REDUCE
    qty: int
    s: float  # reorder point = D_L + SS
    S: float  # order-up-to level
    safety_stock: float
    excess_level: float  # S + D_7: above this inventory position counts as overstock


def _ext(xs: list[float], n: int) -> list[float]:
    """First n days, reusing the last (day-7) forecast beyond the horizon."""
    return (list(xs) + [xs[-1]] * n)[:n]


def policy(on_hand: float, on_order: float, forecast, lower, upper, inv: InventoryCfg) -> Decision:
    f, lo, hi = (np.asarray(x, float) for x in (forecast, lower, upper))
    if not (np.isfinite(f).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("forecast and band must be finite")
    if (f < 0).any() or (hi < lo).any():
        raise ValueError("forecast must be non-negative and upper >= lower")
    z, L, R = inv.z, inv.lead_time, inv.coverage
    sig = list((hi - lo) / inv.band_to_sigma)
    f = list(f)
    safety = z * math.sqrt(sum(x * x for x in _ext(sig, L)))
    s = sum(_ext(f, L)) + safety
    S = sum(_ext(f, L + R)) + z * math.sqrt(sum(x * x for x in _ext(sig, L + R)))
    excess = S + sum(_ext(f, 7))
    ip = on_hand + on_order
    if ip <= s:
        qty = int(math.ceil((S - ip) / inv.case_pack) * inv.case_pack)
        return Decision("RESTOCK", qty, s, S, safety, excess)
    return Decision("REDUCE" if ip > excess else "HOLD", 0, s, S, safety, excess)
