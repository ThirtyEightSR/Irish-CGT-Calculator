from __future__ import annotations

# Irish CGT share-identification order per S.580/S.581 TCA 1997: same-day
# acquisitions first, then acquisitions in the following 4 weeks (the "bed
# and breakfast" rule), then remaining earlier acquisitions on a FIFO basis.
# This ordering applies to share/security disposals only — Exit Tax (ETF)
# disposals are not subject to S.581 and must keep using plain FIFO.

from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

BED_AND_BREAKFAST_WINDOW_DAYS = 28


@dataclass
class Lot:
    date: pd.Timestamp
    qty: float
    unit_cost: float


def match_disposal(sell_date: pd.Timestamp, sell_qty: float, lots: Sequence[Lot]) -> float:
    """Match a disposal against lots using same-day, 4-week B&B, then FIFO order.

    `lots` must contain every acquisition lot for the instrument (past and
    future relative to `sell_date`); each `Lot.qty` is decremented in place as
    it is consumed. Returns the matched cost basis for the disposal.
    """
    sell_date = pd.Timestamp(sell_date).normalize()
    remaining = float(sell_qty)
    cost = 0.0

    def _consume(candidates: List[Lot]) -> None:
        nonlocal remaining, cost
        for lot in sorted(candidates, key=lambda lot: lot.date):
            if remaining <= 1e-12:
                return
            if lot.qty <= 1e-12:
                continue
            take = min(remaining, lot.qty)
            cost += take * lot.unit_cost
            lot.qty -= take
            remaining -= take

    _consume([lot for lot in lots if lot.date == sell_date])

    if remaining > 1e-12:
        window_end = sell_date + pd.Timedelta(days=BED_AND_BREAKFAST_WINDOW_DAYS)
        _consume([lot for lot in lots if sell_date < lot.date <= window_end])

    if remaining > 1e-12:
        _consume([lot for lot in lots if lot.date < sell_date])

    return cost
