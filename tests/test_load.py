import pandas as pd

from retailsense.data.load import select_top_items, zero_share


def wide(rows):
    return pd.DataFrame(
        [
            {"item_id": k, **{f"d_{i + 1}": v for i, v in enumerate(vals)}}
            for k, vals in rows.items()
        ]
    )


def test_top_items_ranked_on_rank_window_only():
    # B is biggest overall only because of d_3 (outside the ranking window).
    w = wide({"A": [5, 5, 0], "B": [1, 1, 100], "C": [3, 3, 0]})
    assert select_top_items(w, rank_end=2, top_n=2) == ["A", "C"]


def test_ties_broken_by_item_id():
    w = wide({"B": [1, 1], "A": [1, 1]})
    assert select_top_items(w, rank_end=2, top_n=1) == ["A"]


def test_zero_share():
    w = wide({"A": [0, 2, 0, 4]})
    assert zero_share(w, 1, 4) == 0.5
    assert zero_share(w, 2, 2) == 0.0
