import pandas as pd
import pytest

from retailsense.data.load import load_calendar
from retailsense.forecast_future import extended_calendar


def test_extended_calendar_continues_the_data_calendar(cfg):
    if not (cfg.path("raw") / "calendar.csv").exists():
        pytest.skip("raw India data not generated")
    cal = load_calendar(cfg)
    ext = extended_calendar(cfg, len(cal) + 7)
    pd.testing.assert_frame_equal(ext.iloc[: len(cal)], cal, check_dtype=False)  # past untouched
    assert len(ext) == len(cal) + 7
    assert (ext.date.diff().dropna() == pd.Timedelta(days=1)).all()
    assert ext.d.tolist() == [f"d_{i}" for i in range(1, len(ext) + 1)]
    assert (ext.wday == (ext.index % 7) + 1).all()  # Sat-Fri weeks continue


def test_llm_outage_falls_back_to_consensus(monkeypatch):
    """A quota error mid-run: answered items replay, fast items still run, and every item that
    needs the LLM afterwards gets the consensus with the reason, never a crash or a zero."""
    from retailsense import forecast_future as ff

    answered = {"a"}  # in the disk cache from before the outage

    def fake_run_all(cfg, cases, tau, trigger=None):
        out = []
        for c in cases:
            if c["item_id"] in {"b", "d"}:
                raise RuntimeError("OpenRouter 429: free-models-per-day")
            out.append({"case": c, "deliberated": c["item_id"] in answered, "llm_calls": 0})
        return out

    monkeypatch.setattr(ff, "run_all", fake_run_all)
    cases = [{"item_id": i} for i in "abcd"]
    states = ff.without_llm(None, cases, 0.1, needs_llm={"a", "b", "d"}, error="429")
    by = {s["case"]["item_id"]: s for s in states}
    assert by["a"]["deliberated"] and not by["a"].get("fallback")  # replayed
    assert by["b"]["fallback"].startswith("LLM unavailable") and not by["b"]["deliberated"]
    assert not by["c"].get("fallback")  # fast path, no LLM needed
    assert by["d"]["fallback"].startswith("LLM unavailable")  # skipped once the API is down
