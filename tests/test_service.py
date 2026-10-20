import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from retailsense.service import app as svc_app
from retailsense.service.store_run import DataError, read_sales, write_raw

STORE = {
    "name": "Main St",
    "country": "in",
    "lat": 19.08,
    "lon": 72.88,
    "inventory": {"lead_time": 3, "coverage": 7, "service_level": 0.95, "case_pack": 6},
}


def sales_csv(days: int = 420, gap: str | None = None) -> bytes:
    dates = pd.date_range("2025-01-01", periods=days)
    df = pd.concat(
        pd.DataFrame({"date": dates, "item_id": i, "units_sold": 3, "price": 10.0})
        for i in ("tea", "rice")
    )
    if gap:
        df = df[~((df.item_id == "tea") & (df.date == gap))]
    return df.assign(date=df.date.dt.strftime("%Y-%m-%d")).to_csv(index=False).encode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(svc_app, "DATA", tmp_path)
    monkeypatch.setattr(svc_app, "DB", tmp_path / "app.db")
    with TestClient(svc_app.app, base_url="http://localhost") as c:
        yield c


def test_only_this_computer_and_this_app(client):
    sid = client.post("/api/stores", json=STORE).json()["id"]
    assert client.get(f"/api/stores/{sid}").json()["country"] == "IN"
    assert client.get("/api/stores", headers={"Host": "evil.example"}).status_code == 400
    r = client.post("/api/stores", json=STORE, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403  # another website open in the same browser cannot write
    ok = client.post("/api/stores", json=STORE, headers={"Origin": "http://localhost:8000"})
    assert ok.status_code == 200
    assert client.get("/api/stores/999").status_code == 404


def test_upload_validation_and_run_guards(client):
    sid = client.post("/api/stores", json=STORE).json()["id"]
    assert client.post(f"/api/stores/{sid}/runs").status_code == 409  # no sales yet
    bad = sales_csv().replace(b"units_sold", b"qty")
    r = client.post(f"/api/stores/{sid}/sales", files={"file": ("s.csv", bad)})
    assert r.status_code == 422 and "missing columns" in r.json()["detail"]
    short = client.post(f"/api/stores/{sid}/sales", files={"file": ("s.csv", sales_csv(100))})
    assert short.status_code == 422 and "at least" in short.json()["detail"]
    ok = client.post(f"/api/stores/{sid}/sales", files={"file": ("s.csv", sales_csv())})
    assert ok.json()["items"] == 2
    bad_store = {**STORE, "inventory": {**STORE["inventory"], "service_level": 0.5}}
    assert client.put(f"/api/stores/{sid}", json=bad_store).status_code == 422


def test_missing_day_stays_missing(tmp_path):
    sales = read_sales(io.BytesIO(sales_csv(gap="2025-03-01")), svc_app.SVC)
    n, d0 = write_raw(sales, {"code": "S1", "country": "IN"}, tmp_path, 7)
    assert pd.Timestamp(d0).day_name() == "Saturday"
    wide = pd.read_csv(tmp_path / "sales_train_evaluation.csv").set_index("item_id")
    gap = (pd.Timestamp("2025-03-01") - pd.Timestamp(d0)).days + 1
    assert pd.isna(wide.loc["tea", f"d_{gap}"]) and wide.loc["rice", f"d_{gap}"] == 3
    assert wide.loc["tea", [f"d_{n + h}" for h in range(1, 8)]].isna().all()  # the future
    assert pd.isna(wide.loc["tea", "d_1"])  # d_1 is the Saturday before the first sale (a Wed)


def test_duplicate_rows_rejected():
    dup = sales_csv() + b"2025-01-01,tea,3,10.0\n"
    with pytest.raises(DataError, match="same item and date"):
        read_sales(io.BytesIO(dup), svc_app.SVC)


def trends_csv(weeks: int = 60, period: str = "Week") -> bytes:
    rows = [f"{d:%Y-%m-%d},{40 + i % 5},{'<1' if i == 3 else 7}" for i, d in
            enumerate(pd.date_range("2024-01-07", periods=weeks, freq="7D"))]
    head = ["Category: All categories", "", f"{period},ice cream: (India),Chai!: (India)"]
    return "\n".join(head + rows).encode()


def test_trends_upload_and_remove(client):
    sid = client.post("/api/stores", json=STORE).json()["id"]
    short = client.post(f"/api/stores/{sid}/trends", files={"file": ("t.csv", trends_csv(20))})
    assert short.status_code == 422 and "at least 52" in short.json()["detail"]
    monthly = client.post(
        f"/api/stores/{sid}/trends", files={"file": ("t.csv", trends_csv(period="Month"))}
    )
    assert monthly.status_code == 422 and "weekly" in monthly.json()["detail"]
    ok = client.post(f"/api/stores/{sid}/trends", files={"file": ("t.csv", trends_csv())})
    assert ok.json()["terms"] == ["ice cream", "Chai!"] and ok.json()["rows"] == 60
    assert client.get(f"/api/stores/{sid}").json()["files"]["trends"]["rows"] == 60
    gone = client.delete(f"/api/stores/{sid}/trends").json()
    assert gone["files"]["trends"] is None  # later runs: three agents


def test_trends_reach_the_research_parser(tmp_path):
    """The upload is rewritten per term in the layout data/trends.py reads, values unchanged."""
    from retailsense.config import load_experiment
    from retailsense.data.trends import trends_table
    from retailsense.service.store_run import read_trends, with_trends

    t = read_trends(io.BytesIO(trends_csv()))
    assert list(t.columns) == ["date", "ice_cream", "chai"]  # safe feature names
    cfg = with_trends(load_experiment(), t, tmp_path)
    tab = trends_table(cfg)
    assert {"tr_ice_cream_rel52", "tr_chai_z52"} <= set(tab.columns)
    assert (tab.available_at - tab.week_start == pd.Timedelta(days=7)).all()  # after the week ends
    assert tab.tr_ice_cream_rel52.iloc[:51].isna().all()  # 52 weeks of history, never filled
    assert tab.tr_ice_cream_rel52.iloc[51:].notna().all()
