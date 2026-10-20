"""RetailSense for retailers, as a local app: stores, CSV uploads, forecast runs and results.
API under /api; the built web app (web/out, `npm run build` in web/) is served at /.

It listens on 127.0.0.1 only and has no accounts: whoever uses this computer uses the app, and
store data never leaves it. Requests must name localhost as their host (no DNS rebinding) and
writes must come from the app's own pages (no other website can post to it).

Each run is one subprocess (`retailsense.service.store_run`) on a copy of the store's files,
so a later upload never changes a running or finished run.

Run: python -m retailsense.service.app      (opens http://localhost:<port>/retailer/)
"""

import io
import json
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import holidays
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from retailsense.config import ROOT, load_service
from retailsense.service.store_run import DataError, read_sales, read_stock, read_trends

SVC = load_service()
DATA = ROOT / SVC.data_dir
DB = DATA / "app.db"
HOSTS = ["localhost", "127.0.0.1"]
SCHEMA = """
CREATE TABLE IF NOT EXISTS stores (id INTEGER PRIMARY KEY, settings TEXT NOT NULL,
    created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, store_id INTEGER NOT NULL,
    status TEXT NOT NULL, created REAL NOT NULL, started REAL, finished REAL, error TEXT);
"""
JOBS: queue.Queue[int] = queue.Queue()
UPLOADS = ("sales", "stock", "trends")  # trends (Google Trends download) is optional


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def q(sql: str, *args) -> list[sqlite3.Row]:
    with db() as con:
        return con.execute(sql, args).fetchall()


def store_dir(sid: int):
    return DATA / str(sid)


def run_dir(sid: int, rid: int):
    return store_dir(sid) / "runs" / str(rid)


# ------------------------------------------------------------------ runs
def worker() -> None:
    """One run at a time. ponytail: single worker; a job queue with several workers when stores
    have to wait for each other."""
    while True:
        rid = JOBS.get()
        (row,) = q("SELECT store_id FROM runs WHERE id = ?", rid)
        path = run_dir(row["store_id"], rid)
        q("UPDATE runs SET status = 'running', started = ? WHERE id = ?", time.time(), rid)
        with open(path / "run.log", "w", encoding="utf-8") as log:
            code = subprocess.run(
                [sys.executable, "-W", "ignore", "-m", "retailsense.service.store_run", str(path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"},
                cwd=ROOT,
            ).returncode
        error = None
        if code:
            lines = (path / "run.log").read_text(encoding="utf-8").splitlines()
            user = [ln for ln in lines if ln.startswith("ERROR: ")]
            error = user[-1][7:] if user else "the run failed; see the run log"
        q(
            "UPDATE runs SET status = ?, finished = ?, error = ? WHERE id = ?",
            "failed" if code else "done",
            time.time(),
            error,
            rid,
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    DATA.mkdir(parents=True, exist_ok=True)
    with db() as con:
        con.executescript(SCHEMA)
    q(
        "UPDATE runs SET status = 'failed', error = 'interrupted by a server restart' "
        "WHERE status = 'running'"
    )
    for r in q("SELECT id FROM runs WHERE status = 'queued' ORDER BY id"):
        JOBS.put(r["id"])
    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(title="RetailSense", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=HOSTS)


@app.middleware("http")
async def same_origin_writes(request: Request, call_next):
    """A browser names the page that sent a request in Origin: only this app's pages may write."""
    origin = request.headers.get("origin")
    if request.method != "GET" and origin and origin.split("://")[-1].split(":")[0] not in HOSTS:
        return JSONResponse({"detail": "requests from other sites are refused"}, status_code=403)
    return await call_next(request)


# ------------------------------------------------------------------ stores
class Inventory(BaseModel):
    lead_time: int = Field(ge=1, le=60)
    coverage: int = Field(ge=1, le=60)  # review period R: days an order should cover
    service_level: Literal[0.90, 0.95, 0.99]
    case_pack: int = Field(ge=1, le=1000)


class StoreSettings(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    country: str = Field(min_length=2, max_length=2)  # ISO code, for public holidays
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    inventory: Inventory


def own_store(sid: int) -> sqlite3.Row:
    rows = q("SELECT * FROM stores WHERE id = ?", sid)
    if not rows:
        raise HTTPException(404, "no such store")
    return rows[0]


Store = Annotated[sqlite3.Row, Depends(own_store)]


def _check_country(s: StoreSettings) -> str:
    code = s.country.upper()
    if code not in holidays.list_supported_countries():
        raise HTTPException(422, f"no holiday calendar for country code {code}")
    return s.model_copy(update={"country": code}).model_dump_json()


def _files(sid: int) -> dict:
    out = {}
    for name in UPLOADS:
        meta = store_dir(sid) / f"{name}.json"
        out[name] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None
    return out


def _store(row: sqlite3.Row) -> dict:
    return {"id": row["id"], **json.loads(row["settings"]), "files": _files(row["id"])}


@app.get("/api/stores")
def stores():
    return [_store(r) for r in q("SELECT * FROM stores ORDER BY id")]


@app.post("/api/stores")
def create_store(s: StoreSettings):
    with db() as con:
        sid = con.execute(
            "INSERT INTO stores (settings, created) VALUES (?, ?)",
            (_check_country(s), time.time()),
        ).lastrowid
    return _store(q("SELECT * FROM stores WHERE id = ?", sid)[0])


@app.get("/api/stores/{sid}")
def get_store(store: Store):
    return _store(store)


@app.put("/api/stores/{sid}")
def update_store(s: StoreSettings, store: Store):
    q("UPDATE stores SET settings = ? WHERE id = ?", _check_country(s), store["id"])
    return _store(q("SELECT * FROM stores WHERE id = ?", store["id"])[0])


@app.delete("/api/stores/{sid}")
def delete_store(store: Store):
    """The store, its uploads and every run's files. Not while a run uses them."""
    sid = store["id"]
    if q("SELECT 1 FROM runs WHERE store_id = ? AND status IN ('queued', 'running')", sid):
        raise HTTPException(409, "wait until this store's run has finished")
    with db() as con:
        con.execute("DELETE FROM runs WHERE store_id = ?", (sid,))
        con.execute("DELETE FROM stores WHERE id = ?", (sid,))
    shutil.rmtree(store_dir(sid), ignore_errors=True)
    return {"deleted": sid}


async def _upload(file: UploadFile, sid: int, name: str, read) -> dict:
    limit = SVC.max_upload_mb * 2**20
    body = await file.read(limit + 1)
    if len(body) > limit:
        raise HTTPException(413, f"file larger than {SVC.max_upload_mb} MB")
    try:
        df = read(io.BytesIO(body))
    except DataError as e:
        raise HTTPException(422, str(e)) from None
    meta = {"filename": file.filename, "rows": len(df), "uploaded": time.time()}
    if "item_id" in df:
        meta["items"] = int(df.item_id.nunique())
    if "terms" in df.attrs:
        meta["terms"] = list(df.attrs["terms"].values())
    if "date" in df:
        meta |= {"first": f"{df.date.min():%Y-%m-%d}", "last": f"{df.date.max():%Y-%m-%d}"}
    store_dir(sid).mkdir(parents=True, exist_ok=True)
    (store_dir(sid) / f"{name}.csv").write_bytes(body)
    (store_dir(sid) / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


@app.post("/api/stores/{sid}/sales")
async def upload_sales(file: UploadFile, store: Store):
    return await _upload(file, store["id"], "sales", lambda b: read_sales(b, SVC))


@app.post("/api/stores/{sid}/stock")
async def upload_stock(file: UploadFile, store: Store):
    return await _upload(file, store["id"], "stock", read_stock)


@app.post("/api/stores/{sid}/trends")
async def upload_trends(file: UploadFile, store: Store):
    return await _upload(file, store["id"], "trends", read_trends)


@app.delete("/api/stores/{sid}/trends")
def remove_trends(store: Store):
    """Back to three agents: later runs have no search interest."""
    for ext in ("csv", "json"):
        (store_dir(store["id"]) / f"trends.{ext}").unlink(missing_ok=True)
    return _store(store)


# ------------------------------------------------------------------ runs
@app.post("/api/stores/{sid}/runs")
def start_run(store: Store):
    sid = store["id"]
    if not (store_dir(sid) / "sales.csv").exists():
        raise HTTPException(409, "upload a sales file first")
    if q("SELECT 1 FROM runs WHERE store_id = ? AND status IN ('queued', 'running')", sid):
        raise HTTPException(409, "a run for this store is already queued or running")
    with db() as con:
        rid = con.execute(
            "INSERT INTO runs (store_id, status, created) VALUES (?, 'queued', ?)",
            (sid, time.time()),
        ).lastrowid
    path = run_dir(sid, rid)
    path.mkdir(parents=True)
    for name in UPLOADS:
        if (store_dir(sid) / f"{name}.csv").exists():
            shutil.copy(store_dir(sid) / f"{name}.csv", path / f"{name}.csv")
    settings = json.loads(store["settings"])
    (path / "store.json").write_text(
        json.dumps({"id": sid, "code": f"S{sid}", **settings}), encoding="utf-8"
    )
    JOBS.put(rid)
    return dict(q("SELECT * FROM runs WHERE id = ?", rid)[0])


@app.get("/api/stores/{sid}/runs")
def runs(store: Store):
    rows = q("SELECT * FROM runs WHERE store_id = ? ORDER BY id DESC LIMIT 20", store["id"])
    return [dict(r) for r in rows]


def own_run(rid: int, store: Store) -> sqlite3.Row:
    rows = q("SELECT * FROM runs WHERE id = ? AND store_id = ?", rid, store["id"])
    if not rows:
        raise HTTPException(404, "no such run")
    return rows[0]


Run = Annotated[sqlite3.Row, Depends(own_run)]


@app.get("/api/stores/{sid}/runs/{rid}/log")
def run_log(run: Run):
    log = run_dir(run["store_id"], run["id"]) / "run.log"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return {"lines": lines[-40:]}


@app.get("/api/stores/{sid}/runs/{rid}/result")
def run_result(run: Run):
    path = run_dir(run["store_id"], run["id"]) / "result.json"
    if run["status"] != "done" or not path.exists():
        raise HTTPException(404, "this run has no result")
    return FileResponse(path, media_type="application/json")


WEB = ROOT / "web" / "out"
if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


def main() -> None:
    if not WEB.exists():
        sys.exit("build the web app first: cd web && npm install && npm run build")
    url = f"http://localhost:{SVC.port}/retailer/"
    threading.Timer(2.0, webbrowser.open, [url]).start()
    print(f"RetailSense: {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=SVC.port)


if __name__ == "__main__":
    main()
