"""LLM client (blueprint 7.7): pinned model from config, temperature 0, JSON-schema output,
disk cache keyed by hash(model, prompt version, prompt name, input JSON), token/latency logging.

The LLM only ever returns schema-validated objects. Invalid output gets one corrective retry;
after that the caller falls back to fast consensus (`LLMResult.value is None`).
"""

import hashlib
import json
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from retailsense.config import ExperimentCfg, env_value

PROMPTS = Path(__file__).parent / "prompts"
# transport(system, user, schema) -> (text, tokens_in, tokens_out, model_version, latency_ms)
# latency_ms is the model call only - never time spent waiting for the rate limiter.
Transport = Callable[[str, str, type[BaseModel]], tuple[str, int, int, str, float]]


@dataclass
class LLMResult:
    value: BaseModel | None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    calls: int = 0  # API calls actually made (0 if every attempt was a cache hit)
    cache_hit: bool = True
    attempts: int = 0
    model_version: str = ""
    error: str | None = None


def load_prompt(name: str, version: str) -> str:
    return (PROMPTS / f"{name}_{version}.md").read_text(encoding="utf-8")


def rate_limiter(rpm: int) -> Callable[[], None]:
    """Client-side rate limit shared by all worker threads: at most rpm calls a minute."""
    lock, next_slot = threading.Lock(), [0.0]

    def wait_for_slot() -> None:
        with lock:
            now = time.monotonic()
            start = max(now, next_slot[0])
            next_slot[0] = start + 60.0 / rpm
        time.sleep(start - now)

    return wait_for_slot


def openrouter_transport(cfg: ExperimentCfg) -> Transport:
    """OpenRouter chat completions (OpenAI-compatible) with a JSON-schema response format.
    require_parameters routes only to providers that honour the schema and temperature."""
    import requests  # noqa: PLC0415

    d = cfg.deliberation
    key = env_value(d.api_key_env)
    wait_for_slot = rate_limiter(d.rpm)

    def call(system: str, user: str, schema: type[BaseModel]):
        body = {
            "model": d.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": d.temperature,
            "max_tokens": d.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
            "provider": {"require_parameters": True},
        }
        for attempt in range(d.api_retries + 1):
            wait_for_slot()
            t = time.perf_counter()
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=120,
            )
            if r.ok and "choices" in r.json():
                j = r.json()
                u = j.get("usage") or {}
                return (
                    j["choices"][0]["message"].get("content") or "",
                    u.get("prompt_tokens", 0),
                    u.get("completion_tokens", 0),
                    j.get("model", d.model),
                    1000 * (time.perf_counter() - t),
                )
            # a daily quota (free tier) does not reset within the backoff: give up at once
            daily = r.status_code == 429 and "per-day" in r.text
            transient = (r.status_code == 429 and not daily) or r.status_code >= 500
            if attempt == d.api_retries or not transient:
                raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:300]}")
            time.sleep(float(r.headers.get("Retry-After", d.api_backoff_s * 2**attempt)))
        raise AssertionError("unreachable")

    return call


def ollama_transport(cfg: ExperimentCfg) -> Transport:
    """Local Ollama (/api/chat) with the JSON schema as the output format. No key, no quota.
    Server address from OLLAMA_HOST (default http://localhost:11434)."""
    import os  # noqa: PLC0415

    import requests  # noqa: PLC0415

    d = cfg.deliberation
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/chat"

    def call(system: str, user: str, schema: type[BaseModel]):
        body = {
            "model": d.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {
                "temperature": d.temperature,
                "seed": cfg.seed,
                "num_predict": d.max_output_tokens,
                "num_ctx": d.context_tokens,
            },
        }
        # 5xx (e.g. a CUDA out-of-memory crash; Ollama reloads the model) and dropped
        # connections are transient: back off and retry instead of killing the run
        for attempt in range(d.api_retries + 1):
            t = time.perf_counter()
            try:
                r = requests.post(url, json=body, timeout=600)
            except requests.ConnectionError as e:
                r, err = None, f"connection error: {e}"
            else:
                if r.ok:
                    break
                err = f"{r.status_code}: {r.text[:300]}"
            if attempt == d.api_retries or (r is not None and r.status_code < 500):
                raise RuntimeError(f"Ollama {err}")
            time.sleep(d.api_backoff_s * 2**attempt)
        j = r.json()
        return (
            j["message"]["content"],
            j.get("prompt_eval_count", 0),
            j.get("eval_count", 0),
            j.get("model", d.model),
            1000 * (time.perf_counter() - t),
        )

    return call


def gemini_transport(cfg: ExperimentCfg) -> Transport:
    from google import genai  # noqa: PLC0415 - only needed for real calls
    from google.genai import errors, types  # noqa: PLC0415

    d = cfg.deliberation
    client = genai.Client(api_key=env_value(d.api_key_env))
    wait_for_slot = rate_limiter(d.rpm)

    def call(system: str, user: str, schema: type[BaseModel]):
        for attempt in range(d.api_retries + 1):
            wait_for_slot()
            t = time.perf_counter()
            try:
                r = client.models.generate_content(
                    model=d.model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=d.temperature,
                        response_mime_type="application/json",
                        response_json_schema=schema.model_json_schema(),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
                u = r.usage_metadata
                return (
                    r.text or "",
                    u.prompt_token_count or 0,
                    u.candidates_token_count or 0,
                    r.model_version or d.model,
                    1000 * (time.perf_counter() - t),
                )
            except errors.APIError as e:  # retry only transient errors
                if attempt == d.api_retries or not (e.code == 429 or e.code >= 500):
                    raise
                hint = re.search(r"retry in ([0-9.]+)s", str(e))  # server-suggested wait
                time.sleep(float(hint.group(1)) if hint else d.api_backoff_s * 2**attempt)
        raise AssertionError("unreachable")

    return call


def hold_transport(cfg: ExperimentCfg) -> Transport:
    """DRY-RUN STUB, never used for results: the critic flags nothing and every agent holds.
    With it, deliberated systems must equal confidence-weighted consensus - a pipeline check."""

    def call(system: str, user: str, schema: type[BaseModel]):
        p = json.loads(user)
        if "agents" in p:  # critic
            ids = [a["agent_id"] for a in p["agents"]]
            out = {
                "primary_conflict": "dry run",
                "conflicting_agents": [],
                "evidence_assessment": [
                    {"agent": a, "strength": "moderate", "reason": "dry run"} for a in ids
                ],
                "missing_or_stale_evidence": [],
                "questions_for_agents": [],
            }
        else:  # revise: echo the displayed value = hold
            out = {
                "confidence_multiplier": 1.0,
                "revised_forecast": p["you"]["forecast"],
                "rationale": "dry run",
                "cited_evidence": ["primary_conflict"],
            }
        return json.dumps(out), 0, 0, "hold-stub", 0.0

    return call


def offline_transport(cfg: ExperimentCfg) -> Transport:
    """Reproduction from cache: any answer not already cached is an error, never an API call."""

    def call(system: str, user: str, schema: type[BaseModel]):
        raise RuntimeError("LLM cache miss in offline mode")

    return call


class LLM:
    def __init__(self, cfg: ExperimentCfg, transport: Transport | None = None):
        self.cfg = cfg
        self.d = cfg.deliberation
        providers = {
            "google": gemini_transport,
            "openrouter": openrouter_transport,
            "ollama": ollama_transport,
        }
        self.transport = transport or providers[self.d.provider](cfg)
        self.cache = cfg.path("cache") / "llm"
        self.cache.mkdir(parents=True, exist_ok=True)
        # one lock per cache key: identical concurrent requests wait for a single answer, so
        # nobody reads a half-written file and the API is not paid twice
        self._locks, self._guard = defaultdict(threading.Lock), threading.Lock()

    def _key(self, name: str, system: str, user: str) -> str:
        """Model, prompt version AND prompt text: editing a prompt can never reuse old answers."""
        prompt_hash = hashlib.sha256(system.encode()).hexdigest()[:16]
        raw = f"{self.d.model}|{self.d.prompt_version}|{name}|{prompt_hash}|{user}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _raw(self, name: str, system: str, user: str, schema, res: LLMResult) -> str:
        key = self._key(name, system, user)
        path = self.cache / f"{key}.json"
        with self._guard:
            lock = self._locks[key]
        with lock:
            return self._fetch(path, name, system, user, schema, res)

    def _fetch(self, path, name, system, user, schema, res: LLMResult) -> str:
        if path.exists():
            hit = json.loads(path.read_text(encoding="utf-8"))
        else:
            text, tin, tout, version, latency = self.transport(system, user, schema)
            hit = {
                "text": text,
                "tokens_in": tin,
                "tokens_out": tout,
                "model_version": version,
                "latency_ms": latency,
                "prompt": name,
                "prompt_version": self.d.prompt_version,
            }
            path.write_text(json.dumps(hit), encoding="utf-8")
            res.calls += 1
            res.cache_hit = False
        res.tokens_in += hit["tokens_in"]
        res.tokens_out += hit["tokens_out"]
        res.latency_ms += hit["latency_ms"]
        res.model_version = hit["model_version"]
        return hit["text"]

    def call(
        self,
        name: str,
        payload: dict,
        schema: type[BaseModel],
        check: Callable[[BaseModel], None] | None = None,
    ) -> LLMResult:
        """Structured call. `check` raises ValueError for semantic problems (e.g. uncited
        evidence); those count as invalid output exactly like broken JSON."""
        system = load_prompt(name, self.d.prompt_version)
        user = json.dumps(payload, sort_keys=True, default=str)
        res = LLMResult(value=None)
        for attempt in range(self.d.max_invalid_retries + 1):
            res.attempts = attempt + 1
            text = self._raw(name, system, user, schema, res)
            try:
                value = schema.model_validate_json(text)
                if check:
                    check(value)
                res.value, res.error = value, None
                return res
            except (ValidationError, ValueError) as e:
                res.error = f"{type(e).__name__}: {str(e)[:300]}"
                user = json.dumps(
                    {**payload, "previous_answer_was_invalid": res.error},
                    sort_keys=True,
                    default=str,
                )
        return res
