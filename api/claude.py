"""
Claude — fully standalone, OpenAI-compatible router for the claude.ai web backend.

Design goals (đại ca's requirements):
  * COMPLETELY separate from chatgpt / gemini / the shared flow. Everything Claude
    lives in THIS one file, so editing it can never break the other providers.
  * Still speaks the OpenAI standard: request/response use the OpenAI
    chat.completions schema, exposed under its own base path:
        POST /v1/claude/chat/completions      (stream + non-stream)
        GET  /v1/claude/models
    Point any OpenAI client/SDK at base_url ".../v1/claude" and it just works.

Backend: "free" path — drives claude.ai's own web API with a logged-in session
cookie (consumes the account's Claude.ai quota, no per-token billing), mirroring
how chatgpt_free uses chatgpt.com. Paid API / OAuth subscription is a future add.

Wire-up (one line, added to api/app.py — the full chatgpt2api app only):
    app.include_router(claude.create_router())

Config (config.json → providers.claude):
    "claude": { "session_key": "sk-ant-sid01-...", "model": "auto",
                "timezone": "Asia/Ho_Chi_Minh" }
Get session_key from claude.ai → DevTools → Application → Cookies → `sessionKey`.

NOTE: claude.ai is reverse-engineered + Cloudflare-protected; endpoints drift.
Untested against the live site from the dev box — verify on the server with a
real cookie and adjust the completion path / _parse_stream event shapes if needed.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

from curl_cffi import requests
from fastapi import APIRouter, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

# ── Lazy imports (avoid import cycles at module load) ───────────────────────

def _config():
    from services.config import config
    return config


def _logger():
    from utils.log import logger
    return logger


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

CLAUDE_BASE_URL = "https://claude.ai"
ROOT_PARENT_UUID = "00000000-0000-4000-8000-000000000000"

# Friendly aliases → claude.ai internal model ids. Unknown values pass through;
# "auto"/"" → omit model field (let claude.ai pick the account default).
CLAUDE_MODEL_ALIASES: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4.5": "claude-sonnet-4-5",
    "opus": "claude-opus-4-8",
    "opus-4.8": "claude-opus-4-8",
    "opus-4.1": "claude-opus-4-1",
    "haiku": "claude-haiku-4-5",
    "haiku-4.5": "claude-haiku-4-5",
}


# ═══════════════════════════════════════════════════════════════════════════
# Request model (OpenAI standard)
# ═══════════════════════════════════════════════════════════════════════════

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "claude/auto"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _claude_cfg() -> dict[str, Any]:
    cfg = (_config().data.get("providers") or {}).get("claude") or {}
    return cfg if isinstance(cfg, dict) else {}


def _base_url() -> str:
    base = str(_claude_cfg().get("base_url") or "").rstrip("/")
    return base or CLAUDE_BASE_URL


# sessionKey fetched from the captcha-solver, cached per profile (5-min TTL)
# so we don't hit the onboard service on every chat request.
_SOLVER_KEY_TTL = 300.0
_solver_key_cache: dict[str, tuple[float, str]] = {}


def _fetch_session_key_from_solver(cfg: dict[str, Any]) -> str:
    """Pull a logged-in claude.ai sessionKey from the captcha-solver.

    Reuses the same Google-account onboard mechanism as ChatGPT/Flow:
    config providers.claude.captcha_solver_url + captcha_solver_api_key +
    profile (or profiles[] — first one that has a key wins).
    """
    base = str(cfg.get("captcha_solver_url") or "").rstrip("/")
    if not base:
        return ""
    profiles = cfg.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        profiles = [str(cfg.get("profile") or "claude-web-default")]
    api_key = str(cfg.get("captcha_solver_api_key") or "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    for profile in profiles:
        profile = str(profile).strip()
        if not profile:
            continue
        cached = _solver_key_cache.get(profile)
        if cached and (time.time() - cached[0]) < _SOLVER_KEY_TTL and cached[1]:
            return cached[1]
        try:
            resp = requests.get(
                f"{base}/v1/claude-web/{profile}/session",
                headers=headers, timeout=15, impersonate="chrome110",
            )
            if resp.status_code == 200:
                key = str((resp.json() or {}).get("session_key") or "")
                if key:
                    _solver_key_cache[profile] = (time.time(), key)
                    return key
        except Exception as exc:
            _logger().warning({"event": "claude_solver_key_fetch_failed", "profile": profile, "error": str(exc)})
    return ""


def _resolve_model(model: str) -> str:
    """alias/prefixed model → claude.ai internal id; '' means auto (omit)."""
    m = str(model or "").strip()
    for pfx in ("cc/", "claude/", "clf/", "cl/"):
        if m.startswith(pfx):
            m = m[len(pfx):].strip()
            break
    
    # Strip effort and thinking suffixes
    for sfx in ("-low", "-medium", "-high", "-max", "-thinking", "-think"):
        if m.endswith(sfx):
            m = m[:-len(sfx)]
    
    if not m or m == "auto":
        m = str(_claude_cfg().get("model") or "").strip()
    if not m or m == "auto":
        return ""
    return CLAUDE_MODEL_ALIASES.get(m, m)


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """OpenAI message array → single claude.ai prompt string.

    claude.ai keeps history server-side and takes one `prompt`, so for a
    stateless OpenAI-style request we serialise the whole conversation into one
    turn with System/User/Assistant prefixes (same approach as reference clients).
    """
    parts: list[str] = []
    for msg in messages or []:
        role = str(msg.get("role") or "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        else:
            text = str(content or "")
        if not text.strip():
            continue
        label = {"system": "System", "assistant": "Assistant", "user": "User"}.get(role, role.capitalize())
        parts.append(f"{label}: {text}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _collect_text(chunks: Iterator[dict[str, Any]]) -> str:
    """Drain an OpenAI-chunk iterator into the full assistant text. Blocking —
    call via run_in_threadpool so claude.ai's network reads don't stall the loop."""
    content = ""
    for chunk in chunks:
        content += chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
    return content


def _openai_chunk(model: str, cid: str, created: int, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


# ═══════════════════════════════════════════════════════════════════════════
# claude.ai web backend
# ═══════════════════════════════════════════════════════════════════════════

class ClaudeFreeBackend:
    """claude.ai web backend via session cookie. Returns OpenAI-format chunks."""

    def __init__(self) -> None:
        self._session: Any = None
        self._session_cookie: str = ""
        self._org_id: str = ""

    def _cookie_header(self) -> str:
        cfg = _claude_cfg()
        full = str(cfg.get("cookie") or "").strip()
        if full:
            return full
        key = str(cfg.get("session_key") or "").strip()
        if key:
            return f"sessionKey={key}"
        # No static cookie → reuse a Google-account session onboarded via the
        # captcha-solver (same path as ChatGPT/Flow login).
        key = _fetch_session_key_from_solver(cfg)
        return f"sessionKey={key}" if key else ""

    @property
    def is_available(self) -> bool:
        return bool(self._cookie_header())

    @property
    def session(self):
        cookie = self._cookie_header()
        if not cookie:
            raise RuntimeError("Claude: missing providers.claude.session_key / cookie / captcha_solver_url")
        # Rebuild the session when the cookie changes (sessionKey rotation or a
        # different Google profile) so we never send a stale credential.
        if self._session is None or self._session_cookie != cookie:
            s = requests.Session(impersonate="chrome110")
            s.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Content-Type": "application/json",
                "Origin": CLAUDE_BASE_URL,
                "Referer": CLAUDE_BASE_URL + "/chats",
                "anthropic-client-platform": "web_claude_ai",
                "Cookie": cookie,
            })
            self._session = s
            self._session_cookie = cookie
            self._org_id = ""  # re-resolve org for the new credential
        return self._session

    def _org_id_get(self) -> str:
        if self._org_id:
            return self._org_id
        pinned = str(_claude_cfg().get("organization_uuid") or "").strip()
        if pinned:
            self._org_id = pinned
            return pinned
        resp = self.session.get(f"{_base_url()}/api/organizations", timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"Claude org lookup failed {resp.status_code}: {resp.text[:160]}")
        orgs = resp.json()
        org_id = ""
        if isinstance(orgs, list) and orgs:
            chat_orgs = [o for o in orgs if isinstance(o, dict) and "chat" in (o.get("capabilities") or [])]
            org_id = (chat_orgs[0] if chat_orgs else orgs[0]).get("uuid", "")
        elif isinstance(orgs, dict):
            org_id = orgs.get("uuid", "")
        if not org_id:
            raise RuntimeError("Claude org lookup: no organization (session_key expired?)")
        self._org_id = org_id
        return org_id

    def _create_conversation(self, org_id: str) -> str:
        conv = str(uuid.uuid4())
        url = f"{_base_url()}/api/organizations/{org_id}/chat_conversations"
        resp = self.session.post(url, json={"uuid": conv, "name": ""}, timeout=30)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Claude create-conversation failed {resp.status_code}: {resp.text[:160]}")
        try:
            return resp.json().get("uuid", conv)
        except Exception:
            return conv

    def chat(self, messages: list[dict[str, Any]], model: str) -> Iterator[dict[str, Any]]:
        """Always-streaming generator of OpenAI chat.completion.chunk dicts."""
        if not self.is_available:
            raise RuntimeError("Claude not configured (providers.claude.session_key)")

        org_id = self._org_id_get()
        conv_id = self._create_conversation(org_id)
        internal_model = _resolve_model(model)

        payload: dict[str, Any] = {
            "prompt": _flatten_messages(messages),
            "parent_message_uuid": ROOT_PARENT_UUID,
            "timezone": str(_claude_cfg().get("timezone") or "Asia/Ho_Chi_Minh"),
            "attachments": [],
            "files": [],
            "sync_sources": [],
            "rendering_mode": "messages",
        }
        if internal_model:
            payload["model"] = internal_model
            
        raw_m = str(model or "").lower()
        effort = None
        if "-low" in raw_m: effort = "low"
        elif "-medium" in raw_m: effort = "medium"
        elif "-high" in raw_m: effort = "high"
        elif "-max" in raw_m: effort = "max"
        
        thinking = "-thinking" in raw_m or "-think" in raw_m
        
        if thinking or effort:
            payload["thinking"] = {"type": "adaptive"}
            if effort:
                payload["output_config"] = {"effort": effort}

        url = f"{_base_url()}/api/organizations/{org_id}/chat_conversations/{conv_id}/completion"
        _logger().info({"event": "claude_request", "model": internal_model or "auto", "msg_count": len(messages or [])})

        resp = self.session.post(
            url, json=payload, timeout=300, stream=True,
            headers={"Accept": "text/event-stream"},
        )
        if resp.status_code != 200:
            body = ""
            try:
                body = resp.text[:200]
            except Exception:
                pass
            raise RuntimeError(f"Claude completion failed {resp.status_code}: {body}")
        return self._parse_stream(resp, model)

    def _parse_stream(self, response, model: str) -> Iterator[dict[str, Any]]:
        cid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        sent_role = False
        try:
            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
                line = line.strip()
                if not line or line.startswith(":") or line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue

                etype = str(event.get("type") or "")
                if etype == "error" or event.get("error"):
                    raise RuntimeError(f"Claude stream error: {str(event.get('error') or event)[:200]}")

                # Incremental text across known claude.ai event shapes.
                text = ""
                if isinstance(event.get("completion"), str):
                    text = event["completion"]
                elif isinstance(event.get("delta"), dict):
                    text = str(event["delta"].get("text") or "")
                elif etype == "content_block_delta":
                    text = str((event.get("delta") or {}).get("text") or "")

                if text:
                    if not sent_role:
                        sent_role = True
                        yield _openai_chunk(model, cid, created, {"role": "assistant", "content": text})
                    else:
                        yield _openai_chunk(model, cid, created, {"content": text})

                if etype in ("message_stop", "completion_stop") or event.get("stop_reason"):
                    break
        except Exception as exc:
            _logger().error({"event": "claude_stream_error", "error": str(exc)})
            if not sent_role:
                yield _openai_chunk(model, cid, created, {"role": "assistant", "content": f"[claude error] {exc}"})
        finally:
            try:
                response.close()
            except Exception:
                pass

        if not sent_role:
            yield _openai_chunk(model, cid, created, {"role": "assistant", "content": ""})
        yield _openai_chunk(model, cid, created, {}, finish="stop")


_backend = ClaudeFreeBackend()


# ═══════════════════════════════════════════════════════════════════════════
# Router (OpenAI-compatible, dedicated /v1/claude/* path)
# ═══════════════════════════════════════════════════════════════════════════

def create_router() -> APIRouter:
    router = APIRouter(prefix="/v1/claude", tags=["claude"])

    @router.post("/chat/completions")
    async def claude_chat_completions(
        body: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ):
        """OpenAI-format chat completion served by the claude.ai web backend."""
        from api.support import require_identity
        require_identity(authorization)

        model = str(body.model or "claude/auto")
        messages = body.messages or []

        if not _backend.is_available:
            raise HTTPException(
                status_code=503,
                detail={"error": "Claude not configured: set providers.claude.session_key in config.json"},
            )

        # Build the (blocking) claude.ai iterator off the event loop.
        try:
            chunks = await run_in_threadpool(_backend.chat, messages, model)
        except Exception as exc:
            _logger().error({"event": "claude_chat_failed", "error": str(exc)})
            raise HTTPException(status_code=502, detail={"error": f"Claude backend error: {exc}"})

        if body.stream:
            def sse() -> Iterator[str]:
                for chunk in chunks:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            # Starlette iterates this sync generator in a threadpool, so the
            # blocking claude.ai reads inside it never stall the event loop.
            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-stream: collect off the event loop.
        content = await run_in_threadpool(_collect_text, chunks)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @router.get("/models")
    async def claude_models(authorization: str | None = Header(default=None)):
        from api.support import require_identity
        require_identity(authorization)
        ids = ["claude/auto", "claude/sonnet-4.5"]
        for b in ["sonnet-4.6", "opus-4.8", "haiku-4.5"]:
            for e in ["", "-medium", "-high", "-max"]:
                for t in ["", "-thinking"]:
                    ids.append(f"claude/{b}{e}{t}")
        return {
            "object": "list",
            "data": [{"id": i, "object": "model", "owned_by": "claude"} for i in ids],
        }

    return router
