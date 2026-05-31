"""
ChatGPT Free provider — STANDALONE, free-tier-only chat handler.

This module owns the ENTIRE free path so it can be debugged in isolation:
  - selects tokens from the free pool only (account_group == "free")
  - calls chatgpt.com/backend-api via OpenAIBackendAPI / text_backend
  - rotates within the free pool on 429 / expiry / payload-too-large
  - vision via /backend-api/files (free JWT works); falls back to gemini_free
    only when the free pool is completely empty

It deliberately contains NO codex / openai-api branches — those live in their
own providers (openai_oauth / openai_api). Decided 2026-05-29 with đại ca:
"tách hoàn toàn code của chatgpt free, độc lập để dễ debug".

Heavy helpers (streaming, HA prefetch, completion builders) are imported
lazily from services.protocol.openai_v1_chat_complete to avoid a circular
import — by the time handle_free_chat() runs at request time, that module is
fully loaded.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator

from services.account_service import account_service
from services.config import config
from utils.log import logger


def _normalize_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """chatgpt.com native backend does NOT support role="tool" messages.
    Convert tool results to user messages so re-dispatch after agentic tool
    execution doesn't 400."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "tool":
            tool_name = m.get("name", "UnknownTool")
            out.append({
                "role": "user",
                "content": f"[KẾT QUẢ TỪ HỆ THỐNG - TOOL {tool_name}]:\n{m.get('content', '')}",
            })
        else:
            out.append(m)
    return out


def _normalize_free_model(model: str) -> str:
    """Strip ONLY the new free-routing prefixes (`free/`, `chatgpt/free/`) and
    pass everything else through unchanged.

    The resulting slug is sent verbatim to chatgpt.com via
    `_conversation_payload(..., "model": model)`, so we must NOT rewrite slugs
    the production paths already rely on: `chatgpt/auto` (HA / saved combos) and
    bare names like `gpt-4o` are left exactly as the old free branch sent them.
    Empty → "auto"."""
    slug = str(model or "").strip()
    for p in ("chatgpt/free/", "cgf/", "free/"):
        if slug.startswith(p):
            slug = slug[len(p):]
            break
    return slug or "auto"


# ── cgf/auto round-robin (đại ca's choice "A", 2026-05-31) ────────────────────
# cgf/auto no longer forwards the literal "auto" to chatgpt.com's own picker.
# Instead it cycles through the *enabled* cgf/* models so free-tier quota is
# spread across models (each request advances to the next model). Empty pool
# → fall back to ChatGPT's native "auto".
_rr_lock = threading.Lock()
_rr_index = 0

# Slugs kept OUT of the auto-rotation pool even when enabled:
#   auto      — would recurse into itself.
#   research  — chatgpt.com deep-research is a long-running job, not a normal
#               chat model; rotating into it would randomly stall ~1/N requests
#               on a multi-minute task. Still callable explicitly via cgf/research.
_AUTO_ROTATE_EXCLUDE = {"auto", "research"}


_FREE_PREFIXES = ("chatgpt/free/", "cgf/", "free/")


def _enabled_free_models() -> list[str]:
    """Enabled FREE model slugs ONLY (ids under cgf/ , free/ , chatgpt/free/),
    prefix stripped, deduped, sorted for a stable round-robin order.

    CRITICAL: ids WITHOUT a free prefix (cx/ codex, gemini*/, deepseek/, flow/
    images, cgw/ chatgpt-web, combo names like "AI Agent", ...) are NOT free-pool
    models and MUST be skipped — otherwise cgf/auto rotates into other providers'
    models and forwards a bogus model name to chatgpt.com. (Robust to the
    chatgpt_free / ChatGPT_free key-casing split since we scan every key but
    filter by prefix.)"""
    ms = config.data.get("model_settings") or {}
    enabled = ms.get("enabled_models") or {}
    slugs: set[str] = set()
    if isinstance(enabled, dict):
        for vals in enabled.values():
            if not isinstance(vals, list):
                continue
            for mid in vals:
                if not isinstance(mid, str):
                    continue
                m = mid.strip()
                stripped: str | None = None
                for p in _FREE_PREFIXES:
                    if m.startswith(p):
                        stripped = m[len(p):]
                        break
                if stripped is None:
                    continue  # not a free-pool model — skip
                if stripped and stripped not in _AUTO_ROTATE_EXCLUDE:
                    slugs.add(stripped)
    return sorted(slugs)


def _pick_rotating_free_model() -> str:
    """Round-robin one concrete model for cgf/auto. Falls back to ChatGPT's own
    "auto" picker when no concrete model is enabled."""
    global _rr_index
    models = _enabled_free_models()
    if not models:
        return "auto"
    with _rr_lock:
        pick = models[_rr_index % len(models)]
        _rr_index += 1
    logger.info({"event": "free_auto_rotate_model", "picked": pick, "pool": len(models)})
    return pick


def handle_free_chat(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    stream: bool,
    body: dict[str, Any],
    route=None,
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """ChatGPT free-tier chat. Pulls only from the free account pool."""
    # Lazy imports — protocol module is fully initialised at call time.
    from services.protocol.openai_v1_chat_complete import (
        _messages_have_images,
        _handle_gemini_chat,
    )

    # Vision: chatgpt.com uploads image_url/input_image via /backend-api/files
    # (estuary) then references them as asset_pointer — needs an authenticated
    # free JWT. If the free pool is empty, fall back to gemini_free which
    # accepts inline base64 directly.
    if _messages_have_images(messages):
        has_free = bool(
            account_service.get_text_access_token(account_type="free")
        )
        if not has_free:
            logger.info({"event": "free_vision_fallback_to_gemini",
                         "reason": "no_active_free_account"})
            return _handle_gemini_chat("auto", messages, stream, body)

    messages = _normalize_tool_messages(messages)
    model = _normalize_free_model(model)
    # cgf/auto → round-robin a concrete enabled model (spread free quota across
    # models). Only the free entry point rotates; call_chatgpt_web (codex paid
    # fallback) keeps its own model untouched. Empty pool → ChatGPT native auto.
    if model == "auto":
        model = _pick_rotating_free_model()

    # Retry loop: when an account 429/quota-burns or expires, rotate to the
    # next free account. Non-quota errors re-raise immediately so real bugs
    # aren't masked.
    excluded_tokens: set[str] = set()
    last_quota_error: Exception | None = None
    for attempt in range(8):
        token = account_service.get_text_access_token(
            excluded_tokens=excluded_tokens, account_type="free"
        )
        if not token:
            break
        try:
            return _try_free_with_token(
                token, model, messages, tools, tool_choice, body, stream
            )
        except RuntimeError as exc:
            err_msg = str(exc).lower()
            is_quota = (
                "429" in err_msg
                or "usage_limit" in err_msg
                or ("quota" in err_msg and "exceeded" in err_msg)
                or "rate limit" in err_msg
                or "rate_limit" in err_msg
                or "too many requests" in err_msg
            )
            is_payload_too_large = (
                "413" in err_msg or "payload too large" in err_msg
            )
            is_expired = (
                "token_expired" in err_msg
                or "token expired" in err_msg
                or "expired" in err_msg
            ) and "401" in err_msg
            is_auth_error = (
                "could not parse" in err_msg
                or "authentication token" in err_msg
            ) and "401" in err_msg

            if is_expired:
                try:
                    account_service.update_account(token, {"status": "disabled"})
                except Exception:
                    pass
                logger.info({"event": "free_account_rotate", "reason": "token_expired", "attempt": attempt})
                excluded_tokens.add(token)
                continue
            if is_auth_error:
                # JWT to wrong endpoint / stale cookie — rotate but don't
                # permanently disable (session may be refreshable).
                logger.info({"event": "free_account_rotate", "reason": "auth_error", "attempt": attempt})
                excluded_tokens.add(token)
                continue
            if is_payload_too_large:
                logger.info({"event": "free_account_rotate", "reason": "payload_too_large", "attempt": attempt})
                excluded_tokens.add(token)
                last_quota_error = exc
                continue
            if not is_quota:
                raise
            logger.info({
                "event": "free_account_rotate", "reason": "quota_burnt",
                "attempt": attempt, "remaining_excluded": len(excluded_tokens) + 1,
            })
            excluded_tokens.add(token)
            last_quota_error = exc
            continue

    if last_quota_error is not None:
        raise last_quota_error
    raise RuntimeError("no usable chatgpt free account")


def _try_free_with_token(
    token: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    body: dict[str, Any],
    stream: bool = False,
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """Single free-account attempt against the chatgpt.com backend."""
    from services.openai_backend_api import OpenAIBackendAPI
    from services.protocol.conversation import ConversationRequest, collect_text, text_backend
    from services.config import _IS_ADDON
    from services.protocol.openai_v1_chat_complete import (
        stream_text_chat_completion,
        completion_response,
        _prefetch_stream,
        _prefetch_ha_context_if_needed,
        _stream_chatgpt_addon,
        _chatgpt_addon_completion,
    )

    # Build a backend bound to OUR rotation-selected token so text_backend()
    # doesn't re-pick a burnt account. A ChatGPT web accessToken (scraped from
    # /api/auth/session) carries aud=api.openai.com but IS the correct Bearer
    # for chatgpt.com/backend-api — so we always use the token here. The old
    # "api.openai.com → go anonymous" guard only made sense when the free pool
    # could contain raw OpenAI-API (standard/sk-) tokens; those are now group
    # "openai" and never reach this path (account_group split), so dropping the
    # token to anonymous just broke every free request.
    if token:
        backend = OpenAIBackendAPI(access_token=token)
    else:
        backend = text_backend()

    if _IS_ADDON:
        # Addon: XML tool-call parsing + force hint for HA.
        if stream:
            gen = _stream_chatgpt_addon(backend, messages, model, tools, tool_choice)
            return _prefetch_stream(gen, "chatgpt.com addon stream failed — token may be invalid")
        return _chatgpt_addon_completion(model, messages, tools, tool_choice)

    # Docker + chatgpt.com free: HA waits for ONE HTTP response, so no agentic
    # loop is possible. Pre-fetch HA context BEFORE the LLM call, inject as a
    # system message, then call once.
    messages = _prefetch_ha_context_if_needed(messages, tools, token)

    if stream:
        gen = stream_text_chat_completion(backend, messages, model, tools, tool_choice)
        return _prefetch_stream(gen, "chatgpt.com backend stream failed — token may be invalid")
    request = ConversationRequest(model=model, messages=messages, tools=tools, tool_choice=tool_choice)
    return completion_response(model, collect_text(backend, request), messages=messages)


def call_chatgpt_web(
    token: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    stream: bool,
    body: dict[str, Any],
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """Shared chatgpt.com web transport for a SPECIFIC token (no pool rotation).

    Used by the codex/paid provider when a paid account carries only a
    chatgpt.com web JWT (no real Codex token) — "phân nhóm theo plan, tự đổi
    route". The free module owns this transport; codex depends on it, never the
    reverse, so the free path stays independent.
    """
    messages = _normalize_tool_messages(messages)
    return _try_free_with_token(
        token, _normalize_free_model(model), messages, tools, tool_choice, body, stream
    )
