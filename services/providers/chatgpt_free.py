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
    if slug.startswith("cgf/"):
        slug = slug[4:]
    return slug or "auto"





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

    if model == "auto":
        try:
            from services.config import config as _config
            ms = _config.data.get("model_settings") or {}
            enabled = (ms.get("enabled_models") or {}).get("ChatGPT_free")
            if not enabled:
                enabled = (ms.get("enabled_models") or {}).get("chatgpt_free")
            if isinstance(enabled, list):
                for m in enabled:
                    m = str(m).strip()
                    if m.startswith("cgf/"):
                        m = m[4:]
                    if m and m not in ("auto", "research"):
                        model = m
                        break
        except Exception:
            pass

    # Retry loop: when an account 429/quota-burns or expires, rotate to the
    # next free account. Non-quota errors re-raise immediately so real bugs
    # aren't masked.
    requires_image = False
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("image_url", "input_image"):
                    requires_image = True
                    break

    excluded_tokens: set[str] = set()
    last_quota_error: Exception | None = None
    for attempt in range(8):
        token = account_service.get_text_access_token(
            excluded_tokens=excluded_tokens, account_type="free", requires_image=requires_image
        )
        if not token:
            break
        try:
            return _try_free_with_token(
                token, model, messages, tools, tool_choice, body, stream
            )
        except RuntimeError as exc:
            logger.info({"event": "free_account_raw_error", "attempt": attempt, "error": str(exc)[:300]})
            err_msg = str(exc).lower()
            is_quota = (
                "429" in err_msg
                or "usage_limit" in err_msg
                or ("quota" in err_msg and "exceeded" in err_msg)
                or "rate limit" in err_msg
                or "rate_limit" in err_msg
                or "too many requests" in err_msg
                or "hit your limit" in err_msg
                or "reached the limit" in err_msg
                or "reached your limit" in err_msg
                or "advanced data analysis" in err_msg
                or ("limit" in err_msg and requires_image)
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
                try:
                    account_service.demote_account(token)
                except Exception:
                    pass
                logger.info({"event": "free_account_rotate", "reason": "payload_too_large", "attempt": attempt})
                excluded_tokens.add(token)
                last_quota_error = exc
                continue
            if not is_quota:
                raise
            
            try:
                if requires_image and ("file" in err_msg or "image" in err_msg or "upload" in err_msg or "413" in err_msg or "limit" in err_msg):
                    account_service.mark_image_failed(token)
                else:
                    account_service.demote_account(token)
            except Exception:
                pass
                
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
        _extract_last_user_text,
        _is_status_only_query,
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

    # For pure status/listing queries, since we already injected the context
    # above, drop all tools. ChatGPT Free rejects payloads >45KB (413 Payload
    # Too Large), and 40 HA tool schemas will crash it.
    user_text = _extract_last_user_text(messages)
    if _is_status_only_query(user_text):
        tools = None

    from services.protocol.openai_v1_chat_complete import _messages_have_images
    force_sync_for_vision = _messages_have_images(messages)

    if stream and not force_sync_for_vision:
        gen = stream_text_chat_completion(backend, messages, model, tools, tool_choice)
        return _prefetch_stream(gen, "chatgpt.com backend stream failed — token may be invalid")
    request = ConversationRequest(model=model, messages=messages, tools=tools, tool_choice=tool_choice)
    content = collect_text(backend, request)
    
    if "advanced data analysis right now" in content or "can’t do more advanced data analysis" in content:
        raise RuntimeError(f"quota exceeded (advanced data analysis): {content}")
        
    if stream and force_sync_for_vision:
        from services.protocol.openai_v1_chat_complete import _extract_xml_tool_calls_from_text, completion_chunk
        import time, uuid
        def _simulated_stream():
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            clean_content = content
            tool_calls = None
            if tools:
                tool_calls = _extract_xml_tool_calls_from_text(content)
                if tool_calls:
                    import re
                    clean_content = re.sub(r"```xml\s*<tool_call[^`]*```", "", content, flags=re.DOTALL).strip()
            
            yield completion_chunk(model, {"role": "assistant", "content": clean_content}, None, completion_id, created)
            if tool_calls:
                tc_delta = [{"index": i, "id": tc.get("id", f"tc_{i}"), "type": "function", "function": tc.get("function", {})} for i, tc in enumerate(tool_calls)]
                yield completion_chunk(model, {"tool_calls": tc_delta}, None, completion_id, created)
                yield completion_chunk(model, {}, "tool_calls", completion_id, created)
            else:
                yield completion_chunk(model, {}, "stop", completion_id, created)
        return _simulated_stream()

    
    if tools:
        from services.protocol.openai_v1_chat_complete import _extract_xml_tool_calls_from_text
        tool_calls = _extract_xml_tool_calls_from_text(content)
        if tool_calls:
            import time, uuid
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls",
                }],
            }
            
    return completion_response(model, content, messages=messages)


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
