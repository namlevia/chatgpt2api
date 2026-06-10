from __future__ import annotations

import itertools
import json
import re
import time
import uuid
from typing import Any, Iterable, Iterator

from fastapi import HTTPException

from services.protocol.conversation import (
    ConversationRequest,
    ImageOutput,
    TOOL_CALL_RE,
    TOOL_CALL_SELF_CLOSING_RE,
    collect_image_outputs,
    collect_text,
    count_message_tokens,
    count_text_tokens,
    encode_images,
    normalize_messages,
    stream_image_outputs_with_pool,
    stream_text_deltas,
    text_backend,
)
from services.account_service import account_service
from services.backend_router import backend_router
from services.config import config
from services.model_cooldown import model_cooldown
from services.search_service import search_service
from utils.helper import build_chat_image_markdown_content, extract_chat_image, extract_chat_prompt, is_image_chat_request, parse_image_count
from utils.log import logger


def _extract_status(error_text: str) -> int:
    """Extract HTTP status code from error message text."""
    import re
    text = str(error_text)
    match = re.search(r'\b(4\d\d|5\d\d|error\s+(\d+))', text, re.IGNORECASE)
    if match:
        code = match.group(2) or match.group(1)
        try:
            return int(code)
        except ValueError:
            pass
    # Check for keyword patterns
    lower = text.lower()
    if "401" in lower or "unauthorized" in lower: return 401
    if "402" in lower: return 402
    if "403" in lower or "forbidden" in lower: return 403
    if "404" in lower: return 404
    if "429" in lower or "rate" in lower or "quota" in lower: return 429
    if "503" in lower or "502" in lower or "500" in lower: return 500
    return 0


def completion_chunk(model: str, delta: dict[str, Any], finish_reason: str | None = None, completion_id: str = "", created: int | None = None) -> dict[str, Any]:
    return {
        "id": completion_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def completion_response(
    model: str,
    content: str,
    created: int | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt_tokens = count_message_tokens(messages, model) if messages else 0
    completion_tokens = count_text_tokens(content, model) if messages else 0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _prefetch_stream(gen: Iterator[dict[str, Any]], error_msg: str) -> Iterator[dict[str, Any]]:
    """Pre-fetch first element from a stream generator so auth/connection
    errors are raised synchronously inside the caller's try/except block.

    Without this, lazy generators from stream_text_chat_completion /
    _stream_chatgpt_addon would raise errors only when iterated by
    _wrap_mcp_stream, which silently catches exceptions and returns an
    empty SSE stream — the client sees a 200 OK with no data.
    """
    try:
        first = next(gen)
    except StopIteration:
        raise RuntimeError(error_msg)
    return itertools.chain([first], gen)


def stream_text_chat_completion(backend, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]] | None = None, tool_choice: Any = None) -> Iterator[dict[str, Any]]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    sent_role = False
    request = ConversationRequest(model=model, messages=messages, tools=tools, tool_choice=tool_choice)
    for delta_text in stream_text_deltas(backend, request):
        if not sent_role:
            sent_role = True
            yield completion_chunk(model, {"role": "assistant", "content": delta_text}, None, completion_id, created)
        else:
            yield completion_chunk(model, {"content": delta_text}, None, completion_id, created)
    if not sent_role:
        yield completion_chunk(model, {"role": "assistant", "content": ""}, None, completion_id, created)
    yield completion_chunk(model, {}, "stop", completion_id, created)


def collect_chat_content(chunks: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        choices = chunk.get("choices")
        first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
        content = str(delta.get("content") or "")
        if content:
            parts.append(content)
    return "".join(parts)


def chat_messages_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        return [message for message in messages if isinstance(message, dict)]
    prompt = str(body.get("prompt") or "").strip()
    if prompt:
        return [{"role": "user", "content": prompt}]
    raise HTTPException(status_code=400, detail={"error": "messages or prompt is required"})


def chat_image_args(body: dict[str, Any]) -> tuple[str, str, int, list[tuple[bytes, str, str]]]:
    model = str(body.get("model") or "gpt-image-2").strip() or "gpt-image-2"
    prompt = extract_chat_prompt(body)
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "prompt is required"})
    images = [
        (data, f"image_{idx}.png", mime)
        for idx, (data, mime) in enumerate(extract_chat_image(body), start=1)
    ]
    return model, prompt, parse_image_count(body.get("n")), images


def text_chat_parts(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]] | None, Any]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = chat_messages_from_body(body)
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        tools = [t for t in tools if isinstance(t, dict)]
    else:
        tools = None
    tool_choice = body.get("tool_choice")
    return model, messages, tools, tool_choice


def image_result_content(result: dict[str, Any]) -> str:
    data = result.get("data")
    if isinstance(data, list) and data:
        return build_chat_image_markdown_content(result)
    return str(result.get("message") or "Image generation completed.")


def image_chat_response(body: dict[str, Any]) -> dict[str, Any]:
    model, prompt, n, images = chat_image_args(body)
    result = collect_image_outputs(stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        n=n,
        response_format="b64_json",
        images=encode_images(images) or None,
    )))
    return completion_response(model, image_result_content(result), int(result.get("created") or 0) or None)


def image_chat_events(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    model, prompt, n, images = chat_image_args(body)
    image_outputs = stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        n=n,
        response_format="b64_json",
        images=encode_images(images) or None,
    ))
    yield from stream_image_chat_completion(image_outputs, model)


def stream_image_chat_completion(image_outputs: Iterable[ImageOutput], model: str) -> Iterator[dict[str, Any]]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    sent_role = False
    sent_text = ""
    for output in image_outputs:
        content = ""
        if output.kind == "progress":
            content = output.text
            sent_text += content
        elif output.kind == "result":
            content = build_chat_image_markdown_content({"data": output.data})
        elif output.kind == "message":
            content = output.text[len(sent_text):] if output.text.startswith(sent_text) else output.text
        if not content:
            continue
        if not sent_role:
            sent_role = True
            yield completion_chunk(model, {"role": "assistant", "content": content}, None, completion_id, created)
        else:
            yield completion_chunk(model, {"content": content}, None, completion_id, created)
    if not sent_role:
        yield completion_chunk(model, {"role": "assistant", "content": ""}, None, completion_id, created)
    yield completion_chunk(model, {}, "stop", completion_id, created)


# ---------------------------------------------------------------------------
# Pipeline combo — kiểu Aider architect/editor: "bố" (model mạnh) lập kế hoạch
# ngắn, "con" (model rẻ/nhanh) viết code dài theo kế hoạch → tiết kiệm 30-50%
# token đầu ra của model đắt. Tạo bằng UI combo sẵn có, không cần UI mới:
# combo nào có entry "architect:<model>" sẽ chạy pipeline; entry
# "editor:<model>" (hoặc entry trần) là chuỗi fallback cho tầng thực thi.
# Ví dụ combo "code": ["architect:claude/auto", "editor:cgf/auto", "nv/..."]

_PIPELINE_ARCHITECT_PROMPT = (
    "Bạn là kiến trúc sư trưởng (architect). Phân tích yêu cầu và lập KẾ HOẠCH "
    "triển khai NGẮN GỌN cho một lập trình viên thực thi: liệt kê các bước, "
    "file/hàm cần sửa, thuật toán, edge case cần xử lý. KHÔNG viết code đầy đủ "
    "— chỉ mô tả và pseudo-code khi thật cần. Trả lời bằng ngôn ngữ của người "
    "dùng, tối đa 400 từ."
)

_PIPELINE_EDITOR_PROMPT = (
    "Bạn là lập trình viên thực thi (editor). Kiến trúc sư trưởng đã duyệt kế "
    "hoạch dưới đây cho yêu cầu của người dùng. Hãy triển khai CHÍNH XÁC theo "
    "kế hoạch, xuất code hoàn chỉnh chạy được, không hỏi lại, không bàn thêm "
    "phương án khác.\n\n=== KẾ HOẠCH ĐÃ DUYỆT ===\n{plan}\n=== HẾT KẾ HOẠCH ==="
)

_PIPELINE_PLAN_MAX_CHARS = 8000


def _parse_pipeline_combo(combo_entries: list[str]) -> tuple[list[str], list[str]] | None:
    """Tách combo thành (architects, editors); None nếu là combo fallback thường."""
    architects: list[str] = []
    editors: list[str] = []
    for entry in combo_entries:
        s = str(entry or "").strip()
        low = s.lower()
        if low.startswith("architect:"):
            m = s.split(":", 1)[1].strip()
            if m:
                architects.append(m)
        elif low.startswith("editor:"):
            m = s.split(":", 1)[1].strip()
            if m:
                editors.append(m)
        elif s:
            editors.append(s)
    if architects and editors:
        return (architects, editors)
    return None


def _pipeline_extract_content(result: Any) -> str:
    if isinstance(result, dict):
        try:
            choices = result.get("choices") or []
            msg = choices[0].get("message") or {}
            return str(msg.get("content") or "")
        except Exception:
            return ""
    try:
        return collect_chat_content(result)
    except Exception:
        return ""


def _run_pipeline_combo(
    combo_name: str,
    architects: list[str],
    editors: list[str],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    body: dict[str, Any],
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    # ---- Tầng 1: architect (bố) — non-stream, không tool, chỉ lập plan ----
    plan = ""
    plan_model = ""
    arch_body = dict(body)
    arch_body["stream"] = False
    arch_messages = [{"role": "system", "content": _PIPELINE_ARCHITECT_PROMPT}] + list(messages)
    for am in architects:
        try:
            route = backend_router.route(am)
            cooldown = model_cooldown.get_cooldown_info(route.model)
            if cooldown:
                logger.warning({"event": "pipeline_architect_cooldown", "combo": combo_name, "model": am, **cooldown})
                continue
            logger.info({"event": "pipeline_architect_try", "combo": combo_name, "provider": route.provider, "model": route.model})
            result = _dispatch(route, arch_messages, None, None, arch_body)
            content = _pipeline_extract_content(result).strip()
            if content:
                plan = content[:_PIPELINE_PLAN_MAX_CHARS]
                plan_model = am
                model_cooldown.record_success("pipeline:" + combo_name, route.model)
                logger.info({"event": "pipeline_architect_ok", "combo": combo_name, "model": am, "plan_chars": len(plan)})
                break
        except Exception as exc:
            logger.warning({"event": "pipeline_architect_fail", "combo": combo_name, "model": am, "error": str(exc)[:200]})
            continue

    # ---- Tầng 2: editor (con) — stream theo client, fallback chain ----
    editor_messages = list(messages)
    if plan:
        editor_messages.append({"role": "system", "content": _PIPELINE_EDITOR_PROMPT.format(plan=plan)})
    else:
        # Tất cả architect chết → degrade về gọi thẳng editor, không hard-fail
        logger.warning({"event": "pipeline_no_plan", "combo": combo_name, "architects": architects})

    last_error = ""
    for em in editors:
        try:
            route = backend_router.route(em)
            cooldown = model_cooldown.get_cooldown_info(route.model)
            if cooldown:
                last_error = cooldown["message"]
                logger.warning({"event": "pipeline_editor_cooldown", "combo": combo_name, "model": em, **cooldown})
                continue
            logger.info({"event": "pipeline_editor_try", "combo": combo_name, "provider": route.provider, "model": route.model, "has_plan": bool(plan), "architect": plan_model})
            result = _dispatch(route, editor_messages, tools, tool_choice, body)
            model_cooldown.record_success("pipeline:" + combo_name, route.model)
            return result
        except Exception as exc:
            last_error = str(exc)
            logger.warning({"event": "pipeline_editor_fail", "combo": combo_name, "model": em, "error": last_error[:200]})
            model_cooldown.record_failure(
                account_id="pipeline:" + combo_name, model=em,
                status_code=_extract_status(last_error), error_body=last_error, provider="",
            )
            continue
    return completion_response(model=combo_name, content=f"All pipeline editors failed. Last error: {last_error[:200]}", messages=messages)


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """Wrapper: lớp ký ức OpenMemory (mặc định TẮT) bọc ngoài flow chat chính.

    prepare() recall + inject ký ức liên quan vào messages; capture() lưu lượt
    chat sau khi có response (tee stream, thread nền). Mọi lỗi memory đều nuốt
    — flow chatgpt/gemini/claude/HA không đổi khi memory tắt hoặc chết.
    """
    mem_ctx = None
    try:
        from services.memory_service import memory_service
        mem_ctx = memory_service.prepare(body)
    except Exception:
        mem_ctx = None
    result = _handle_main(body)
    if mem_ctx is not None:
        try:
            result = mem_ctx.capture(result)
        except Exception:
            pass
    return result


def _handle_main(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    try:
        import json
        with open("/tmp/last_req.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(body, ensure_ascii=False))
    except Exception:
        pass

    # Image chat requests always use existing DALL-E flow
    if is_image_chat_request(body):
        if body.get("stream"):
            return image_chat_events(body)
        return image_chat_response(body)

    model, messages, tools, tool_choice = text_chat_parts(body)

    # Detect vision request once — both combo and single-model paths use this
    # to skip MCP/HA tool injection (saves ~2.5s discovery + removes 60+ tools
    # that vision models have to scan before answering).
    is_vision_request = _messages_have_images(messages)

    # Detect HA intent on the PRISTINE user message before any search/HA
    # injection runs. Search results often contain phrases like "mở cửa"
    # (trading session jargon) or "đèn" (news headline) that would trip
    # the HA regex if we checked post-injection.
    try:
        from services.ha_client import is_ha_query as _is_ha
        ha_query_pristine = _is_ha(messages)
    except Exception:
        ha_query_pristine = False
        
    original_user_text = _extract_last_user_text(messages)

    # Check if this is a combo model — try each model until success
    if backend_router.is_combo(model):
        # Pipeline combo (architect/editor): combo có entry "architect:<model>"
        # chạy 2 tầng bố-con thay vì fallback chain thường.
        _combo_entries = backend_router._get_combo_models(model) or []
        _pipeline = _parse_pipeline_combo(_combo_entries)
        if _pipeline:
            return _run_pipeline_combo(model, _pipeline[0], _pipeline[1], messages, tools, tool_choice, body)
        routes = backend_router.route_combo(model)
        last_error = ""

        # Skip web search for pure HA queries — they're answered from the
        # registry, no need to spend 8s on grounding.
        search_injected = False
        if search_service.is_enabled and not ha_query_pristine and not is_vision_request and not bool(body.get("_is_ha_request")):
            before_size = _messages_size(messages)
            messages_copy = search_service.process_messages(messages)
            search_injected = _messages_size(messages_copy) > before_size
            # Auto-curate search results to RAG after response (best-effort bg)
            _curate_search_results(messages_copy)
        else:
            messages_copy = messages
            
        # Mirror the non-combo path: inject HA registry as a system message for
        # HA-related queries so the LLM can answer in ONE round-trip instead of
        # doing GetLiveContext / ha_get_state -> wait -> final answer (saves ~7s).
        for route in routes:
            messages_for_route = list(messages_copy)
            ha_context_injected = False
            if ha_query_pristine and route.provider != "chatgpt_free":
                try:
                    from services.ha_client import inject_ha_context
                    before_len = len(messages_for_route)
                    messages_for_route = inject_ha_context(messages_for_route)
                    ha_context_injected = len(messages_for_route) > before_len
                except Exception:
                    pass

            tools_with_mcp = _inject_mcp_tools(
                tools, skip_ha_search=ha_context_injected,
                is_vision=is_vision_request, search_injected=search_injected,
                user_text=original_user_text,
                is_free_model=(route.provider == "chatgpt_free"),
            )
            try:
                cooldown = model_cooldown.get_cooldown_info(route.model)
                if cooldown:
                    logger.warning({"event": "model_cooldown_skip", "model": route.model, **cooldown})
                    last_error = cooldown["message"]
                    continue

                logger.info({"event": "combo_try", "combo": model, "provider": route.provider, "model": route.model})
                
                result = _dispatch(route, messages_for_route, tools_with_mcp, tool_choice, body)
                # Execute MCP tools server-side for combo too
                if not isinstance(result, dict):
                    result = _wrap_mcp_stream(result, messages_for_route, route, body)
                elif isinstance(result, dict):
                    result = _execute_mcp_tools_in_response(messages_for_route, result, route, body)
                result = _maybe_strip_markdown(result, messages_for_route, force=ha_context_injected or bool(body.get("_is_ha_request")))
                model_cooldown.record_success("combo:" + model, route.model)
                return result
            except Exception as exc:
                last_error = str(exc)
                logger.warning({"event": "combo_fail", "combo": model, "provider": route.provider, "error": last_error[:200]})
                model_cooldown.record_failure(
                    account_id="combo:" + model, model=route.model,
                    status_code=_extract_status(last_error), error_body=last_error, provider=route.provider,
                )
                continue
        return completion_response(model=model, content=f"All providers failed. Last error: {last_error[:200]}", messages=messages)

    # Single model — route directly
    route = backend_router.route(model, messages)

    # Apply search injection for all backends — but skip when the user query
    # is a pure HA command/status (answered from registry) or a vision task.
    search_injected = False
    if search_service.is_enabled and not ha_query_pristine and not is_vision_request and not bool(body.get("_is_ha_request")):
        before_size = _messages_size(messages)
        messages = search_service.process_messages(messages)
        search_injected = _messages_size(messages) > before_size

    # Inject HA smart home context only when the PRISTINE user message looked
    # like an HA query. This avoids false positives from search-result text.
    ha_context_injected = False
    if ha_query_pristine and route.provider != "chatgpt_free":
        try:
            from services.ha_client import inject_ha_context
            before_len = len(messages)
            messages = inject_ha_context(messages)
            ha_context_injected = len(messages) > before_len
        except Exception:
            pass

    # Inject MCP tools from enabled presets
    tools = _inject_mcp_tools(
        tools, skip_ha_search=ha_context_injected,
        is_vision=is_vision_request, search_injected=search_injected,
        user_text=original_user_text,
        is_free_model=(route.provider == "chatgpt_free"),
    )

    result = _dispatch(route, messages, tools, tool_choice, body)

    # Execute MCP tools server-side — HA doesn't know these tools
    if not isinstance(result, dict):
        # Streaming (Iterator) — wrap to intercept tool calls
        if route.provider != "chatgpt_free":
            result = _wrap_mcp_stream(result, messages, route, body)
    elif isinstance(result, dict):
        if route.provider != "chatgpt_free":
            result = _execute_mcp_tools_in_response(messages, result, route, body)
        import json
        logger.info({"event": "debug_final_result", "result": json.dumps(result, ensure_ascii=False)[:2000]})

    result = _maybe_strip_markdown(result, messages, force=ha_context_injected or bool(body.get("_is_ha_request")))
    try:
        import json
        with open("/tmp/last_response.json", "w", encoding="utf-8") as f:
            if isinstance(result, dict):
                f.write(json.dumps(result, ensure_ascii=False))
            else:
                f.write("STREAM_GENERATOR")
    except Exception:
        pass
    return result


def _maybe_strip_markdown(result, messages, force=False):
    """Always strip backend artifacts (citation markers etc.). Conditionally
    strip markdown when:
    - the request looks like a plain-text surface (device-keyword heuristic), OR
    - the caller forces it (e.g. HA voice / Conversation API which cannot
      render markdown tables — `giá xăng hôm nay` should arrive as plain
      text on HA even though it has no device keyword).
    Stream and dict results are both supported.
    """
    # Always strip backend artifacts — these never belong in user-facing text
    if isinstance(result, dict):
        result = _strip_artifacts_in_response(result)
    else:
        result = _strip_artifacts_in_stream(result)
    # Conditionally strip markdown on top
    if not (force or _request_wants_plain_text(messages)):
        return result
    if isinstance(result, dict):
        return _strip_markdown_in_response(result)
    return _strip_markdown_in_stream(result)


def _strip_artifacts_in_response(result: dict[str, Any]) -> dict[str, Any]:
    choices = result.get("choices") or []
    for ch in choices:
        msg = ch.get("message") if isinstance(ch, dict) else None
        if isinstance(msg, dict):
            txt = msg.get("content")
            if isinstance(txt, str):
                msg["content"] = _strip_artifacts_inline(txt)
    return result


def _strip_artifacts_in_stream(it: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for chunk in it:
        try:
            choices = chunk.get("choices") or []
            for ch in choices:
                if not isinstance(ch, dict):
                    continue
                delta = ch.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        delta["content"] = _strip_artifacts_inline(content)
        except Exception:
            pass
        yield chunk


def _curate_search_results(messages: list[dict[str, Any]]) -> None:
    """Extract last user query + search results → curate to RAG in background."""
    try:
        query = ""
        search_text = ""
        for m in reversed(messages):
            if m.get("role") == "user" and not query:
                c = m.get("content", "")
                query = c if isinstance(c, str) else str(c)[:200]
            if m.get("role") == "system" and "Search results" in str(m.get("content", "")):
                search_text = str(m.get("content", ""))[:2000]
        if query and search_text:
            # Curate into the topic KB the IntentRouter picked (kb_tu_nhien,
            # kb_y_te, …) instead of a catch-all kb_general, so the enrichment
            # loop deposits knowledge where ask_<col> will later find it.
            collection = ""
            try:
                from services.search_service import _intent_router
                cols = _intent_router.detect(query).get("kb_collections") or []
                collection = cols[0] if cols else ""
            except Exception:
                collection = ""
            search_service.curate_response(query, search_text, collection)
    except Exception:
        pass


def _auto_search_enrich(query: str) -> str:
    """Run search alongside MCP tool execution for richer context."""
    if not search_service.is_enabled:
        return ""
    try:
        results = search_service.search_all(query)
        if not results:
            return ""
        lines = ["\n---\n## Kết quả tìm kiếm bổ sung\n"]
        for r in results[:5]:
            title = r.get("title", "")
            snippet = (r.get("snippet") or "")[:300]
            url = r.get("url", "")
            if title:
                lines.append(f"- **{title}**")
            if snippet:
                lines.append(f"  {snippet}")
            if url:
                lines.append(f"  {url}")
        return "\n".join(lines)
    except Exception:
        return ""


def _extract_user_query(messages: list[dict[str, Any]]) -> str:
    """Get the last user message text for search enrichment."""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c[:200]
            if isinstance(c, list):
                return str(c[0].get("text", ""))[:200] if c else ""
    return ""


def _wrap_mcp_stream(
    stream_iter, messages: list[dict[str, Any]], route, body: dict[str, Any]
):
    """Wrap a streaming response to execute MCP/HA tools and return final answer.

    Collects the full stream, checks for server-side tool calls, executes them in
    an agentic loop (multi-step: e.g. ha_search_entities → ha_get_state → answer),
    then streams the final LLM response.
    """
    # Collect full response from stream
    chunks = []
    full_content = ""
    final_tool_calls: list | None = None
    model = ""

    try:
        for chunk in stream_iter:
            chunks.append(chunk)
            if isinstance(chunk, dict):
                model = chunk.get("model", model)
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                full_content += delta.get("content") or ""
                tc = delta.get("tool_calls")
                if tc:
                    final_tool_calls = tc
    except Exception as exc:
        logger.error({"event": "mcp_stream_error", "error": str(exc)[:300],
                       "chunks_collected": len(chunks)})
        for c in chunks:
            yield c
        return

    # No native tool_calls — check for XML tool calls in content text.
    # ChatGPT web backend models output ```xml <tool_call> instead of
    # native function-call objects. Parse them so server-side tools
    # (GetLiveContext, ha_*) are executed here.
    was_xml = False
    if not final_tool_calls:
        xml_calls = _extract_xml_tool_calls_from_text(full_content)
        if xml_calls:
            was_xml = True
            final_tool_calls = []
            for i, xc in enumerate(xml_calls):
                fn = xc.get("function", {})
                final_tool_calls.append({
                    "id": f"xml_stream_{i}",
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    },
                })

    # No tool calls → stream as-is
    if not final_tool_calls:
        for c in chunks:
            yield c
        return

    # Filter to server-side tools only
    from services.mcp_client import get_enabled_mcp_tools
    from services.ha_client import get_ha_tools
    known_server_tools = {
        t.get("function", {}).get("name", "")
        for t in get_enabled_mcp_tools() + get_ha_tools()
    }

    mcp_calls = [tc for tc in final_tool_calls
                 if tc.get("function", {}).get("name", "") in known_server_tools]

    if not mcp_calls:
        if was_xml:
            import re
            clean_content = re.sub(
                r"```xml\s*<tool_call[^`]*```", "", full_content,
                flags=re.DOTALL,
            ).strip()
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            if clean_content:
                yield completion_chunk(model, {"role": "assistant", "content": clean_content}, None, completion_id, created)
            tool_calls_delta = [{"index": i, "id": tc["id"], "type": "function", "function": tc["function"]} for i, tc in enumerate(final_tool_calls)]
            yield completion_chunk(model, {"role": "assistant", "tool_calls": tool_calls_delta}, None, completion_id, created)
            yield completion_chunk(model, {}, "tool_calls", completion_id, created)
            return

        for c in chunks:
            yield c
        return

    # Build a synthetic non-stream result to feed into the agentic loop
    # Strip XML tool-call fence from content — tool_calls carry the intent.
    import re
    clean_content = re.sub(
        r"```xml\s*<tool_call[^`]*```", "", full_content,
        flags=re.DOTALL,
    ).strip()
    synthetic_result = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": clean_content,
                "tool_calls": final_tool_calls,
            },
            "finish_reason": "tool_calls",
        }],
    }

    # Run agentic loop (handles multi-step chains)
    try:
        final_result = _execute_mcp_tools_in_response(messages, synthetic_result, route, body)
    except Exception as exc:
        logger.warning({"event": "mcp_stream_loop_failed", "error": str(exc)})
        for c in chunks:
            yield c
        return

    # Stream the final result back to client
    if hasattr(final_result, "__iter__") and not isinstance(final_result, (dict, str)):
        yield from final_result
    elif isinstance(final_result, dict):
        # Convert non-streaming result into stream chunks
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        choice = (final_result.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        tool_calls = (choice.get("message") or {}).get("tool_calls")
        yield completion_chunk(model, {"role": "assistant", "content": content}, None, completion_id, created)
        if tool_calls:
            tool_calls_delta = [{"index": i, "id": tc.get("id"), "type": "function", "function": tc.get("function")} for i, tc in enumerate(tool_calls)]
            yield completion_chunk(model, {"tool_calls": tool_calls_delta}, None, completion_id, created)
            yield completion_chunk(model, {}, "tool_calls", completion_id, created)
        else:
            yield completion_chunk(model, {}, "stop", completion_id, created)
    else:
        for c in chunks:
            yield c


def _execute_mcp_tools_in_response(
    messages: list[dict[str, Any]], result: dict, route, body: dict[str, Any],
    max_iterations: int = 4,
) -> dict[str, Any]:
    """Execute MCP/HA tool calls in an agentic loop until final answer or max_iterations.

    Supports multi-step tool chains like:
      ha_search_entities → ha_get_state → final LLM answer
    """
    from services.mcp_client import get_enabled_mcp_tools
    from services.ha_client import get_ha_tools

    current_result = result
    current_messages = list(messages)

    for iteration in range(max_iterations):
        choice = (current_result.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = list(msg.get("tool_calls") or [])
        xml_calls = None

        # ChatGPT web backend returns XML tool calls in text content,
        # not native function-call objects. Parse them so server-side
        # tools (GetLiveContext, ha_*) are executed here instead of
        # being passed through as text to HA pipeline.
        if not tool_calls:
            content_text = msg.get("content") or ""
            xml_calls = _extract_xml_tool_calls_from_text(content_text)
            if xml_calls:
                for i, xc in enumerate(xml_calls):
                    fn = xc.get("function", {})
                    tool_calls.append({
                        "id": f"xml_{iteration}_{i}",
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        },
                    })

        if not tool_calls:
            return current_result  # No more tool calls → final answer

        # Identify server-side vs native (HA pipeline) tools
        known_server_tools = {
            t.get("function", {}).get("name", "")
            for t in get_enabled_mcp_tools() + get_ha_tools()
        }

        mcp_calls = []
        native_calls = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            if name in known_server_tools:
                mcp_calls.append(tc)
            else:
                native_calls.append(tc)

        if not mcp_calls:
            return current_result  # Only native HA tools → pass through

        # Strip XML tool-call fence from content so the model doesn't
        # see duplicate calls when we re-query with native tool_calls.
        assistant_content = msg.get("content") or ""
        if xml_calls:
            import re
            assistant_content = re.sub(
                r"```xml\s*<tool_call[^`]*```", "", assistant_content,
                flags=re.DOTALL,
            ).strip()

        # Append assistant message with all server-side tool calls
        current_messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": mcp_calls,
        })

        # Execute ALL server-side tool calls IN PARALLEL for speed
        is_action_only = len(mcp_calls) > 0 and all(tc.get("function", {}).get("name") == "ha_call_service" for tc in mcp_calls) and not native_calls

        if len(mcp_calls) > 1:
            # Parallel execution for multiple tool calls
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(mcp_calls)) as pool:
                future_map = {}
                for tc in mcp_calls:
                    args_str = tc.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except Exception:
                        args = {}
                    tool_name = tc.get("function", {}).get("name", "")
                    logger.info({"event": "mcp_tool_exec_parallel", "tool": tool_name})
                    future_map[pool.submit(_execute_mcp_tool, tool_name, args)] = tc

                # Collect results and append to messages
                for future in concurrent.futures.as_completed(future_map, timeout=30):
                    tc = future_map[future]
                    tool_name = tc.get("function", {}).get("name", "")
                    tool_id = tc.get("id", f"mcp_{iteration}")
                    try:
                        mcp_result = future.result()
                    except Exception as exc:
                        mcp_result = f"Tool error: {exc}"
                    if mcp_result is None:
                        mcp_result = f"Tool '{tool_name}' returned no result."
                    current_messages.append({
                        "role": "tool", "tool_call_id": tool_id,
                        "name": tool_name, "content": mcp_result,
                    })
        else:
            # Single tool call — sequential is fine
            for tc in mcp_calls:
                args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    args = {}
                tool_name = tc.get("function", {}).get("name", "")
                tool_id = tc.get("id", f"mcp_{iteration}")
                logger.info({"event": "mcp_tool_exec", "tool": tool_name, "iteration": iteration})
                mcp_result = _execute_mcp_tool(tool_name, args)
                if mcp_result is None:
                    mcp_result = f"Tool '{tool_name}' returned no result."
                current_messages.append({
                    "role": "tool", "tool_call_id": tool_id,
                    "name": tool_name, "content": mcp_result,
                })

        if is_action_only:
            logger.info({"event": "ha_fast_short_circuit"})
            final_text = msg.get("content") or "Đã thực hiện xong lệnh điều khiển thiết bị."
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": current_result.get("model", ""),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_text,
                    },
                    "finish_reason": "stop",
                }],
            }

        # Re-dispatch with updated messages
        tools = _inject_mcp_tools(body.get("tools"), is_free_model=(route.provider == "chatgpt_free"))
        try:
            current_result = _dispatch(route, current_messages, tools, body.get("tool_choice"), body)
            if not isinstance(current_result, dict):
                # Got a stream back — yield it directly
                return current_result
        except Exception as exc:
            logger.warning({"event": "mcp_followup_failed", "error": str(exc), "iteration": iteration})
            return current_result

    return current_result


def _dispatch(route, messages, tools, tool_choice, body):
    """Dispatch to the correct provider handler."""
    # RTK compression thresholds — 24KB → 80KB → 100KB. We sit at the
    # chatgpt.com hard-limit cap; over-cap payloads still get compressed
    # by _rtk_compress_messages so requests never 413.
    # File upload bypass runs independently of RTK — when enabled for chatgpt
    # provider, large user messages (>80KB) are uploaded to /backend-api/files
    # and referenced via asset_pointer instead of being head+tail compressed.
    file_upload_enabled = (route.provider in ("chatgpt", "chatgpt_free"))
    if route.provider in ("chatgpt", "chatgpt_free"):
        rtk_on = config.rtk_enabled
        rtk_threshold = 100_000
    else:
        rtk_on = config.rtk_other_enabled
        rtk_threshold = 100_000
    if rtk_on or file_upload_enabled:
        from services.protocol.conversation import _rtk_compress_messages
        file_upload_threshold = 80_000 if file_upload_enabled else 0
        messages = _rtk_compress_messages(messages, rtk_threshold, file_upload_threshold=file_upload_threshold)

    if route.provider == "opencode":
        return _handle_opencode_chat(route.model, messages, body.get("stream"), body)
    elif route.provider == "ninerouter":
        return _handle_ninerouter_chat(route.model, messages, tools, tool_choice, body.get("stream"), body)
    elif route.provider in ("openai_oauth", "codex"):
        return _handle_openai_oauth_chat(route.model, messages, tools, tool_choice, body.get("stream"), body)
    elif route.provider == "gemini_free":
        return _handle_gemini_chat(route.model, messages, body.get("stream"), body)
    elif route.provider == "antigravity":
        return _handle_antigravity_chat(route.model, messages, tools, tool_choice, body.get("stream"), body)
    elif route.provider == "nvidia_nim":
        return _handle_nvidia_chat(route.model, messages, tools, tool_choice, body.get("stream"), body)
    elif route.provider == "gemini_web":
        from services.providers.web_proxy import handle_gemini_web_chat
        return handle_gemini_web_chat(route.model, messages, body.get("stream"), body)
    elif route.provider.startswith("custom:"):
        return _handle_custom_openai_chat(route.provider, route.model, messages, tools, tool_choice, body.get("stream"), body)
    elif route.provider in ("chatgpt_free", "chatgpt"):
        # Standalone free-tier module. `chatgpt/` and bare/unprefixed models are
        # now aliases for free — handle_free_chat owns the whole free path
        # (vision-fallback to gemini, tool→user normalization, free-pool
        # rotation). Codex/paid traffic uses cx/ | codex/ | paid/; OpenAI-API
        # (sk-/standard) uses oai/.
        from services.providers.chatgpt_free import handle_free_chat
        return handle_free_chat(route.model, messages, tools, tool_choice, body.get("stream"), body, route)
    elif route.provider == "openai_api":
        # 3rd path kept separate (đại ca's decision): raw OpenAI API key (sk-)
        # or `standard` JWT accounts → api.openai.com via custom:openai.
        return _handle_openai_api_chat(route.model, messages, tools, tool_choice, body.get("stream"), body)
    elif route.provider == "claude":
        # claude.ai free web (claude/ | clf/ | cc/) — same backend as the
        # dedicated /v1/claude/* endpoint, reachable from the main route so
        # HA/automations can pick claude models from /v1/models directly.
        from api.claude import handle_claude_chat
        return handle_claude_chat(route.model, messages, body.get("stream"), body)
    elif route.provider == "gemini_web_api":
        # gemini.google.com qua cookie 1PSID (gma/ | gemini-web/) — HTTP API
        # trực tiếp (gemini_webapi), nhanh hơn DOM scrape gmw/.
        from api.gemini_web import handle_gemini_web_api_chat
        return handle_gemini_web_api_chat(route.model, messages, body.get("stream"), body)
    else:
        logger.warning({"event": "unknown_provider", "provider": route.provider, "fallback": "chatgpt_free"})
        from services.providers.chatgpt_free import handle_free_chat
        return handle_free_chat(route.model, messages, tools, tool_choice, body.get("stream"), body, route)


def _restore_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Undo normalize_messages tool→user conversion for OpenAI API compatibility.

    normalize_messages preserves tool_call_id field even when converting to user role.
    We check for that field to restore proper tool messages.
    """
    import re
    result: list[dict[str, Any]] = []
    stop_pattern = re.compile(r'\n\n\[STOP:.*$', re.DOTALL)

    for msg in messages:
        tool_call_id = str(msg.get("tool_call_id") or "")
        if msg.get("role") == "user" and tool_call_id:
            # This was originally a tool message — restore it
            content = str(msg.get("content") or "")
            # Strip [STOP:...] failure suffix if present
            content = stop_pattern.sub("", content).strip()
            result.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
        else:
            result.append(msg)
    return result


def _convert_images_for_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal image format → OpenAI vision API format.
    Downloads HTTP URLs and converts to base64 (OpenAI can't fetch external URLs).
    """
    import base64
    from curl_cffi import requests as cffi_requests
    result: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype == "image":
                        data = part.get("data")
                        mime = part.get("mime", "image/png")
                        if isinstance(data, bytes):
                            b64 = base64.b64encode(data).decode("ascii")
                            new_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            })
                        elif isinstance(data, str) and data.startswith("data:"):
                            new_parts.append({
                                "type": "image_url",
                                "image_url": {"url": data},
                            })
                        continue
                    elif ptype == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if isinstance(url, str) and url.startswith("data:"):
                            new_parts.append(part)  # Already base64
                        elif isinstance(url, str) and url.startswith("http"):
                            # Download and convert to base64 (OpenAI can't fetch external URLs)
                            try:
                                # Use standard requests for image downloads (no impersonation needed)
                                import urllib.request
                                req = urllib.request.Request(url, headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                                })
                                with urllib.request.urlopen(req, timeout=15) as resp:
                                    img_data = resp.read()
                                    mime = resp.headers.get("Content-Type", "image/jpeg")
                                    b64 = base64.b64encode(img_data).decode("ascii")
                                    new_parts.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                                    })
                            except Exception as e:
                                logger.warning({"event": "image_download_failed", "url": url[:120], "error": str(e)[:100]})
                        continue
                new_parts.append(part)
            result.append({**msg, "content": new_parts})
        else:
            result.append(msg)
    return result


def _ensure_openai_provider():
    """Auto-create openai custom provider if missing (for web session routing)."""
    from services.providers.custom_openai import get_custom_providers
    providers = get_custom_providers()
    if "openai" not in providers:
        cfg = config.data
        cfg.setdefault("custom_providers", {})["openai"] = {
            "name": "OpenAI",
            "prefix": "openai",
            "base_url": "https://api.openai.com",
            "api_key": "sk-auto-created",
            "enabled": True,
        }
        config._save()
        logger.info({"event": "openai_provider_auto_created"})


def _handle_openai_api_chat(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    stream: bool,
    body: dict[str, Any],
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """OpenAI-API path (3rd group, kept fully separate from free & codex).

    Serves raw OpenAI API-key (sk-...) or `standard`/`openai` JWT accounts by
    calling api.openai.com through the custom:openai provider. Reached only via
    the explicit `oai/` prefix — never auto-detected inside the free path.
    """
    openai_model = model[4:] if model.startswith("oai/") else model
    if not openai_model or openai_model in ("auto", "chatgpt/auto"):
        openai_model = config.openai_default_model or "gpt-4o"
    if openai_model.startswith("chatgpt/"):
        openai_model = openai_model[len("chatgpt/"):]

    token = account_service.get_text_access_token(account_type="openai")
    if not token:
        raise RuntimeError("no usable OpenAI-API (sk-/standard) account")

    messages = _restore_tool_messages(messages)
    messages = _convert_images_for_openai(messages)
    _ensure_openai_provider()

    logger.info({"event": "openai_api_chat_routed", "model": openai_model})
    return _handle_custom_openai_chat(
        "custom:openai", openai_model, messages, tools, tool_choice,
        stream, body, force_token=token,
    )

# Device keywords that should trigger tool call forcing
_FORCE_TOOL_KEYWORDS = [
    "trạng thái", "bật", "tắt", "mở", "đóng", "kiểm tra",
    "đèn", "quạt", "cửa", "điều hòa", "máy lạnh", "camera",
    "cảm biến", "công tắc", "ổ cắm", "rèm", "bình nóng lạnh",
    "tivi", "ti vi", "loa", "máy bơm", "nhiệt độ", "độ ẩm",
    "phòng khách", "phòng ngủ", "phòng học", "phòng bếp",
    "ban công", "nhà tắm", "nhà vệ sinh", "hành lang", "sân",
    "tầng", "cầu thang", "garage", "cổng",
    "thiết bị", "toàn bộ", "tất cả", "thời tiết",
]

# Greetings and trivial chat patterns that never need MCP tools.
# Matching is prefix-based: if the user message starts with one of these
# (after stripping), it's considered a trivial chat and MCP tools are skipped.
_TRIVIAL_GREETINGS = [
    "xin chào", "chào", "hello", "hi ", "hi.", "hi\n", "hey", "ê ", "alo", "a lô",
    "good morning", "good afternoon", "good evening",
    "cảm ơn", "thanks", "thank you",
    "tạm biệt", "bye", "goodbye",
    "có đó không", "khỏe không", "ăn cơm chưa",
    "ok", "okay", "được rồi", "ừ ", "ờ ",
]

# Tool-relevant domain keywords — if any of these appear in the user text,
# the query is NOT trivial and needs MCP tools.
_TOOL_DOMAIN_KEYWORDS = [
    "thời tiết", "nhiệt độ", "mưa", "nắng", "bão", "gió", "áp suất", "độ ẩm",
    "tìm", "kiếm", "search", "tra cứu", "wikipedia", "định nghĩa",
    "chứng khoán", "cổ phiếu", "giá vàng", "tỷ giá", "ngoại tệ", "xăng dầu",
    "tin tức", "báo ", "tin mới", "bản tin",
    "arxiv", "nghiên cứu", "paper", "bài báo",
    "luật ", "nghị định", "thông tư",
    "bệnh", "thuốc", "triệu chứng", "y tế", "bác sĩ",
    "học ", "giáo dục", "bài tập", "giảng", "trường",
    "youtube", "video", "transcript",
    "dịch", "translate", "phiên âm",
    "lịch âm", "âm lịch", "ngày", "tết",
    "phạt nguội", "biển số",
]


def _is_trivial_chat(user_text: str) -> bool:
    """Return True if this is a simple greeting/chat that doesn't need MCP tools."""
    if not user_text:
        return False
    text = user_text.strip()
    text_lower = text.lower()
    # Must be reasonably short to be trivial
    if len(text) > 80:
        return False
    # Must not contain tool-relevant keywords (weather, search, stocks, etc.)
    for kw in _TOOL_DOMAIN_KEYWORDS:
        if kw in text_lower:
            return False
    # Must not contain HA device keywords
    for kw in _FORCE_TOOL_KEYWORDS:
        if kw in text_lower:
            return False
    # Must start with a known greeting pattern OR be very short (< 15 chars)
    for g in _TRIVIAL_GREETINGS:
        if text_lower.startswith(g):
            return True
    if len(text) <= 12:
        return True
    return False


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract the text of the last user message — handles BOTH a plain string
    and HA's structured list content (e.g. [{"type":"text","text":"trạng thái
    nhà"}, {"type":"image_url",...}]). Returning "" for list content silently
    broke the status/control tool heuristics on real HA requests."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(
                    str(p.get("text") or "") for p in c
                    if isinstance(p, dict) and p.get("type") in ("text", "input_text")
                )
            return ""
    return ""


def _prefetch_ha_context_if_needed(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    token: str,
) -> list[dict[str, Any]]:
    """Pre-fetch HA device context BEFORE calling the LLM (chatgpt free path).

    HA sends a single HTTP request and reads a single streaming response.
    There is no way to do a 2nd LLM call inside the same connection.

    Strategy (mirrors how Codex reasons):
    1. Extract device/room keywords from user query
    2. Call ha_search_entities to find matching entity_ids (compact result)
    3. Call ha_get_state on each matched entity for live on/off state
    4. Inject compact result (~500 chars) into user message
    Falls back to a short summary from format_states_context if no specific
    entity is found.
    """
    # UNCONDITIONALLY trim HA's own exposed-entity dump for chatgpt_free.
    # When ~600+ entities are exposed to Assist, HA appends a 40KB+ block
    # "areas and the devices in this smart home" to the system prompt.
    # chatgpt.com free rejects payloads >45KB (413), so we always cut it.
    _HA_ASSIST_MARKER = "areas and the devices in this smart home"
    cleaned_messages = []
    for m in messages:
        content = str(m.get("content", ""))
        if m.get("role") == "system" and _HA_ASSIST_MARKER in content:
            idx = content.find(_HA_ASSIST_MARKER)
            line_start = content.rfind("\n", 0, idx)
            trimmed = (content[:line_start] if line_start > 0 else content[:idx]).rstrip()
            logger.info({"event": "ha_prefetch_trim_assist_entities",
                         "removed_chars": len(content) - len(trimmed)})
            cleaned_messages.append({**m, "content": trimmed})
            continue
        cleaned_messages.append(m)
    messages = cleaned_messages

    from services.ha_client import is_ha_query

    # Skip HA prefetch if search results are already injected.
    # search_service injects results containing weather/nature keywords (e.g.
    # "nhiệt độ", "mây") that can falsely trigger is_ha_query, which then
    # injects HA device context over the search content — causing "Xin chào" AI.
    _SEARCH_RESULT_MARKER = "kết quả tìm kiếm"
    for m in messages:
        if m.get("role") == "user" and _SEARCH_RESULT_MARKER in str(m.get("content", "")).lower():
            logger.info({"event": "ha_prefetch_skip", "reason": "search_results_already_injected"})
            return messages

    if not is_ha_query(messages):
        return messages



    # Only pre-fetch if user query is about device state
    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            user_text = c if isinstance(c, str) else " ".join(
                str(p.get("text", "")) for p in c if isinstance(p, dict)
            )
            break

    if not _has_device_keyword(user_text):
        return messages

    # Already has live tool result injected? Skip.
    for m in messages:
        if m.get("role") == "user" and "KẾT QUẢ TỪ HỆ THỐNG" in str(m.get("content", "")):
            return messages
        if m.get("role") == "user" and "DỮ LIỆU THỜI GIAN THỰC TỪ HOME ASSISTANT" in str(m.get("content", "")):
            return messages


    # ── Targeted lookup (Codex-style: search → get_state) ──────────────────
    # Extract room/device keywords from query. Use the same keyword list
    # the static registry uses so lookup is consistent.
    import unicodedata as _ud

    def _strip_diacritics(t: str) -> str:
        nfkd = _ud.normalize("NFKD", t.lower())
        return "".join(c for c in nfkd if not _ud.combining(c))

    user_folded = _strip_diacritics(user_text)

    # Keywords to search for in HA: room names + device type words
    _SEARCH_TOKENS = [
        # rooms
        "ban công", "bep", "phong ngu", "phong khach", "phong hoc",
        "phong tam", "hanh lang", "san", "cau thang", "garage",
        # devices
        "den", "quat", "dieu hoa", "may lanh", "rem", "cua",
        "cong tac", "o cam", "khoa", "may bom",
    ]
    _TOKEN_MAP = {
        # map folded → Vietnamese search term for ha_search_entities
        "ban cong": "ban công", "bep": "bếp", "phong ngu": "phòng ngủ",
        "phong khach": "phòng khách", "phong hoc": "phòng học",
        "phong tam": "phòng tắm", "hanh lang": "hành lang",
        "den": "đèn", "quat": "quạt", "dieu hoa": "điều hòa",
        "may lanh": "máy lạnh", "rem": "rèm", "cua": "cửa",
        "cong tac": "công tắc", "o cam": "ổ cắm", "khoa": "khóa",
        "may bom": "máy bơm",
    }

    found_tokens: list[str] = []
    for tok in _SEARCH_TOKENS:
        # Check if the room/device token is in the user query (ignoring diacritics)
        # We replace spaces with empty string to match "ban công" -> "bancong" if needed,
        # but the simplest is just checking tok in user_folded.
        # But wait, tok is already diacritic-less in _SEARCH_TOKENS except for room names?
        # Let's fix _SEARCH_TOKENS to be fully folded.
        pass

    # Let's just do a direct multi-token search on the cached states.
    context_lines: list[str] = []
    
    # Extract ALL tokens from user_folded to match against entity names
    user_words = user_folded.split()
    
    # Check if this is a general query
    general_phrases = ["trang thai nha", "tong quan", "ca nha", "tat ca", "tinh hinh", "nha hien tai", "trong nha"]
    is_general = any(p in user_folded for p in general_phrases)
    
    if is_general:
        # Force fallback (full context) for general queries
        search_words = set()
    else:
        # Meaningful words to look for (ignore stop words)
        search_words = set([w for w in user_words if len(w) > 1 and w not in (
            "dang", "bat", "hay", "tat", "cho", "xin", "hoi", "thong", "tin",
            "trang", "thai", "cua", "co", "khong", "la", "gi", "nhe", "nha", "oi",
            "hien", "tai", "tat", "ca", "cac", "thiet", "bi"
        )])
    
    
    try:
        from services.ha_client import get_states
        # Real-time: a status query must reflect the CURRENT state, not the
        # hourly cache. Fetch fresh (refreshes the shared cache too, so the
        # exposed-only block below reuses it without a second HA call).
        states = get_states(use_cache=False)
        if states:
            # Score each entity by how many search words it matches
            matched_entities = []
            for s in states:
                eid = s.get("entity_id", "").lower()
                name = s.get("attributes", {}).get("friendly_name", "")
                name_folded = _strip_diacritics(name)
                # Combine eid and folded name for searching
                searchable = f"{eid} {name_folded}"
                
                score = sum(1 for w in search_words if w in searchable)
                if score > 0:
                    matched_entities.append((score, s))
            
            # Sort by score descending, take top 15
            matched_entities.sort(key=lambda x: x[0], reverse=True)
            
            for score, s in matched_entities[:15]:
                # If score is too low and we have many matches, maybe skip. 
                # But taking top 15 is safe.
                eid = s.get("entity_id", "")
                st = str(s.get("state", "unknown"))
                attrs = s.get("attributes", {}) or {}
                name = attrs.get("friendly_name", eid)
                unit = attrs.get("unit_of_measurement", "")
                state_str = f"{st} {unit}".strip() if unit else st
                context_lines.append(f"- {name} ({eid}): **{state_str}**")
                
    except Exception as exc:
        logger.warning({"event": "ha_prefetch_search_failed", "error": str(exc)[:80]})


    # Preferred fallback for general queries: report exactly the entities HA
    # exposes to Assist (curated ≈116) with their live states — not all ~989.
    # Honors the user's "Expose" config and keeps the injected context small.
    if not context_lines:
        try:
            from services.ha_client import get_states, get_exposed_entity_ids
            exposed = get_exposed_entity_ids()
            if exposed:
                ex_lines = []
                for s in get_states():
                    eid = s.get("entity_id", "")
                    if eid not in exposed:
                        continue
                    attrs = s.get("attributes", {}) or {}
                    name = attrs.get("friendly_name", eid)
                    st = str(s.get("state", "unknown"))
                    unit = str(attrs.get("unit_of_measurement", "") or "").strip()
                    state_str = f"{st} {unit}".strip() if unit else st
                    ex_lines.append(f"- {name} ({eid}): **{state_str}**")
                if ex_lines:
                    context_lines = ex_lines
                    logger.info({"event": "ha_prefetch_exposed_only",
                                 "exposed_total": len(exposed),
                                 "lines": len(ex_lines)})
        except Exception as exc:
            logger.warning({"event": "ha_prefetch_exposed_failed",
                            "error": str(exc)[:80]})

    if not context_lines:
        # Fallback: use static cache but only take the controllable device lines
        # (skip sensors/weather) and limit to 25000 chars total
        try:
            from services.ha_client import format_states_context
            cached = format_states_context()
            # Smart filter: only include ACTIVE core devices, ALL doors/locks, and IMPORTANT sensors
            compact_lines = []
            for line in cached.splitlines():
                lower = line.lower()
                
                is_core = any(x in lower for x in ('light.', 'switch.', 'climate.', 'fan.', 'cover.', 'lock.'))
                is_sensor = any(x in lower for x in ('sensor.', 'binary_sensor.'))
                
                if not (is_core or is_sensor):
                    continue
                    
                # Remove the check that skips off devices so ALL lights/switches are visible
                        
                # For sensors, only include important ones to avoid spam
                if is_sensor:
                    important_keywords = [
                        'nhiệt độ', 'độ ẩm', 'chuyển động', 'khói', 'cửa', 'pin', 'power', 
                        'nhiet', 'am', 'door', 'motion', 'smoke', 'battery',
                        'âm lịch', 'rằm', 'giỗ', 'công suất', 'điện', 'aptomat', 'hôm nay',
                        'lịch', 'calendar', 'aqi', 'không khí', 'air'
                    ]
                    if not any(k in lower for k in important_keywords):
                        continue
                        
                compact_lines.append(line)
                if len("\n".join(compact_lines)) > 25000: # Safe upper bound
                    break
            if compact_lines:
                context_lines = compact_lines
                logger.info({"event": "ha_prefetch_fallback_compact",
                             "lines": len(context_lines)})
        except Exception:
            pass

    if not context_lines:
        logger.info({"event": "ha_prefetch_no_data"})
        return messages

    live_summary = "\n".join(context_lines)
    logger.info({"event": "ha_prefetch_ok", "context_len": len(live_summary)})

    msg_context = (
        f"\n\n[DỮ LIỆU THỜI GIAN THỰC TỪ HOME ASSISTANT]:\n"
        f"Đây là danh sách trạng thái hiện tại của các thiết bị.\n"
        f"Nguyên tắc trả lời:\n"
        f"1. Nếu user hỏi TỔNG QUAN (ví dụ: trạng thái nhà): Hãy trình bày theo ĐÚNG THỨ TỰ sau để đảm bảo luôn đầy đủ, không bị thiếu sót ngẫu nhiên:\n"
        f"   - An ninh & Cửa: Trạng thái cửa chính, khoá, các cảm biến chuyển động/khói.\n"
        f"   - Nhiệt độ & Môi trường: Quạt, Điều hoà, nhiệt độ/độ ẩm các phòng, thời tiết/AQI ngoài trời.\n"
        f"   - Ánh sáng: Gom nhóm trạng thái của toàn bộ các đèn.\n"
        f"   - Thiết bị & Điện năng: Aptomat tổng, công suất, điện tiêu thụ, bình nóng lạnh.\n"
        f"   - Pin: Chỉ nhắc đến các thiết bị sắp hết pin (0-15%) hoặc cảnh báo cần thiết.\n"
        f"   - Sự kiện & Âm lịch: Lịch âm, ngày rằm, ngày giỗ.\n"
        f"   Tuyệt đối BỎ QUA các thông số kỹ thuật mạng (như Remote UI, Ping, thiết bị nội bộ) không liên quan đến sinh hoạt.\n"
        f"2. Nếu user hỏi THIẾT BỊ CỤ THỂ: Trả lời CHỈ BẰNG 1 CÂU DUY NHẤT (ví dụ: 'Đèn phòng học đang tắt'). TUYỆT ĐỐI CẤM giải thích dài dòng. CẤM nhắc đến các thông số phụ (như manual, auto, công tắc, automation) trừ khi user chủ động hỏi.\n"
        f"Mỗi dòng dữ liệu bên dưới có định dạng: `Tên (entity_id) | Trạng thái`.\n\n"
        f"{live_summary}\n"
    )

    # Strip static Device Registry (server's own) — live prefetch replaces it.
    cleaned_messages = []
    for m in messages:
        content = str(m.get("content", ""))
        if m.get("role") == "system" and "Device Registry" in content:
            logger.info({"event": "ha_prefetch_strip_registry", "reason": "live_context_available"})
            continue
        cleaned_messages.append(m)
    messages = cleaned_messages

    # Hard cap: measure current payload size, trim context to fit within 38KB total
    # chatgpt.com free rejects payloads >45KB (413). Leave 7KB headroom for overhead.
    _MAX_PAYLOAD_CHARS = 38_000
    current_payload_chars = sum(len(str(m.get("content", ""))) for m in messages)
    available = _MAX_PAYLOAD_CHARS - current_payload_chars
    if available < 500:
        logger.info({"event": "ha_prefetch_skip_payload_full",
                     "current_chars": current_payload_chars, "max": _MAX_PAYLOAD_CHARS})
        return messages
    if len(msg_context) > available:
        msg_context = msg_context[:available]
        logger.info({"event": "ha_prefetch_context_trimmed",
                     "trimmed_to": available, "original": len(live_summary)})

    # Inject into the LAST user message
    injected = []
    injected_flag = False
    for m in reversed(messages):
        if m.get("role") == "user" and not injected_flag:
            new_content = str(m.get("content", "")) + msg_context
            injected.append({**m, "content": new_content})
            injected_flag = True
        else:
            injected.append(m)

    return list(reversed(injected))



def _has_device_keyword(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in _FORCE_TOOL_KEYWORDS)


def _request_wants_plain_text(messages: list[dict[str, Any]]) -> bool:
    """Heuristic: last user turn looks like a device-control / status query
    aimed at HA voice or a plain-text surface. We strip markdown for these so
    `**tắt**` doesn't leak through as literal asterisks. Skip when the message
    explicitly contains a markdown table — user clearly wants rich formatting.

    Also force plain text when a system message explicitly forbids markdown
    (e.g. the HA "AI Agent" voice prompt: "Format responses using plain text
    only. Do not use markdown..."). This covers search/knowledge answers
    ("giá xăng", "giá vàng") that have no device keyword but still must arrive
    as plain text — while the sibling agent whose prompt says "using markdown"
    is left untouched.
    """
    for m in messages:
        if m.get("role") != "system":
            continue
        sys_text = m.get("content")
        if isinstance(sys_text, str):
            low = sys_text.lower()
            if "plain text only" in low or "do not use markdown" in low or "không dùng markdown" in low:
                return True
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                str(p.get("text") or "") for p in content
                if isinstance(p, dict) and p.get("type") in ("text", "input_text")
            )
        else:
            text = ""
        if not text:
            return False
        # Hint: user wrote a table or explicitly asked for one → keep markdown
        if "|--" in text or "bảng" in text.lower() or "table" in text.lower():
            return False
        return _has_device_keyword(text)
    return False


# Markdown patterns we strip. Bold / italic / inline code / strike / headings.
# Tables (lines with `|`) and code fences (```...```) are left alone.
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_BOLD_UNDER = re.compile(r"__(.+?)__", re.DOTALL)
_MD_ITALIC_STAR = re.compile(r"(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])")
_MD_ITALIC_UNDER = re.compile(r"(?<![_\w])_(?!\s)([^_\n]+?)(?<!\s)_(?![_\w])")
_MD_CODE = re.compile(r"`([^`\n]+)`")
_MD_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)

# Backend artifact patterns that should NEVER reach the user.
# ChatGPT web-search backends sometimes leak raw citation markers when the
# response isn't converted to proper inline citations.
#   citeturn0search0 / citeturn0search0turn0search2turn0search8 / ...
#   [oaicite:0] / 【oaicite:0】 / oaicite:N (various brackets)
_CITE_TURN = re.compile(r"cite(?:turn\d+\w+)+")
_OAICITE = re.compile(r"[\[【]?\s*oaicite[^\]】\)]*[\]】\)]?")
# Tool-call args leaking as text: `entity["city","Hà Nội",...]` — match the
# `entity[ "string", "string", ... ]` shape with at least one quoted arg so we
# don't accidentally strip valid `entity[0]` / `entity[i]` code in answers.
# Also handles ChatGPT's private-use Unicode wrappers \ue200...\ue202...\ue201 that
# the web model inserts around entity references in its streamed output.
_ENTITY_LEAK = re.compile(
    r'\ue200?'               # optional Unicode start sentinel
    r'\bentity'
    r'\ue202?'               # optional Unicode bracket-open sentinel
    r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]'  # ["arg","arg",...] body
    r'\ue201?'               # optional Unicode end sentinel
)
# Internal trace appended by openai_backend_api._api_messages_to_conversation_messages
# when an assistant turn carried tool_calls. The ChatGPT web model sometimes
# echoes this line verbatim at the start of its next answer (observed on
# "trạng thái nhà" → "[System Log: You executed tool GetLiveContext with args {}]").
# It is internal bookkeeping and must never reach the user.
_SYSLOG_LEAK = re.compile(r"\n*\[System Log:[^\]]*\]\n*")
# ChatGPT web model leaks image-gen directives: image_group{"aspect_ratio":...}
_IMAGE_GROUP_LEAK = re.compile(r'\bimage_group\s*\{[^}]*\}', re.DOTALL)
# After entity[] removal, orphan "- :" bullet lines remain
_ORPHAN_BULLET = re.compile(r'^-\s*:\s*$', re.MULTILINE)


def _strip_artifacts_inline(text: str) -> str:
    if not text:
        return text
    out = _CITE_TURN.sub("", text)
    out = _OAICITE.sub("", out)
    out = _ENTITY_LEAK.sub("", out)
    out = _IMAGE_GROUP_LEAK.sub("", out)
    out = _SYSLOG_LEAK.sub("", out)
    out = _ORPHAN_BULLET.sub("", out)
    return out


def _strip_markdown_inline(text: str) -> str:
    if not text:
        return text
    out = _strip_artifacts_inline(text)
    out = _MD_BOLD.sub(r"\1", out)
    out = _MD_BOLD_UNDER.sub(r"\1", out)
    out = _MD_ITALIC_STAR.sub(r"\1", out)
    out = _MD_ITALIC_UNDER.sub(r"\1", out)
    out = _MD_CODE.sub(r"\1", out)
    out = _MD_STRIKE.sub(r"\1", out)
    out = _MD_HEADING.sub("", out)
    return out


def _strip_markdown_in_response(result: dict[str, Any]) -> dict[str, Any]:
    choices = result.get("choices") or []
    for ch in choices:
        msg = ch.get("message") if isinstance(ch, dict) else None
        if isinstance(msg, dict):
            txt = msg.get("content")
            if isinstance(txt, str):
                msg["content"] = _strip_markdown_inline(txt)
    return result


def _strip_markdown_in_stream(it: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Collect ALL content chunks, strip markdown on the joined text, then
    replay them as: pass-through (non-content chunks) + one stripped content
    chunk just before the finish_reason chunk.

    Why not stream-strip incrementally: markdown markers like `**` can span
    multiple chunks and the OpenAI streaming protocol has no way to "un-emit"
    a character already sent. Per-chunk strip heuristics leak the opening
    marker when its close hasn't arrived yet. Device-control responses are
    short (~100 chars) so dropping live-typing UX is acceptable.

    Tool-call chunks (delta.tool_calls) pass through unchanged so MCP/HA
    server-side execution still works.
    """
    pending: list[dict[str, Any]] = []
    full_text = ""
    emitted_final = False
    for chunk in it:
        try:
            choices = chunk.get("choices") or []
            has_finish = False
            has_content = False
            for ch in choices:
                if not isinstance(ch, dict):
                    continue
                if ch.get("finish_reason"):
                    has_finish = True
                delta = ch.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        full_text += content
                        # Strip content from THIS chunk; we'll re-emit the
                        # whole stripped text just before the finish chunk.
                        delta["content"] = ""
                        has_content = True

            if has_finish and not emitted_final and full_text:
                import logging
                logging.getLogger("uvicorn.error").info({"event": "debug_final_stream", "text": full_text[:1000]})
                
                # Emit the stripped full text as a content chunk first
                stripped = _strip_markdown_inline(full_text)
                content_chunk = {
                    "id": chunk.get("id"),
                    "object": chunk.get("object"),
                    "created": chunk.get("created"),
                    "model": chunk.get("model"),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": stripped},
                        "finish_reason": None,
                    }],
                }
                yield content_chunk
                emitted_final = True
        except Exception:
            pass
        yield chunk


def _inject_tool_force_hint(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    # Legacy: No longer used since HA redesigned
    return messages


def _stream_chatgpt_addon(backend, messages, model, tools, tool_choice):
    """Stream from chatgpt.com backend, extracting XML tool calls from response."""
    messages = _inject_tool_force_hint(messages, tools)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    sent_role = False
    accumulated = ""
    request = ConversationRequest(model=model, messages=messages, tools=tools, tool_choice=tool_choice)
    for delta_text in stream_text_deltas(backend, request):
        accumulated += delta_text
        if not sent_role:
            sent_role = True
            yield completion_chunk(model, {"role": "assistant", "content": delta_text}, None, completion_id, created)
        else:
            yield completion_chunk(model, {"content": delta_text}, None, completion_id, created)

    if tools:
        tool_calls = _extract_xml_tool_calls_from_text(accumulated)
        if tool_calls:
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}],
            }

    if not sent_role:
        yield completion_chunk(model, {"role": "assistant", "content": ""}, None, completion_id, created)
    yield completion_chunk(model, {}, "stop", completion_id, created)


def _chatgpt_addon_completion(model, messages, tools, tool_choice):
    """Non-streaming chatgpt.com backend, extracting XML tool calls from response."""
    messages = _inject_tool_force_hint(messages, tools)
    backend = text_backend()
    request = ConversationRequest(model=model, messages=messages, tools=tools, tool_choice=tool_choice)
    content = collect_text(backend, request)

    if tools:
        tool_calls = _extract_xml_tool_calls_from_text(content)
        if tool_calls:
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
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": count_message_tokens(messages, model),
                    "completion_tokens": count_text_tokens(content, model),
                    "total_tokens": count_message_tokens(messages, model) + count_text_tokens(content, model),
                },
            }

    return completion_response(model, content, messages=messages)


def _handle_opencode_chat(
    model: str,
    messages: list[dict[str, Any]],
    stream: bool,
    body: dict[str, Any],
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """OpenCode chat — no 24KB payload limit, no auth required."""
    from services.providers.opencode import opencode_provider

    # Strip oc/ prefix if present
    opencode_model = model
    if model.startswith("oc/"):
        opencode_model = model[3:]
    elif model == "auto":
        opencode_model = "auto"

    # Resolve auto using enabled_models order from settings
    if opencode_model == "auto" or not opencode_model:
        ms = config.data.get("model_settings") or {}
        enabled = (ms.get("enabled_models") or {}).get("opencode") if isinstance(ms, dict) else None
        if isinstance(enabled, list):
            for m in enabled:
                m = str(m).strip()
                if not m or m == "auto":
                    continue
                if m.startswith("oc/"):
                    m = m[3:]
                if m:
                    opencode_model = m
                    break

    logger.info({
        "event": "opencode_chat_routed",
        "model": opencode_model,
        "stream": stream,
        "message_count": len(messages),
    })

    temperature = float(body.get("temperature") or 0.7)
    max_tokens = body.get("max_tokens")

    if stream:
        return _stream_opencode_response(opencode_model, messages, temperature, max_tokens, body)
    else:
        return _opencode_completion_response(opencode_model, messages, temperature, max_tokens)


def _stream_opencode_response(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int | None,
    body: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Stream response from OpenCode — extract tool calls from text if present."""
    from services.providers.opencode import opencode_provider

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    sent_role = False
    accumulated = ""

    try:
        sse_stream = opencode_provider.chat_completions(
            messages=messages, model=model, stream=True,
            temperature=temperature, max_tokens=max_tokens,
        )

        for line in sse_stream:
            if line.startswith("data: "):
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta_text = ""
                    choices = chunk.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        delta_text = str(choices[0].get("delta", {}).get("content", "") or "")
                    accumulated += delta_text
                    chunk["id"] = completion_id
                    chunk["created"] = created
                    chunk["model"] = model
                    if delta_text and not sent_role:
                        chunk["choices"][0]["delta"] = {"role": "assistant", "content": delta_text}
                        sent_role = True
                    yield chunk
                except Exception:
                    continue

        # On completion, check if response contains tool calls
        tool_calls = _extract_tool_calls_from_text(accumulated)
        if tool_calls:
            yield {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}],
            }

        if not sent_role:
            yield completion_chunk(model, {"role": "assistant", "content": ""}, None, completion_id, created)
        yield completion_chunk(model, {}, "stop", completion_id, created)

    except Exception as exc:
        logger.error({"event": "opencode_stream_fatal", "error": str(exc)})
        yield completion_chunk(model, {"role": "assistant", "content": f"OpenCode error: {exc}"}, "stop", completion_id, created)


def _opencode_completion_response(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Non-streaming response from OpenCode — parse text JSON into native tool_calls."""
    from services.providers.opencode import opencode_provider

    try:
        result = opencode_provider.chat_completions(
            messages=messages,
            model=model,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = ""
        choices = result.get("choices", [])
        if choices and isinstance(choices[0], dict):
            content = str(choices[0].get("message", {}).get("content", "") or "")

        # Parse text JSON tool calls into native format
        tool_calls = _extract_tool_calls_from_text(content) or _extract_xml_tool_calls_from_text(content)
        message = {"role": "assistant", "content": ""}
        if tool_calls:
            message["tool_calls"] = tool_calls
        else:
            message["content"] = content

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": count_message_tokens(messages, model),
                "completion_tokens": count_text_tokens(content, model),
                "total_tokens": count_message_tokens(messages, model) + count_text_tokens(content, model),
            },
        }

    except Exception as exc:
        logger.error({"event": "opencode_completion_error", "error": str(exc)})
        return completion_response(
            model=model,
            content=f"OpenCode error: {exc}",
            messages=messages,
        )


# ── Helper for entity_id → domain conversion ──

def _convert_params(params):
    """Convert OpenCode params to HA-compatible format (entity_ids → domain)."""
    if isinstance(params, dict) and "entity_ids" in params:
        eids = params["entity_ids"]
        if isinstance(eids, list) and eids:
            domains = list(set(eid.split(".")[0] for eid in eids if isinstance(eid, str)))
            return {"domain": domains}
    if isinstance(params, list):
        if all(isinstance(x, str) for x in params):
            if any("." in str(x) for x in params):
                domains = list(set(str(x).split(".")[0] for x in params))
                return {"domain": domains}
            return {"entities": params}
        return {"entities": params}
    if not isinstance(params, dict):
        return {}
    return params


def _extract_tool_calls_from_text(text: str) -> list[dict[str, Any]] | None:
    """Parse text tool calls from OpenCode response.

    Only extract if the response is PURELY a tool call (no conversational answer).
    If there's text after the tool call JSON, assume it's already a complete answer.
    """
    if not text:
        return None
    import re as _re

    # Check if this is a pure tool call — first non-whitespace is a tool name or JSON
    stripped = text.strip()

    # If text contains both a tool call AND a conversational answer (after the JSON),
    # the answer is the main intent — don't extract tool call
    # Pattern: "ToolName\n{json}\n\nAnswer text..." → already answered, skip

    # Format 1: JSON with "action" key
    match = _re.search(r'\{[^{}]*"action"\s*:\s*"([^"]+)"\s*[,}][^{}]*\}', stripped)
    if match:
        # Only use if this is MOSTLY a tool call (not followed by long text)
        after_json = stripped[match.end():].strip()
        if len(after_json) < 50:  # Short or no follow-up text → pure tool call
            try:
                data = json.loads(match.group(0))
                action = data.get("action", "")
                params = _convert_params(data.get("params") or data.get("entity_ids") or data.get("domain") or {})
                if action:
                    return [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                             "function": {"name": action, "arguments": json.dumps(params, ensure_ascii=False)}}]
            except (json.JSONDecodeError, AttributeError):
                pass

    # Format 2: ToolName\n{JSON}
    match = _re.search(r'^([A-Z][A-Za-z0-9_]+)\s*\n\s*(\[[^\]]*\]|\{[^{}]*\})', stripped)
    if match:
        after_json = stripped[match.end():].strip()
        if len(after_json) < 50:
            try:
                tool_name = match.group(1)
                params = _convert_params(json.loads(match.group(2)))
                if not isinstance(params, dict): params = {}
                return [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                         "function": {"name": tool_name, "arguments": json.dumps(params, ensure_ascii=False)}}]
            except (json.JSONDecodeError, AttributeError):
                pass

    # Format 3: {"tool": "X"} or {"name": "X"}
    match = _re.search(r'\{\s*"(?:tool|name)"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{.*?\}|\[.*?\])\s*\}', stripped, _re.DOTALL)
    if match:
        after_json = stripped[match.end():].strip()
        if len(after_json) < 50:
            try:
                tool_name = match.group(1)
                params = _convert_params(json.loads(match.group(2)))
                if not isinstance(params, dict): params = {}
                return [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                         "function": {"name": tool_name, "arguments": json.dumps(params, ensure_ascii=False)}}]
            except (json.JSONDecodeError, AttributeError):
                pass

    return None


def _extract_xml_tool_calls_from_text(text: str) -> list[dict[str, Any]] | None:
    """Parse XML-wrapped tool calls from chatgpt.com backend text responses.

    The AI is instructed by _build_tool_prompt to wrap tool calls in:
    ```xml
    <tool_call name="tool_name">{"arg": "value"}</tool_call>
    ```

    Returns OpenAI-format tool_calls list, or None if no tool calls found.
    """
    if not text or not text.strip():
        return None

    import re as _re

    # Prefer matches inside ```xml ... ``` fenced blocks
    fence_pattern = _re.compile(r'```(?:xml)?\s*\n?(.*?)```', _re.DOTALL)
    fence_matches = fence_pattern.findall(text)
    search_text = " ".join(fence_matches) if fence_matches else text

    tool_calls = []
    seen_names: set[str] = set()

    for match in TOOL_CALL_RE.finditer(search_text):
        name = match.group(1).strip()
        args_text = match.group(2).strip()
        try:
            args = json.loads(args_text) if args_text else {}
            if not isinstance(args, dict):
                args = {}
        except (json.JSONDecodeError, TypeError):
            logger.warning({"event": "xml_tool_call_parse_failed", "name": name, "args_raw": args_text[:200]})
            continue

        if name in seen_names:
            continue
        seen_names.add(name)

        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })

    # Self-closing <tool_call name="X"/>
    for match in TOOL_CALL_SELF_CLOSING_RE.finditer(search_text):
        name = match.group(1).strip()
        if name not in seen_names:
            seen_names.add(name)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            })

    return tool_calls if tool_calls else None


def _handle_openai_oauth_chat(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    stream: bool,
    body: dict[str, Any],
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """Use Codex OAuth token to call chatgpt.com/backend-api/codex/responses — same as 9router."""
    from services.providers.openai_oauth import codex_oauth

    pure_model = model
    for _p in ("cx/", "codex/", "paid/"):
        if pure_model.startswith(_p):
            pure_model = pure_model[len(_p):]
            break
    if not pure_model or pure_model == "auto":
        pure_model = "auto"

    logger.info({
        "event": "openai_oauth_chat",
        "model": pure_model,
        "stream": stream,
    })

    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")

    attempted: set[str] = set()
    last_error = ""
    usage_limit_hits = 0
    max_account_switches = 8  # codext-style: try up to 8 accounts before giving up

    while True:
        try:
            token = codex_oauth.get_token_for_request(attempted)
        except RuntimeError as exc:
            raise RuntimeError(str(exc))  # Raise so combo can fallback

        if token in attempted or usage_limit_hits >= max_account_switches:
            break
        attempted.add(token)

        # A paid account that landed in the codex pool *by plan only* (logged
        # in via Google → chatgpt.com web JWT, no real Codex token) is tagged
        # by plan, NOT by a "codex" type. Route it to the shared chatgpt.com
        # transport instead of the Codex responses API. We key off the account
        # TYPE tag (not JWT introspection): a real Codex onboard always tags
        # type="codex"; detect_token_type() is unreliable here because it
        # returns "google" for codex tokens issued through a Google login
        # before it ever checks chatgpt_account_id. "phân nhóm theo plan, tự
        # đổi route". On any lookup failure, default to the Codex path.
        try:
            _acc = account_service.get_account(token) or {}
            _is_real_codex = "codex" in str(_acc.get("type") or "").split(",")
        except Exception as _exc:
            logger.warning({"event": "codex_type_lookup_failed", "error": str(_exc)[:120]})
            _is_real_codex = True
        if not _is_real_codex:
            from services.providers.chatgpt_free import call_chatgpt_web
            logger.info({"event": "codex_webjwt_fallback", "reason": "paid_plan_no_codex_token"})
            account_service.mark_text_used(token)
            return call_chatgpt_web(token, pure_model, messages, tools, tool_choice, stream, body)

        try:
            if stream:
                result = codex_oauth.chat_completions(
                    access_token=token, messages=messages, model=pure_model,
                    stream=True, temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice=tool_choice,
                )
                # On successful stream start, clear any parked resume for this token
                try:
                    from services.account_switch_resume import account_switch_resume
                    account_switch_resume.clear_parked(token[:40], reason="stream_started")
                except Exception:
                    pass
                return result
            else:
                result = codex_oauth.chat_completions(
                    access_token=token, messages=messages, model=pure_model,
                    stream=False, temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice=tool_choice,
                )
                account_service.mark_text_used(token)
                # Clear any parked resume on success
                try:
                    from services.account_switch_resume import account_switch_resume
                    account_switch_resume.clear_parked(token[:40], reason="success")
                except Exception:
                    pass
                return result
        except Exception as exc:
            last_error = str(exc)
            err_lower = last_error.lower()
            # On 401/expired → skip this token, try next
            if any(x in err_lower for x in ("expired", "401")):
                continue
            # On usage limit → codext-style: park resume prompt, demote, try next account
            if any(x in err_lower for x in ("usage_limit", "quota", "capacity")):
                usage_limit_hits += 1
                # Park a recovery prompt for this account (codext-style)
                try:
                    if config.auto_switch_on_rate_limit:
                        from services.account_switch_resume import account_switch_resume
                        resume_prompt = config.usage_limit_resume_prompt
                        if resume_prompt is not None:
                            account_switch_resume.set_resume_prompt(resume_prompt)
                        account_switch_resume.park_task(
                            account_id=token[:40],
                            model=pure_model,
                            messages=messages,
                        )
                except Exception:
                    pass
                # Account is already demoted + marked limited in the provider,
                # so the next get_token_for_request() will pick the NEXT account.
                continue
            # On 400/429 → try next token
            if any(x in err_lower for x in ("400", "429", "rate")):
                continue
            break

    # Raise exception so combo fallback can try next provider
    raise RuntimeError(f"OpenAI OAuth error: {last_error}")


def _handle_gemini_chat(
    model: str,
    messages: list[dict[str, Any]],
    stream: bool,
    body: dict[str, Any],
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """Gemini AI Studio chat — native function calling support."""
    from services.providers.gemini_free import gemini_provider, GEMINI_DEFAULT_MODEL

    pure_model = model
    for prefix in ("gemini/", "gemini_free/"):
        if model.startswith(prefix):
            pure_model = model[len(prefix):]
            break
    if not pure_model or pure_model == "auto":
        # Try enabled_models order first, then provider config, then default
        ms = config.data.get("model_settings") or {}
        enabled = (ms.get("enabled_models") or {}).get("gemini_free") if isinstance(ms, dict) else None
        chosen = ""
        if isinstance(enabled, list):
            for m in enabled:
                m = str(m).strip()
                if not m or m == "auto":
                    continue
                for prefix in ("gemini/", "gemini_free/"):
                    if m.startswith(prefix):
                        m = m[len(prefix):]
                        break
                if m:
                    chosen = m
                    break
        if not chosen:
            provider_cfg = (config.data.get("providers") or {}).get("gemini_free") or {}
            chosen = str(provider_cfg.get("model") or "") or GEMINI_DEFAULT_MODEL
        pure_model = chosen

    logger.info({"event": "gemini_chat", "model": pure_model})

    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    try:
        # Gemini always streams via SSE API — iterator handles both cases
        result_iter = gemini_provider.chat_completions(
            messages=messages, model=pure_model,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice,
        )
        if stream:
            return result_iter
        else:
            # Collect stream into single response
            content = ""
            tc = []
            for chunk in result_iter:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content += delta.get("content", "")
                if delta.get("tool_calls"):
                    tc = delta["tool_calls"]
            msg = {"role": "assistant", "content": content}
            if tc:
                msg["tool_calls"] = tc
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": pure_model,
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
    except Exception as exc:
        logger.error({"event": "gemini_fatal", "error": str(exc)})
        return completion_response(model=model, content=f"Gemini error: {exc}", messages=messages)


def _handle_nvidia_chat(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    stream: bool,
    body: dict[str, Any],
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """NVIDIA NIM chat — OpenAI-compatible proxy, no format conversion needed."""
    from services.providers.nvidia_nim import nvidia_nim_provider

    pure_model = model
    if model.startswith("nv/"):
        pure_model = model[3:]

    logger.info({"event": "nvidia_nim_chat", "model": pure_model, "stream": stream})

    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")

    try:
        result = nvidia_nim_provider.chat_completions(
            messages=messages, model=pure_model, stream=stream,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice,
            top_p=body.get("top_p"),
            frequency_penalty=body.get("frequency_penalty"),
            presence_penalty=body.get("presence_penalty"),
        )
        if stream:
            return result
        else:
            return result
    except Exception as exc:
        logger.error({"event": "nvidia_nim_fatal", "error": str(exc)})
        return completion_response(
            model=model,
            content=f"NVIDIA NIM error: {exc}",
            messages=messages,
        )


def _handle_custom_openai_chat(
    provider_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    stream: bool,
    body: dict[str, Any],
    force_token: str = "",
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """Custom OpenAI-compatible provider — generic proxy.

    If force_token is provided, it overrides the provider's configured API key.
    """
    from services.providers.custom_openai import CustomOpenAIProvider, get_custom_providers

    # Extract provider ID from "custom:deepseek" format
    provider_id = provider_key[len("custom:"):]

    providers = get_custom_providers()
    cfg = dict(providers.get(provider_id) or {})
    if not cfg:
        return completion_response(
            model=model,
            content=f"Custom provider '{provider_id}' not found or disabled",
            messages=messages,
        )

    if force_token:
        cfg["api_key"] = force_token

    provider = CustomOpenAIProvider(cfg)

    logger.info({"event": "custom_openai_chat", "provider": provider.name, "model": model})

    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")

    try:
        result = provider.chat_completions(
            messages=messages, model=model, stream=stream,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice,
            top_p=body.get("top_p"),
            frequency_penalty=body.get("frequency_penalty"),
            presence_penalty=body.get("presence_penalty"),
        )
        if stream:
            return result
        else:
            return result
    except Exception as exc:
        logger.error({"event": "custom_openai_fatal", "provider": provider.name, "error": str(exc)})
        return completion_response(
            model=model,
            content=f"[{provider.name}] Error: {exc}",
            messages=messages,
        )


def _handle_antigravity_chat(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    stream: bool,
    body: dict[str, Any],
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """Use Antigravity rotated Google Cloud companion tokens for chat completions."""
    from services.providers.antigravity import antigravity_provider

    pure_model = model[3:] if model.startswith("ag/") else model
    if not pure_model or pure_model == "auto":
        ms = config.data.get("model_settings") or {}
        enabled = (ms.get("enabled_models") or {}).get("antigravity") if isinstance(ms, dict) else None
        chosen = ""
        if isinstance(enabled, list):
            for m in enabled:
                m = str(m).strip()
                if not m or m == "auto":
                    continue
                if m.startswith("ag/"):
                    m = m[3:]
                if m:
                    chosen = m
                    break
        pure_model = chosen or "gemini-3.1-pro-high"

    logger.info({
        "event": "antigravity_chat",
        "model": pure_model,
        "stream": stream,
    })

    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")

    attempted: set[str] = set()
    last_error = ""

    while True:
        try:
            account = antigravity_provider.get_token_for_request(attempted)
        except RuntimeError as exc:
            raise RuntimeError(str(exc))

        token = account.get("access_token", "")
        if not token or token in attempted:
            break
        attempted.add(token)

        try:
            if stream:
                return antigravity_provider.chat_completions(
                    account=account, messages=messages, model=pure_model,
                    stream=True, temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice=tool_choice,
                )
            else:
                result = antigravity_provider.chat_completions(
                    account=account, messages=messages, model=pure_model,
                    stream=False, temperature=temperature, max_tokens=max_tokens,
                    tools=tools, tool_choice=tool_choice,
                )
                account_service.mark_text_used(token)
                return result
        except Exception as exc:
            last_error = str(exc)
            # On 401/expired → skip this token, try next
            if any(x in last_error.lower() for x in ("expired", "401", "unauthorized")):
                continue
            # On 400/429/quota → try next
            if any(x in last_error.lower() for x in ("400", "429", "rate", "quota")):
                continue
            break

    raise RuntimeError(f"Antigravity error: {last_error}")


def _messages_size(messages: list[dict[str, Any]] | None) -> int:
    """Total character count across all message content — used to detect
    whether search_service or HA context injected anything new."""
    if not messages:
        return 0
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    total += len(str(part.get("text", "")))
    return total


def _messages_have_images(messages: list[dict[str, Any]] | None) -> bool:
    """True when any message carries an image_url / input_image part.

    Used to detect vision requests so we can skip MCP/HA tool injection — a
    "phân tích ảnh" task never needs Wikipedia / weather / device control,
    and the 60+ tool definitions just bloat the prompt + slow vision models
    that have to scan the tool list before answering.
    """
    for m in messages or []:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                return True
    return False


_STATUS_QUERY_KEYWORDS = [
    "trạng thái", "tình trạng", "liệt kê", "có những", "kiểm tra", "thế nào",
    "ra sao", "tổng quan", "như thế nào", "đang bật", "đang tắt", "bao nhiêu",
]
_CONTROL_VERBS = [
    "bật", "tắt", "mở", "đóng", "đặt", "chỉnh", "tăng", "giảm", "kích hoạt",
    "khởi động", "dừng", "khoá", "khóa", "mở khoá", "mở khóa", "set ",
]


def _is_status_only_query(text: str) -> bool:
    """True for a pure status/listing question with NO control verb. Such a
    query is answered entirely from the prefetched live context, so we can ship
    ZERO tools — HA otherwise attaches ~40 control tools whose schemas bloat the
    free-account payload past chatgpt.com's limit (→ 413 → generic reply)."""
    t = (text or "").lower()
    if not any(k in t for k in _STATUS_QUERY_KEYWORDS):
        return False
    if any(v in t for v in _CONTROL_VERBS):
        return False
    return True


def _inject_mcp_tools(
    tools: list[dict[str, Any]] | None,
    skip_ha_search: bool = False,
    is_vision: bool = False,
    search_injected: bool = False,
    user_text: str = "",
    is_free_model: bool = False,
) -> list[dict[str, Any]] | None:
    """Inject tools from enabled MCP servers + HA into the tools list."""
    logger.info({"event": "mcp_inject_start", "input_tools": len(tools or [])})
    try:
        # Vision request — skip all injection. Return the caller's tools as-is.
        if is_vision:
            logger.info({"event": "mcp_inject_skipped", "reason": "vision_request"})
            return tools if tools else None

        # Pure status/listing query whose answer is already in the prefetched
        # live context → ship NO tools. Drops HA's ~40 control-tool schemas
        # (the dominant payload bloat that 413s the free backend and makes the
        # model reply "what do you want me to do?"). Control queries keep tools.
        if (skip_ha_search or is_free_model) and _is_status_only_query(user_text):
            logger.info({"event": "mcp_inject_skipped", "reason": "status_only_query_free_or_ha"})
            return None

        if is_free_model:
            logger.info({"event": "mcp_inject_skipped", "reason": "free_model_no_agentic_loop"})
            return tools if tools else None

        # Search results already injected. We used to skip tool injection here
        # to save prompt space, but users want to see explicit tool calls
        # (e.g. for weather) or fallback to them if search timed out.
        if search_injected:
            logger.info({"event": "mcp_inject_proceeding", "reason": "search_injected_but_tools_requested"})
            # Do NOT return early, let the tools be injected so the LLM can explicitly call them if needed.

        from services.mcp_client import get_enabled_mcp_tools
        from services.ha_client import get_ha_tools

        # Skip the MCP discovery + injection when the prompt already carries the
        # HA registry — those tools won't be useful here and the LLM may waste
        # a round-trip calling one.
        if skip_ha_search:
            mcp_tools = []
            logger.info({"event": "mcp_inject_skipped", "reason": "ha_context_injected"})
        elif not tools and _is_trivial_chat(user_text):
            # Trivial greeting/chat with no explicit tools requested — skip
            # all 43 MCP tools. The payload would exceed ChatGPT's per-account
            # size limit and cause 413 errors. Keep HA tools for smart home.
            mcp_tools = []
            logger.info({"event": "mcp_inject_skipped", "reason": "trivial_chat"})
        else:
            mcp_tools = get_enabled_mcp_tools()
            logger.info({"event": "mcp_inject_got_tools", "count": len(mcp_tools)})

        tools = list(tools or [])
        existing_names = {t.get("function", {}).get("name", "") for t in tools}

        client_is_ha = any(name.startswith("Hass") or name == "GetLiveContext" for name in existing_names)
        # HA clients bring their own control tools (HassTurnOn, etc.) but
        # we keep read-only query tools so the LLM can call GetLiveContext
        # to fetch live device state, matching how Gemini pipeline works.
        if client_is_ha:
            ha_tools = get_ha_tools()
            _keep_readonly = {"GetLiveContext", "ha_search_entities", "ha_get_state"}
            ha_tools = [t for t in ha_tools if t.get("function", {}).get("name", "") in _keep_readonly]
        else:
            ha_tools = get_ha_tools()

        # Keep all HA tools. We used to drop read-only tools here assuming the
        # context was prefetched, but only ChatGPT Free actually prefetches.
        # Gemini needs these tools to dynamically query states.
        if skip_ha_search:
            logger.info({"event": "ha_read_tools_kept", "reason": "model_needs_tools"})
        all_new_tools = mcp_tools + ha_tools
        if not all_new_tools:
            return tools if tools else None

        for mt in all_new_tools:
            if mt.get("function", {}).get("name", "") not in existing_names:
                tools.append(mt)
        logger.info({"event": "mcp_tools_injected", "mcp_count": len(mcp_tools),
                     "ha_count": len(ha_tools), "total_tools": len(tools)})
        return tools
    except Exception as exc:
        logger.warning({"event": "mcp_tools_inject_failed", "error": str(exc)})
        return tools


def _execute_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Execute an MCP or HA tool call and return the result text."""
    # Try MCP first
    try:
        from services.mcp_client import call_mcp_tool
        result = call_mcp_tool(tool_name, arguments)
        if result is not None:
            return result
    except Exception as exc:
        logger.warning({"event": "mcp_tool_call_failed", "tool": tool_name, "error": str(exc)})
    # Try HA tools
    try:
        from services.ha_client import execute_ha_tool
        result = execute_ha_tool(tool_name, arguments)
        if result is not None:
            return result
    except Exception as exc:
        logger.warning({"event": "ha_tool_call_failed", "tool": tool_name, "error": str(exc)})
    return None
