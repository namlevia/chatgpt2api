"""
MemoryService — lớp ký ức dài hạn qua OpenMemory self-host (CaviraOSS/OpenMemory).

Mục đích: giữ "dòng suy nghĩ" của các phiên code khi xoay account / mở phiên
mới — trước khi dispatch, recall top-K ký ức liên quan và inject làm system
message; sau khi trả lời xong, lưu cặp user/assistant vào OpenMemory (thread
nền, không chặn response).

An toàn flow: MẶC ĐỊNH TẮT. Chỉ chạy khi config providers.openmemory.enabled
= true và có base_url. Mọi lỗi (OpenMemory chết, timeout...) đều nuốt — chat
hoạt động y như khi không có memory. Không áp dụng cho HA query, vision và
image-generation.

Config (providers.openmemory):
    enabled: bool (default false)
    base_url: "http://172.16.10.38:8080" (OpenMemory backend)
    api_key: OM_API_KEY (header x-api-key)
    user_id: default "chatgpt2api" (override per-request bằng field chuẩn
             OpenAI `user` trong body)
    k: số ký ức recall (default 5)
    min_score: ngưỡng relevance (default 0.4)
    recall_timeout: giây (default 2.0 — không để memory làm chậm chat)
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any, Iterator

from services.config import config
from utils.helper import IMAGE_MODELS
from utils.log import logger

_INJECT_HEADER = (
    "Ký ức liên quan từ các phiên làm việc trước (long-term memory — dùng để "
    "tiếp nối mạch công việc, KHÔNG cần nhắc lại với người dùng):"
)

_STORE_USER_MAX = 2000
_STORE_ASSISTANT_MAX = 3000
_INJECT_TOTAL_MAX = 2400
_MIN_QUERY_CHARS = 8


def _messages_have_images(messages: list[dict[str, Any]] | None) -> bool:
    for msg in messages or []:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image", "image_url", "input_image"):
                    return True
    return False


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return "\n".join(parts).strip()
    return ""


class _MemoryCapture:
    """Giữ context 1 lượt chat để lưu ký ức sau khi có response."""

    def __init__(self, service: "MemoryService", user_id: str, user_text: str, model: str):
        self.service = service
        self.user_id = user_id
        self.user_text = user_text
        self.model = model

    def _store_turn(self, assistant_text: str) -> None:
        assistant_text = (assistant_text or "").strip()
        if not assistant_text:
            return
        content = (
            f"USER: {self.user_text[:_STORE_USER_MAX]}\n"
            f"ASSISTANT: {assistant_text[:_STORE_ASSISTANT_MAX]}"
        )
        self.service.store_async(content, self.user_id, tags=["chat", self.model], metadata={"model": self.model})

    def capture(self, result: Any) -> Any:
        # Non-stream: đọc content từ dict, trả nguyên vẹn
        if isinstance(result, dict):
            try:
                choices = result.get("choices") or []
                msg = choices[0].get("message") or {}
                self._store_turn(str(msg.get("content") or ""))
            except Exception:
                pass
            return result
        # Stream: tee — yield nguyên chunk, gom content, lưu khi stream xong
        return self._tee(result)

    def _tee(self, gen: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        parts: list[str] = []
        try:
            for chunk in gen:
                try:
                    choices = chunk.get("choices") or []
                    first = choices[0] if choices and isinstance(choices[0], dict) else {}
                    delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
                    text = str(delta.get("content") or "")
                    if text:
                        parts.append(text)
                except Exception:
                    pass
                yield chunk
        finally:
            try:
                self._store_turn("".join(parts))
            except Exception:
                pass


class MemoryService:
    @property
    def _cfg(self) -> dict[str, Any]:
        cfg = (config.data.get("providers") or {}).get("openmemory")
        return cfg if isinstance(cfg, dict) else {}

    @property
    def is_enabled(self) -> bool:
        cfg = self._cfg
        return bool(cfg.get("enabled")) and bool(str(cfg.get("base_url") or "").strip())

    # ------------------------------------------------------------------ HTTP
    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        cfg = self._cfg
        base = str(cfg.get("base_url") or "").rstrip("/")
        headers = {"Content-Type": "application/json"}
        api_key = str(cfg.get("api_key") or "").strip()
        if api_key:
            headers["x-api-key"] = api_key
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def recall(self, query: str, user_id: str) -> list[dict[str, Any]]:
        cfg = self._cfg
        try:
            data = self._post(
                "/api/memory/query",
                {
                    "query": query[:8000],
                    "k": int(cfg.get("k") or 5),
                    "user_id": user_id,
                },
                timeout=float(cfg.get("recall_timeout") or 2.0),
            )
            matches = data.get("matches") or []
            min_score = float(cfg.get("min_score") or 0.4)
            return [m for m in matches if isinstance(m, dict) and float(m.get("score") or 0) >= min_score]
        except Exception as exc:
            logger.warning({"event": "memory_recall_fail", "error": str(exc)[:200]})
            return []

    def store_async(self, content: str, user_id: str, tags: list[str] | None = None, metadata: dict[str, Any] | None = None) -> None:
        def _worker():
            try:
                self._post(
                    "/api/memory/add",
                    {
                        "content": content,
                        "tags": [t for t in (tags or []) if t][:8],
                        "metadata": metadata or {},
                        "user_id": user_id,
                    },
                    timeout=10.0,
                )
            except Exception as exc:
                logger.warning({"event": "memory_store_fail", "error": str(exc)[:200]})

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------ chat hooks
    def prepare(self, body: dict[str, Any]) -> _MemoryCapture | None:
        """Recall + inject ký ức vào body['messages']. Trả capture-context để
        lưu lượt chat sau khi có response; None nếu memory không áp dụng."""
        if not self.is_enabled:
            return None
        if bool(body.get("_is_ha_request")):
            return None
        model = str(body.get("model") or "").strip()
        if model in IMAGE_MODELS:
            return None
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return None
        if _messages_have_images(messages):
            return None
        user_text = _last_user_text(messages)
        if len(user_text) < _MIN_QUERY_CHARS:
            return None
        try:
            from services.ha_client import is_ha_query
            if is_ha_query(messages):
                return None
        except Exception:
            pass

        cfg = self._cfg
        user_id = str(body.get("user") or cfg.get("user_id") or "chatgpt2api").strip()

        matches = self.recall(user_text, user_id)
        if matches:
            lines: list[str] = []
            total = 0
            for m in matches:
                line = "- " + str(m.get("content") or "").strip().replace("\n", " ")[:600]
                if total + len(line) > _INJECT_TOTAL_MAX:
                    break
                lines.append(line)
                total += len(line)
            if lines:
                messages.append({"role": "system", "content": _INJECT_HEADER + "\n" + "\n".join(lines)})
                logger.info({"event": "memory_injected", "count": len(lines), "user_id": user_id})

        return _MemoryCapture(self, user_id, user_text, model)


# Singleton
memory_service = MemoryService()
