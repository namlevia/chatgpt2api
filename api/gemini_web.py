"""Gemini web-cookie provider — gemini.google.com qua HTTP API (gemini_webapi).

Đường thứ 3 cho Gemini, song song với:
  - gemini_free (gemini/): AI Studio API key, generativelanguage.googleapis.com
  - gemini_web  (gmw/):    DOM scrape qua captcha-solver browser (chậm)

Path này nói chuyện THẲNG với backend gemini.google.com bằng cookie Google
(`__Secure-1PSID` + `__Secure-1PSIDTS`) — pattern y hệt Claude free sessionKey
(tham khảo https://github.com/luuquangvu/Gemini-FastAPI, lib HanaokaYuzu/Gemini-API).

Cookie lấy theo thứ tự:
  1. config providers.gemini_web_api.psid / psidts (dán tay)
  2. captcha-solver GET /v1/gemini-web/{profile}/cookies (reuse Google profile
     đã onboard — như Claude fetch sessionKey), cache 5'.

Model prefix: gma/ (vd gma/auto, gma/gemini-3-flash). Hỗ trợ vision (files),
downscale 896 qua knob gemini_vision_max_dim (0 = tắt).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Iterator

from curl_cffi import requests


def _config():
    from services.config import config
    return config


def _logger():
    from utils.log import logger
    return logger


def _cfg() -> dict[str, Any]:
    return (_config().data.get("providers") or {}).get("gemini_web_api") or {}


# ── Cookie source ────────────────────────────────────────────────────────────

_cookie_cache: dict[str, tuple[float, dict[str, str]]] = {}
_COOKIE_TTL = 300  # captcha-solver fetch cache


def _solver_cfg() -> dict[str, str]:
    """captcha-solver url/key: own config → gemini_web → flow (same solver)."""
    providers = _config().data.get("providers") or {}
    for name in ("gemini_web_api", "gemini_web", "flow"):
        c = providers.get(name) or {}
        url = str(c.get("captcha_solver_url") or "").rstrip("/")
        if url:
            return {"url": url, "api_key": str(c.get("captcha_solver_api_key") or "")}
    return {"url": "", "api_key": ""}


def _profiles() -> list[str]:
    cfg = _cfg()
    profiles: list[str] = []
    
    for entry in (cfg.get("accounts") or []):
        if isinstance(entry, dict):
            p = str(entry.get("profile") or "").strip()
            if p and p not in profiles:
                profiles.append(p)
                
    profs = cfg.get("profiles")
    if isinstance(profs, list):
        for p in profs:
            p = str(p).strip()
            if p and p not in profiles:
                profiles.append(p)
                
    if not profiles:
        # fallback: dùng chính profile của gemini_web DOM scrape (đã login sẵn)
        gw = (_config().data.get("providers") or {}).get("gemini_web") or {}
        gw_accs = gw.get("accounts") if isinstance(gw.get("accounts"), list) else []
        for a in gw_accs:
            if isinstance(a, dict):
                p = str(a.get("profile") or "").strip()
                if p and p not in profiles:
                    profiles.append(p)
        p = str(gw.get("profile") or "").strip()
        if p and p not in profiles:
            profiles.append(p)
            
    return profiles or ["gemini-web-default"]


def _fetch_cookies_from_solver(profile: str) -> dict[str, str]:
    now = time.time()
    hit = _cookie_cache.get(profile)
    if hit and (now - hit[0]) < _COOKIE_TTL:
        return hit[1]
    sc = _solver_cfg()
    if not sc["url"]:
        return {}
    try:
        headers = {"Authorization": f"Bearer {sc['api_key']}"} if sc["api_key"] else {}
        r = requests.get(f"{sc['url']}/v1/gemini-web/{profile}/cookies",
                         headers=headers, timeout=30, impersonate="chrome110")
        if r.status_code == 200:
            cookies = (r.json() or {}).get("cookies") or {}
            if cookies.get("__Secure-1PSID"):
                _cookie_cache[profile] = (now, cookies)
                return cookies
        _logger().warning({"event": "gma_cookie_fetch_failed", "profile": profile,
                           "status": r.status_code, "body": r.text[:120]})
    except Exception as exc:
        _logger().warning({"event": "gma_cookie_fetch_error", "profile": profile,
                           "error": str(exc)[:120]})
    return {}


def _get_cookies_ranked(required_features: list[str] = None) -> list[tuple[str, str, str]]:
    """Return a list of (psid, psidts, profile) ranked by health/quota.
    Falls back to single psid config if present."""
    cfg = _cfg()
    psid = str(cfg.get("psid") or "").strip()
    if psid:
        return [(psid, str(cfg.get("psidts") or "").strip(), "static-config")]
        
    from services.account_service import account_service
    profiles = _profiles()
    raw_accounts = [{"profile": p, "status": "active"} for p in profiles]
    
    ranked = account_service.normalize_and_rank_accounts(
        raw_accounts,
        account_type="gemini_web_api",
        required_features=required_features or ["text"],
    )
    
    results = []
    for acc in ranked:
        profile = acc.get("profile")
        if not profile: continue
        c = _fetch_cookies_from_solver(profile)
        if c.get("__Secure-1PSID"):
            results.append((c["__Secure-1PSID"], c.get("__Secure-1PSIDTS", ""), profile))
            
    return results

def is_available() -> bool:
    return len(_get_cookies_ranked()) > 0


# ── Dedicated asyncio loop (gemini_webapi là async-only) ────────────────────

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            t = threading.Thread(target=_loop.run_forever, daemon=True,
                                 name="gemini-webapi-loop")
            t.start()
        return _loop


def _run(coro, timeout: float = 240):
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result(timeout)


# ── Client cache ─────────────────────────────────────────────────────────────

_clients: dict[str, Any] = {}
_client_lock = threading.Lock()


def _get_client(psid: str, psidts: str):
    key = psid[:32]
    with _client_lock:
        cli = _clients.get(key)
        if cli is not None:
            return cli
        from gemini_webapi import GeminiClient
        # gemini_webapi 2.x uses lowercase constructor params.
        cli = GeminiClient(secure_1psid=psid, secure_1psidts=psidts or None)
        _run(cli.init(timeout=30, auto_close=False, auto_refresh=True), timeout=60)
        _clients[key] = cli
        _logger().info({"event": "gma_client_init", "psid_prefix": psid[:12]})
        return cli


def _drop_client(psid: str) -> None:
    with _client_lock:
        _clients.pop(psid[:32], None)
    _cookie_cache.clear()


# ── Model & message helpers ──────────────────────────────────────────────────

# Tên thân thiện theo UI Gemini (3.5 Flash / 3.1 Pro + Tiêu chuẩn/Mở rộng) →
# model_name nội bộ của gemini_webapi. "Mở rộng" = tier advanced (tư duy sâu).
# Lib KHÔNG có model "Flash-Lite" riêng → map về flash. Tên lib gốc vẫn route OK.
_GMA_ALIASES = {
    # Tên khớp UI Gemini (không dấu cho an toàn client) — bộ hiển thị chính
    "3.5-flash": "gemini-3-flash",                  # 3.5 Flash (Tiêu chuẩn)
    "3.5-flash-mo-rong": "gemini-3-flash-advanced", # 3.5 Flash (Mở rộng)
    "3.1-pro": "gemini-3-pro",                      # 3.1 Pro (Tiêu chuẩn)
    "3.1-pro-mo-rong": "gemini-3-pro-advanced",     # 3.1 Pro (Mở rộng)
    "3.1-flash-lite": "gemini-3-flash",             # Flash-Lite (lib chưa tách → flash)
    # Alias cũ — vẫn nhận để không vỡ request đã cấu hình
    "flash": "gemini-3-flash",
    "flash-lite": "gemini-3-flash",
    "flash-thinking": "gemini-3-flash-thinking",
    "flash-extended": "gemini-3-flash-advanced",
    "pro": "gemini-3-pro",
    "pro-extended": "gemini-3-pro-advanced",
}


def _resolve_model(model: str):
    """alias → gemini_webapi Model enum; None = để server tự chọn."""
    m = str(model or "").strip().lower()
    for pfx in ("gma/", "gemini-web/", "gemini_web_api/"):
        if m.startswith(pfx):
            m = m[len(pfx):]
            break
    if not m or m == "auto":
        try:
            from services.config import config as _config
            ms = _config.data.get("model_settings") or {}
            
            # 1. Check explicit default_model
            default_model = (ms.get("default_models") or {}).get("gemini_web_api")
            if default_model:
                m = str(default_model).strip()
                if m.startswith("gma/"):
                    m = m[4:]
                    
            # 2. Fallback to first enabled model
            if not m or m == "auto":
                enabled = (ms.get("enabled_models") or {}).get("gemini_web_api")
                if isinstance(enabled, list):
                    for em in enabled:
                        em = str(em).strip()
                        if em.startswith("gma/"):
                            em = em[4:]
                        if em and em != "auto":
                            m = em
                            break
        except Exception:
            pass

    if not m or m == "auto":
        m = str(_cfg().get("model") or "").strip().lower()
    if not m or m == "auto":
        return None
    # UI-friendly alias → lib model_name. The lib labels models "gemini-3-*"
    # while the Gemini UI shows "3.5 Flash / 3.1 Pro" + thinking Tiêu chuẩn(基)
    # /Mở rộng(advanced). Map both so a HA/app pick matches what the user sees.
    m = _GMA_ALIASES.get(m, m)
    try:
        from gemini_webapi.constants import Model
        return Model.from_name(m)
    except Exception:
        _logger().info({"event": "gma_unknown_model_fallback", "model": m})
        return None


def _downscale(data: bytes, mime: str) -> tuple[bytes, str]:
    try:
        max_dim = int(_config().data.get("gemini_vision_max_dim", 896) or 0)
    except Exception:
        max_dim = 896
    if not max_dim:
        return data, mime
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(data))
        w, h = img.size
        if max(w, h) <= max_dim:
            return data, mime
        scale = max_dim / float(max(w, h))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        resized = img.convert("RGB") if img.mode not in ("RGB", "L") else img
        resized = resized.resize(size, Image.LANCZOS)
        buf = BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        out = buf.getvalue()
        _logger().info({"event": "gma_image_downscaled", "from": [w, h],
                        "to": list(size), "bytes": [len(data), len(out)]})
        return out, "image/jpeg"
    except Exception:
        return data, mime


def _prepare_files(messages: list[dict[str, Any]]) -> list[str]:
    """image_url parts → temp files (gemini_webapi nhận path). Caller xoá sau."""
    import base64
    paths: list[str] = []
    for msg in messages or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for p in content:
            if not isinstance(p, dict) or p.get("type") != "image_url":
                continue
            url = str(((p.get("image_url") or {}).get("url") or "")).strip()
            data, mime = b"", "image/png"
            if url.startswith("data:"):
                try:
                    head, b64 = url.split(",", 1)
                    mime = (head[5:].split(";")[0] or "image/png").lower()
                    data = base64.b64decode(b64)
                except Exception:
                    continue
            elif url.startswith("http"):
                try:
                    rr = requests.get(url, timeout=20, impersonate="chrome110")
                    if rr.status_code == 200 and rr.content:
                        mime = (rr.headers.get("content-type") or "image/png").split(";")[0].lower()
                        data = rr.content
                except Exception:
                    continue
            if not data:
                continue
            data, mime = _downscale(data, mime)
            ext = {"image/jpeg": ".jpg", "image/png": ".png",
                   "image/webp": ".webp", "image/gif": ".gif"}.get(mime, ".png")
            fd, path = tempfile.mkstemp(suffix=ext, prefix="gma_")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            paths.append(path)
    return paths


def _cleanup(paths: list[str]) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except Exception:
            pass


def _openai_chunk(model: str, cid: str, created: int,
                  delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


# ── Handler (main /v1/chat/completions router) ──────────────────────────────

def _generate_text(client, prompt: str, files: list[str], model_enum) -> str:
    kwargs: dict[str, Any] = {}
    if files:
        kwargs["files"] = files
    if model_enum is not None:
        kwargs["model"] = model_enum
    resp = _run(client.generate_content(prompt, **kwargs))
    return str(getattr(resp, "text", "") or "")


def handle_gemini_web_api_chat(
    model: str,
    messages: list[dict[str, Any]],
    stream: Any,
    body: dict[str, Any] | None = None,
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """Provider handler cho router chính (gma/* models)."""
    from api.claude import _flatten_messages  # cùng pattern stateless prompt
    from services.account_service import account_service

    prompt = _flatten_messages(messages)
    files = _prepare_files(messages)
    model_enum = _resolve_model(model)
    if files:
        _logger().info({"event": "gma_images", "count": len(files)})
    _logger().info({"event": "gma_request", "model": str(model_enum or "auto"),
                    "msg_count": len(messages or [])})

    req_features = ["file_upload"] if files else ["text"]
    available_creds = _get_cookies_ranked(required_features=req_features)
    if not available_creds:
        _cleanup(files)
        raise RuntimeError(
            "Gemini web-api not configured or all accounts exhausted: set providers.gemini_web_api.psid "
            "(cookie __Secure-1PSID) or onboard a gemini_web profile")

    def _call_with_retry() -> str:
        last_exc = None
        for psid, psidts, profile in available_creds:
            try:
                client = _get_client(psid, psidts)
                text = _generate_text(client, prompt, files, model_enum)
                
                # Detect quota limits in text response
                lower_text = str(text).lower()
                if any(k in lower_text for k in ("reached your limit", "giới hạn", "usage cap", "hết lượt")):
                    raise RuntimeError(f"QUOTA_EXHAUSTED: {text[:100]}")
                    
                return text
            except Exception as exc:
                err = str(exc).lower()
                
                # Quota exhaustion
                if "quota_exhausted" in err:
                    _logger().warning({"event": "gma_quota_hit", "profile": profile})
                    if profile and profile != "static-config":
                        account_service.record_profile_quota_failure(
                            profile=profile,
                            quota_type="file_upload" if files else "text_limit",
                            account_type="gemini_web_api"
                        )
                    last_exc = exc
                    continue
                    
                # Auth/Cookie invalidation
                if any(k in err for k in ("auth", "cookie", "1psid", "401", "403")):
                    _logger().warning({"event": "gma_auth_retry", "error": str(exc)[:120]})
                    _drop_client(psid)
                    last_exc = exc
                    continue
                    
                raise exc
        
        # If we loop through all credentials and fail
        if last_exc:
            raise last_exc
        raise RuntimeError("No available accounts to fulfill request")

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if stream:
        def sse() -> Iterator[dict[str, Any]]:
            try:
                text = _call_with_retry()
                yield _openai_chunk(model, cid, created, {"role": "assistant", "content": ""})
                yield _openai_chunk(model, cid, created, {"content": text})
                yield _openai_chunk(model, cid, created, {}, finish="stop")
            finally:
                _cleanup(files)
        return sse()

    try:
        text = _call_with_retry()
    finally:
        _cleanup(files)
    return {
        "id": cid, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
