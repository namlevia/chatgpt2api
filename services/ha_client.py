"""Home Assistant REST API client via Long-Lived Access Token.

Fetches entity states and calls services so the LLM can see and control
the smart home directly, without needing the HA voice pipeline.
"""

from __future__ import annotations

import json, logging, time, threading
from typing import Any
import urllib.request

from services.config import config
from utils.log import logger

# ── Module-level state cache ────────────────────────────────────────────────
_state_cache: list[dict] = []
_state_cache_ts: float = 0.0
_context_cache: str = ""
_context_cache_ts: float = 0.0
_state_cache_lock = threading.Lock()
_DEFAULT_TTL = 60  # 60s — short enough that registry state stays fresh
                   # for "trạng thái đèn X?" answers, long enough that 985
                   # entities don't cost more than ~1 HA call per minute.
_scheduler_started = False

# Entity_ids HA exposes to Assist (voice). Refreshed lazily / by the scheduler.
_exposed_cache: set[str] = set()
_exposed_cache_ts: float = 0.0
_EXPOSED_TTL = 600  # exposure config changes rarely → refresh every 10 min


def _get_ha_settings() -> dict:
    """Get HA settings: url, token, refresh_interval, refresh_times."""
    try:
        return config.data.get("home_assistant") or {}
    except Exception:
        return {}


def _get_cache_ttl() -> int:
    """Get refresh interval from HA settings, default 3600s."""
    try:
        return int(_get_ha_settings().get("refresh_interval", 3600))
    except Exception:
        return 3600


def _get_services() -> dict[str, list[str]]:
    """Fetch available services from HA API (real data, not hardcoded)."""
    cfg = _get_ha_config()
    if not cfg:
        return {}
    try:
        req = urllib.request.Request(
            f"{cfg['url']}/api/services",
            headers={"Authorization": f"Bearer {cfg['token']}", "Content-Type": "application/json"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        services: dict[str, list[str]] = {}
        for item in data:
            domain = item.get("domain", "")
            svcs = item.get("services", {})
            names = sorted(svcs.keys()) if isinstance(svcs, dict) else []
            if names:
                services[domain] = names
        return services
    except Exception as exc:
        logger.warning({"event": "ha_services_failed", "error": str(exc)})
        return {}


def _get_refresh_times() -> list[str]:
    """Get scheduled refresh times (e.g., ['00:30', '06:00'])."""
    try:
        times = _get_ha_settings().get("refresh_times", [])
        return times if isinstance(times, list) else []
    except Exception:
        return []


def _get_ha_config() -> dict[str, str] | None:
    ha = config.data.get("home_assistant") or {}
    url = str(ha.get("url") or "").strip().rstrip("/")
    token = str(ha.get("token") or "").strip()
    if not url or not token:
        return None
    return {"url": url, "token": token}


def get_states(use_cache: bool = True) -> list[dict[str, Any]]:
    """Fetch all entity states from HA. Cache respects configurable TTL."""
    global _state_cache, _state_cache_ts
    ttl = _get_cache_ttl()
    now = time.time()
    if use_cache and _state_cache and (now - _state_cache_ts) < ttl:
        return _state_cache
    cfg = _get_ha_config()
    if not cfg:
        return []
    try:
        req = urllib.request.Request(
            f"{cfg['url']}/api/states",
            headers={"Authorization": f"Bearer {cfg['token']}", "Content-Type": "application/json"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        with _state_cache_lock:
            _state_cache = data
            _state_cache_ts = now
        return data
    except Exception as exc:
        logger.warning({"event": "ha_states_failed", "error": str(exc)})
        return _state_cache or []  # return stale cache on error


# ── Exposed-entity (Assist) list ────────────────────────────────────────────
# HA exposes only a curated subset of entities to the voice assistant
# ("Settings → Voice assistants → Expose"). For a general "trạng thái nhà"
# query we report exactly that set (≈116) instead of all ~989 entities. The
# list is only available over the WebSocket API; we speak raw WS with the
# stdlib (no extra deps) and cache the result.
def _ws_fetch_exposed(url: str, token: str) -> set[str]:
    import socket, base64, os, struct

    netloc = url.split("//", 1)[-1].split("/")[0]
    host = netloc.split(":")[0]
    port = int(netloc.rsplit(":", 1)[1]) if ":" in netloc else 8123
    key = base64.b64encode(os.urandom(16)).decode()
    s = socket.create_connection((host, port), timeout=8)
    s.settimeout(8)
    try:
        s.sendall((
            f"GET /api/websocket HTTP/1.1\r\nHost: {host}:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        ).encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = s.recv(4096)
            if not chunk:
                raise RuntimeError("ws handshake closed")
            resp += chunk

        buf = bytearray()

        def _need(n: int) -> None:
            while len(buf) < n:
                c = s.recv(8192)
                if not c:
                    raise RuntimeError("ws closed")
                buf.extend(c)

        def _recv() -> str:
            _need(2)
            ln = buf[1] & 0x7F
            idx = 2
            if ln == 126:
                _need(4); ln = struct.unpack(">H", bytes(buf[2:4]))[0]; idx = 4
            elif ln == 127:
                _need(10); ln = struct.unpack(">Q", bytes(buf[2:10]))[0]; idx = 10
            _need(idx + ln)
            p = bytes(buf[idx:idx + ln])
            del buf[:idx + ln]
            return p.decode("utf-8", "replace")

        def _send(obj: dict) -> None:
            d = json.dumps(obj).encode()
            m = os.urandom(4)
            h = bytearray([0x81])
            n = len(d)
            if n < 126:
                h.append(0x80 | n)
            elif n < 65536:
                h.append(0x80 | 126); h += struct.pack(">H", n)
            else:
                h.append(0x80 | 127); h += struct.pack(">Q", n)
            h += m
            s.sendall(bytes(h) + bytes(b ^ m[i % 4] for i, b in enumerate(d)))

        _recv()  # auth_required
        _send({"type": "auth", "access_token": token})
        _recv()  # auth_ok / auth_invalid
        _send({"id": 1, "type": "homeassistant/expose_entity/list"})
        exposed: set[str] = set()
        for _ in range(8):
            msg = json.loads(_recv())
            if msg.get("id") == 1 and msg.get("type") == "result":
                ex = (msg.get("result") or {}).get("exposed_entities") or {}
                exposed = {
                    eid for eid, amap in ex.items()
                    if isinstance(amap, dict) and any(amap.values())
                }
                break
        return exposed
    finally:
        try:
            s.close()
        except Exception:
            pass


def get_exposed_entity_ids(use_cache: bool = True) -> set[str]:
    """Entity_ids HA exposes to Assist. Cached; returns empty set on failure so
    callers can treat 'empty' as 'no filter' (never a regression)."""
    global _exposed_cache, _exposed_cache_ts
    now = time.time()
    if use_cache and _exposed_cache and (now - _exposed_cache_ts) < _EXPOSED_TTL:
        return _exposed_cache
    cfg = _get_ha_config()
    if not cfg:
        return _exposed_cache
    try:
        ids = _ws_fetch_exposed(cfg["url"], cfg["token"])
        if ids:
            _exposed_cache = ids
            _exposed_cache_ts = now
            logger.info({"event": "ha_exposed_refreshed", "count": len(ids)})
        return _exposed_cache
    except Exception as exc:
        logger.warning({"event": "ha_exposed_failed", "error": str(exc)[:120]})
        return _exposed_cache  # stale or empty → caller skips filter


def get_state(entity_id: str) -> dict[str, Any] | None:
    """Fetch a single entity's state."""
    cfg = _get_ha_config()
    if not cfg:
        return None
    try:
        req = urllib.request.Request(
            f"{cfg['url']}/api/states/{entity_id}",
            headers={"Authorization": f"Bearer {cfg['token']}", "Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req, timeout=10).read())
    except Exception as exc:
        logger.debug({"event": "ha_state_failed", "entity": entity_id, "error": str(exc)})
        return None


def call_service(domain: str, service: str, data: dict[str, Any] | None = None) -> bool:
    """Call an HA service (e.g., light.turn_on). Passes full data dict as payload."""
    cfg = _get_ha_config()
    if not cfg:
        return False
    try:
        payload = data or {}
        body = json.dumps(payload)
        req = urllib.request.Request(
            f"{cfg['url']}/api/services/{domain}/{service}",
            data=body.encode(),
            headers={"Authorization": f"Bearer {cfg['token']}", "Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        # Invalidate cache after any write operation
        global _state_cache_ts
        _state_cache_ts = 0.0
        return True
    except Exception as exc:
        logger.warning({"event": "ha_service_failed", "domain": domain, "service": service, "error": str(exc)})
        return False


# All domains shown to AI — users interact by friendly_name, not entity_id
# Limit per domain keeps token count reasonable even for large setups
_CONTEXT_DOMAINS = [
    "light", "switch", "climate", "cover", "lock", "fan", "media_player",
    "sensor", "binary_sensor", "input_boolean", "input_number", "input_select",
    "scene", "script", "automation", "vacuum", "camera", "weather",
]
# Max entities per domain shown in context (keep token count low, but user requested all devices)
_MAX_PER_DOMAIN = 9999


def format_states_context() -> str:
    """Return cached device registry. NEVER blocks on HA API call.

    The registry is refreshed by a background thread on schedule.
    Chat requests always get instant cached data (no latency added).
    """
    global _context_cache
    _ensure_scheduler_running()
    if _context_cache:
        return _context_cache
    # First call: build cache synchronously (cold start only)
    _refresh_context()
    return _context_cache


def _refresh_context() -> None:
    """Background: fetch states and rebuild context string."""
    global _context_cache, _context_cache_ts
    try:
        states = get_states(use_cache=False)
        if not states:
            return
        _context_cache = _build_context(states)
        _context_cache_ts = time.time()
        logger.info({"event": "ha_context_refreshed", "devices": len(states)})
        try:
            get_exposed_entity_ids(use_cache=False)  # keep Assist-exposed set warm
        except Exception:
            pass
    except Exception as exc:
        logger.warning({"event": "ha_context_refresh_failed", "error": str(exc)})


def _ensure_scheduler_running() -> None:
    """Start background refresh scheduler (idempotent)."""
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True
    # Initial fetch immediately
    try:
        _refresh_context()
    except Exception:
        pass
    t = threading.Thread(target=_scheduler_loop, daemon=True, name="ha-scheduler")
    t.start()
    logger.info({"event": "ha_scheduler_started"})


def _scheduler_loop() -> None:
    """Background loop: refresh at scheduled times or after interval."""
    from datetime import datetime
    while True:
        time.sleep(30)  # Check every 30s
        try:
            ttl = _get_cache_ttl()
            refresh_times = _get_refresh_times()
            now = time.time()
            # Refresh if TTL expired
            if (now - _context_cache_ts) >= ttl:
                _refresh_context()
                continue
            # Refresh if scheduled time just passed
            if refresh_times:
                current_time = datetime.now().strftime("%H:%M")
                last_ts = datetime.fromtimestamp(_context_cache_ts).strftime("%Y-%m-%d %H:%M") if _context_cache_ts else ""
                for rt in refresh_times:
                    if current_time == rt and not last_ts.endswith(rt):
                        _refresh_context()
                        break
        except Exception:
            pass

def _build_context(states: list[dict]) -> str:
    """Build context string from state list. Pure computation, no I/O.

    Format per entity: `name | entity_id | state` (state included so the LLM
    can answer "trạng thái đèn X?" without calling ha_get_state — saves a
    full round-trip). For sensors with units the unit is appended:
    `Nhiệt độ phòng học | sensor.nhiet_am_phong_hoc_temperature | 28.5 °C`.
    """
    by_domain: dict[str, list[dict]] = {}
    valid_count = 0
    for s in states:
        eid = s.get("entity_id", "")
        domain = eid.split(".")[0] if "." in eid else ""
        if domain not in _CONTEXT_DOMAINS:
            continue
        valid_count += 1
        by_domain.setdefault(domain, []).append(s)

    lines = [
        "## Smart Home — Device Registry — Trạng thái thiết bị (LIVE, làm mới ~60s/lần)",
        f"{valid_count} thiết bị. Mỗi dòng: `tên | entity_id | trạng thái`.",
        "**CÁCH DÙNG:**",
        "- Khi user hỏi trạng thái / liệt kê / tổng quan → TRẢ LỜI TRỰC TIẾP từ dữ "
        "liệu bên dưới (mỗi dòng đã kèm sẵn trạng thái). TUYỆT ĐỐI KHÔNG gọi "
        "`GetLiveContext` hay tool đọc nào nữa — dữ liệu này CHÍNH LÀ trạng thái hiện tại.",
        "- Khi user yêu cầu điều khiển (bật/tắt/mở/đóng/đặt) → tìm entity_id trong "
        "registry bên dưới, rồi gọi `ha_call_service` MỘT LẦN với entity_id chính xác.",
        "",
        "--- KHI HỎI TRẠNG THÁI NHÀ ('trạng thái nhà', 'chi tiết toàn bộ thiết bị') ---",
        "1. TỔNG HỢP & BÁO CÁO NGAY từ dữ liệu dưới đây theo nhóm: Đèn, Quạt, Điều hoà, Cửa, Công tắc, Khóa.",
        "2. BỎ QUA cảm biến (thời tiết, nhiệt độ, độ ẩm, contact) trừ khi được hỏi ĐÍCH DANH.",
        "3. KHÔNG gọi bất kỳ tool đọc nào (GetLiveContext/ha_get_state/ha_search_entities) — trả lời thẳng.",
        "",
    ]

    for domain in sorted(by_domain.keys()):
        entities = by_domain[domain]
        lines.append(f"[{domain}] ({len(entities)})")
        for s in entities[:_MAX_PER_DOMAIN]:
            eid = s.get("entity_id", "")
            attrs = s.get("attributes", {}) or {}
            name = attrs.get("friendly_name", "")
            state = str(s.get("state", "") or "").strip()
            unit = str(attrs.get("unit_of_measurement", "") or "").strip()
            if state and unit:
                state_str = f"{state} {unit}"
            elif state:
                state_str = state
            else:
                state_str = "unknown"
            if name:
                lines.append(f"  {name} | {eid} | {state_str}")
            else:
                lines.append(f"  {eid} | {state_str}")
        if len(entities) > _MAX_PER_DOMAIN:
            lines.append(f"  ... còn {len(entities) - _MAX_PER_DOMAIN} thiết bị [{domain}]")

    lines.append("")
    lines.append("## Available Services (chỉ dùng cho điều khiển)")
    svc = _get_services()
    for domain in sorted(by_domain.keys()):
        svc_list = svc.get(domain, [])
        if svc_list:
            lines.append(f"  {domain}: {', '.join(svc_list[:10])}")
    lines.append("")
    lines.append("`ha_call_service` là tool DUY NHẤT cần dùng khi điều khiển.")

    return "\n".join(lines)


# Smart-home intent detection.
#
# Single ASCII words like "den" or "nha" are too ambiguous to use as triggers
# ("đen" = black, "nhà" can be a particle), so we match only multi-word
# phrases or unambiguous tokens. Each pattern requires either a strong noun
# ("home assistant", "thiết bị / thiet bi") or a verb+noun pair that only
# makes sense in a smart-home context (e.g. "bật đèn", "liet ke quat",
# "trang thai cua"). Patterns run against both the original lowercased text
# and a diacritic-folded copy so queries with or without dấu both register.
import re as _re

# Device / room nouns (will be paired with action/listing verbs below).
_HA_NOUNS = (
    r"den|đèn|quat|quạt|may\s*lanh|máy\s*lạnh|dieu\s*hoa|điều\s*hòa|"
    r"cua|cửa|khoa|khóa|rem|rèm|cong\s*tac|công\s*tắc|o\s*cam|ổ\s*cắm|"
    r"cam\s*bien|cảm\s*biến|nhiet\s*do|nhiệt\s*độ|do\s*am|độ\s*ẩm|"
    r"thiet\s*bi|thiết\s*bị|fan|light|switch|sensor|climate|cover|lock|"
    r"outlet|plug|curtain|blind|thermostat|smart\s*plug"
)

# Verbs that, paired with a device noun, are unambiguous HA intents.
_HA_VERBS = (
    r"bat|bật|tat|tắt|mo|mở|dong|đóng|kiem\s*tra|kiểm\s*tra|"
    r"dieu\s*khien|điều\s*khiển|on|off|toggle|turn(?:\s*on|\s*off)?"
)

# Listing / status verbs (also unambiguous when paired with a device noun).
_HA_LISTING = (
    r"liet\s*ke|liệt\s*kê|danh\s*sach|danh\s*sách|co\s*nhung|có\s*những|"
    r"trang\s*thai|trạng\s*thái|tinh\s*trang|tình\s*trạng|"
    r"list|show|status|state|enumerate"
)

# Strong standalone tokens — these alone are enough to flag HA intent.
_HA_STRONG = (
    r"home\s*assistant|smart\s*home|smarthome|"
    r"entity_id|ha_(?:get_state|search_entities|call_service)"
)

_HA_INTENT_PATTERNS = [
    _re.compile(rf"\b(?:{_HA_STRONG})\b", _re.IGNORECASE),
    _re.compile(rf"\b(?:{_HA_VERBS})\s+(?:cac\s+|các\s+|tat\s+ca\s+|tất\s+cả\s+)?(?:{_HA_NOUNS})\b", _re.IGNORECASE),
    _re.compile(rf"\b(?:{_HA_LISTING})\s+(?:cac\s+|các\s+|tat\s+ca\s+|tất\s+cả\s+|hết\s+|het\s+)?(?:{_HA_NOUNS})\b", _re.IGNORECASE),
    # "trạng thái nhà / status of the house" — house-level status
    _re.compile(rf"\b(?:{_HA_LISTING})\s+(?:nha|nhà|house|home)\b", _re.IGNORECASE),
    # Direct mention of a room paired with a device noun
    _re.compile(
        rf"\b(?:{_HA_NOUNS})\s+(?:phong|phòng|bep|bếp|tam|tắm|ngu|ngủ|khach|khách|"
        r"ban\s*cong|ban\s*công|room|bedroom|kitchen|bathroom|living)\b",
        _re.IGNORECASE,
    ),
]


def _fold_diacritics(text: str) -> str:
    """Lowercase + strip Vietnamese diacritics for keyword matching."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def is_ha_query(messages: list[dict[str, Any]]) -> bool:
    """Public wrapper for HA intent detection. Used by handle() to decide
    on the PRISTINE user message before search/other injections so that
    e.g. "mở cửa" appearing inside gold-price search results doesn't get
    misread as a "mở cửa" smart-home command."""
    return _is_ha_query(messages)


def _is_ha_query(messages: list[dict[str, Any]]) -> bool:
    """Heuristic: is the last user message asking about smart home devices?

    Requires an unambiguous phrase (verb+device, listing+device, strong token,
    or device+room) so generic words like "đen" / "den" / "nhà" alone do not
    trigger HA injection. Checks both the lowercased original text and a
    diacritic-folded copy so input with or without dấu both register.
    """
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        raw = str(m.get("content") or "").lower()
        folded = _fold_diacritics(raw)
        for pat in _HA_INTENT_PATTERNS:
            if pat.search(raw) or pat.search(folded):
                return True
        return False
    return False


def inject_ha_context(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inject HA entity registry as a system message — only for HA-related queries."""
    if not _is_ha_query(messages):
        return messages

    ctx = format_states_context()
    if not ctx:
        return messages

    result = list(messages)
    insert_pos = len(result)
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "user":
            insert_pos = i
            break
    result.insert(insert_pos, {"role": "system", "content": ctx})
    logger.info({"event": "ha_context_injected", "chars": len(ctx)})
    return result


def get_ha_tools() -> list[dict[str, Any]]:
    """Return OpenAI-format tools for HA control (get state, call service)."""
    cfg = _get_ha_config()
    if not cfg:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": "GetLiveContext",
                "description": "Lấy TOÀN BỘ trạng thái hiện tại của tất cả thiết bị trong nhà (đèn, cảm biến, công tắc, khóa...). GỌI ĐẦU TIÊN khi user hỏi 'chi tiết', 'toàn bộ', 'tổng quan', 'trạng thái hiện tại'. Không cần tham số — trả về danh sách đầy đủ tên + entity_id + state + unit.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ha_get_state",
                "description": "Lấy TRẠNG THÁI HIỆN TẠI của 1 thiết bị (đang bật/tắt, nhiệt độ, độ ẩm...). CHỈ DÙNG khi user hỏi về trạng thái cụ thể (ví dụ: 'đèn bếp đang bật không', 'nhiệt độ phòng ngủ'). KHÔNG dùng cho câu hỏi liệt kê.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string", "description": "Entity ID (vd: light.ban_cong, sensor.nhiet_do)"}
                    },
                    "required": ["entity_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ha_search_entities",
                "description": "LIỆT KÊ thiết bị theo từ khóa. DÙNG cho câu hỏi 'danh sách', 'có những X nào', 'liệt kê'. Trả về name + entity_id (KHÔNG có trạng thái). Tự động lọc theo domain (đèn → light.*, quạt → fan.*, công tắc → switch.*). Để xem automation/scene của thứ gì, thêm 'tự động hóa' / 'scene' vào query (vd: 'tự động hóa đèn'). KHÔNG cần gọi ha_get_state sau đó nếu user chỉ hỏi danh sách.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Từ khóa tìm kiếm (vd: đèn, quạt, đèn ban công, tự động hóa đèn)"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ha_call_service",
                "description": "Gọi Home Assistant service để ĐIỀU KHIỂN thiết bị (bật/tắt đèn, khóa cửa, đặt nhiệt độ). CHỈ DÙNG khi user yêu cầu hành động (ví dụ: 'tắt đèn bếp', 'mở rèm').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string", "description": "Domain: light, switch, lock, climate, cover..."},
                        "service": {"type": "string", "description": "Service: turn_on, turn_off, toggle, lock, unlock..."},
                        "entity_id": {"type": "string", "description": "Entity ID đầy đủ (vd: light.ban_cong)"},
                    },
                    "required": ["domain", "service", "entity_id"],
                },
            },
        },
    ]


def execute_ha_tool(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Execute an HA tool and return result text."""
    if tool_name == "GetLiveContext":
        return format_states_context()
    elif tool_name == "ha_get_state":
        eid = arguments.get("entity_id", "")
        state = get_state(eid)
        if state is None:
            return f"Không tìm thấy thiết bị '{eid}'"
        return json.dumps(state, ensure_ascii=False, indent=2)
    elif tool_name == "ha_search_entities":
        query = arguments.get("query", "").lower().strip()
        states = get_states()

        # Detect domain intent from query keywords (Vietnamese + English).
        # If user asks "đèn" → only light.*, not switch/automation/scene that also contain "đèn"
        # in friendly_name. To include automations, user must say "tự động hóa đèn" / "automation đèn".
        DOMAIN_KEYWORDS: dict[str, list[str]] = {
            "light": ["đèn", "light"],
            "switch": ["công tắc", "switch", "ổ cắm", "ổ điện"],
            "climate": ["điều hòa", "máy lạnh", "climate", "nhiệt độ", "thermostat"],
            "cover": ["rèm", "mành", "cửa cuốn", "cover"],
            "lock": ["khóa", "lock"],
            "fan": ["quạt", "fan"],
            "media_player": ["loa", "tivi", "tv", "media"],
            "sensor": ["cảm biến", "sensor"],
            "scene": ["scene", "ngữ cảnh"],
            "automation": ["tự động hóa", "automation"],
            "script": ["script", "kịch bản"],
            "vacuum": ["robot hút bụi", "vacuum"],
        }
        # Force-domain takes priority: phrases that mention "đèn" but explicitly ask
        # for automation/scene/script of that thing.
        force_domain: str | None = None
        for kw in DOMAIN_KEYWORDS["automation"]:
            if kw in query:
                force_domain = "automation"
                break
        if force_domain is None:
            for kw in DOMAIN_KEYWORDS["scene"]:
                if kw in query:
                    force_domain = "scene"
                    break
        if force_domain is None:
            for kw in DOMAIN_KEYWORDS["script"]:
                if kw in query:
                    force_domain = "script"
                    break
        # Match primary thing (light/switch/etc) only when no force_domain
        primary_domain: str | None = None
        if force_domain is None:
            for domain, kws in DOMAIN_KEYWORDS.items():
                if domain in ("automation", "scene", "script"):
                    continue
                if any(kw in query for kw in kws):
                    primary_domain = domain
                    break

        target_domain = force_domain or primary_domain

        matches = []
        for s in states:
            eid = s.get("entity_id", "").lower()
            name = s.get("attributes", {}).get("friendly_name", "").lower()
            domain = eid.split(".")[0] if "." in eid else ""
            # If we detected a target domain, hard-filter to that domain only
            if target_domain and domain != target_domain:
                continue
            if query in eid or query in name:
                real_name = s.get("attributes", {}).get("friendly_name", "")
                label = f"{real_name} | {eid}" if real_name else eid
                matches.append(label)
        if not matches:
            scope = f" (domain={target_domain})" if target_domain else ""
            return f"Không tìm thấy thiết bị nào khớp với '{query}'{scope}"
        scope = f" [domain={target_domain}]" if target_domain else ""
        return f"Thiết bị khớp '{query}'{scope} ({len(matches)}):\n" + "\n".join(matches[:30])
    elif tool_name == "ha_call_service":
        domain = arguments.get("domain", "")
        service = arguments.get("service", "")
        entity_id = arguments.get("entity_id", "")
        ok = call_service(domain, service, {"entity_id": entity_id})
        return f"Đã gọi {domain}.{service} cho {entity_id}" if ok else f"Lỗi gọi {domain}.{service}"
    return None

# Start background scheduler on module import
try:
    _ensure_scheduler_running()
except Exception:
    pass
