"""Claude Web login (claude.ai via Google account).

Mirror of gemini_web_login.py: reuses auto_login.do_google_login_steps for
the Google email/password/2FA dance and click_google_oauth_consent for the
"Continue as <email>" shortcut when the profile already holds a Google
session (from Flow / ChatGPT / Gemini onboard) — that's the "reuse Google
account" path đại ca wants: no second 2FA.

Unlike Gemini Web (no standalone token), claude.ai DOES expose a usable
credential: the `sessionKey` cookie. After login we scrape it so the main
chatgpt2api app can call claude.ai's API with it (see api/claude.py).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from .auto_login import (
    LoginSession,
    _already_logged_in,
    click_google_oauth_consent,
    do_google_login_steps,
)
from .browser_pool import pool

logger = logging.getLogger(__name__)

_CLAUDE_HOME = "https://claude.ai/"
_CLAUDE_LOGIN = "https://claude.ai/login"
_SESSION_COOKIE = "sessionKey"

# "Continue with Google" button on the claude.ai login screen.
_GOOGLE_BTN_SELECTORS = (
    'button[data-provider="google"]',
    'button:has-text("Continue with Google")',
    'button:has-text("Tiếp tục với Google")',
    'a[href*="accounts.google.com"]',
    'button:has-text("Google")',
    'div[role="button"]:has-text("Google")',
)


@dataclass
class ClaudeWebLoginSession(LoginSession):
    """Captures the claude.ai sessionKey cookie on success."""
    session_key: str = ""

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["session_key"] = self.session_key
        base["session_key_preview"] = (self.session_key[:18] + "...") if self.session_key else ""
        base["has_session_key"] = bool(self.session_key)
        return base


_sessions: dict[str, ClaudeWebLoginSession] = {}
_tasks: dict[str, asyncio.Task] = {}


def get_session(profile: str) -> Optional[ClaudeWebLoginSession]:
    return _sessions.get(profile)


def submit_2fa_code(profile: str, code: str) -> bool:
    session = _sessions.get(profile)
    if not session or session.state != "need_code":
        return False
    session.pending_code = code.strip()
    session.message = "Đã nhận mã, đang submit..."
    return True


async def _scrape_session_key(ctx) -> str:
    """Read the claude.ai `sessionKey` cookie value from the context."""
    try:
        cookies = await ctx.cookies("https://claude.ai")
    except Exception:
        try:
            cookies = await ctx.cookies()
        except Exception:
            return ""
    for c in cookies:
        if c.get("name") == _SESSION_COOKIE and str(c.get("value") or ""):
            return str(c.get("value"))
    return ""


async def start_claude_web_login(
    profile: str,
    email: str,
    password: str,
    totp_secret: str = "",
) -> ClaudeWebLoginSession:
    """Kick off background Claude Web login.

    If the profile already has a Google session (from another onboard), the
    Google login is skipped — claude.ai rides the SSO via OAuth consent.
    """
    old_task = _tasks.pop(profile, None)
    if old_task and not old_task.done():
        old_task.cancel()

    session = ClaudeWebLoginSession(
        profile=profile,
        email=email,
        state="starting",
        message="Khởi tạo Chrome",
        totp_secret=totp_secret,
    )
    _sessions[profile] = session

    task = asyncio.create_task(_run(session, password))
    _tasks[profile] = task
    return session


async def _run(session: ClaudeWebLoginSession, password: str) -> None:
    try:
        session.state = "starting"
        session.message = "Đang mở Chrome (headful → noVNC)"
        ctx = await pool.get(profile=session.profile, headless=False, force_recreate=True)

        pages = ctx.pages
        page = pages[0] if pages else await ctx.new_page()
        try:
            await page.bring_to_front()
        except Exception:
            pass

        # ── Already have a claude.ai session on this profile? short-circuit ──
        session.state = "running"
        session.message = "Mở claude.ai..."
        await page.goto(_CLAUDE_HOME, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(3.0)

        existing = await _scrape_session_key(ctx)
        if existing:
            session.session_key = existing
            session.state = "success"
            session.message = "Profile đã có Claude session — không cần đăng nhập"
            session.completed_at = time.time()
            return

        # ── Open the login screen + click "Continue with Google" ──
        session.message = "Mở trang đăng nhập claude.ai..."
        try:
            if "/login" not in (page.url or ""):
                await page.goto(_CLAUDE_LOGIN, wait_until="domcontentloaded", timeout=30_000)
                await asyncio.sleep(2.0)
        except Exception:
            pass

        session.message = "Click 'Continue with Google'..."
        google_clicked = False
        for sel in _GOOGLE_BTN_SELECTORS:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.click(timeout=5_000)
                    google_clicked = True
                    logger.info("claude_login: clicked Google via %s", sel)
                    break
            except Exception:
                continue
        if not google_clicked:
            try:
                google_clicked = await page.evaluate("""() => {
                    const all = document.querySelectorAll('button, a, div[role="button"]');
                    for (const el of all) {
                        if (!el.offsetParent) continue;
                        const t = (el.innerText || el.getAttribute('aria-label') || '').toLowerCase();
                        const h = el.getAttribute('href') || '';
                        if (t.includes('google') || h.includes('accounts.google.com')) { el.click(); return true; }
                    }
                    return false;
                }""")
            except Exception:
                google_clicked = False
        if not google_clicked:
            session.state = "failed"
            session.error = "Không tìm thấy nút 'Continue with Google' trên claude.ai"
            session.completed_at = time.time()
            return

        # Wait for redirect to accounts.google.com.
        try:
            await page.wait_for_url("**/accounts.google.com/**", timeout=20_000)
            logger.info("claude_login: at google login page %s", page.url)
        except Exception:
            logger.warning("claude_login: no accounts.google.com redirect after click")

        try:
            on_google = "accounts.google.com" in (page.url or "")
        except Exception:
            on_google = False

        # Pre-consent: profile already has Google cookies → one-button confirm.
        try:
            pre_consent = await click_google_oauth_consent(page, timeout=6.0)
            if pre_consent:
                session.message = "Đã bấm OAuth consent..."
                await asyncio.sleep(2.0)
        except Exception:
            pre_consent = False

        if on_google and not pre_consent:
            # Fresh Google login (email/password/2FA) on this profile.
            ok = await do_google_login_steps(session, page, ctx, password)
            if not ok:
                return
            try:
                if await click_google_oauth_consent(page, timeout=8.0):
                    session.message = "Đã bấm OAuth consent..."
                    await asyncio.sleep(2.0)
            except Exception:
                pass

        # Wait for redirect back to claude.ai and the sessionKey cookie.
        try:
            await page.wait_for_url("**/claude.ai/**", timeout=30_000)
        except Exception:
            logger.warning("claude_login: no return to claude.ai (url=%s)", getattr(page, "url", "?"))
        await asyncio.sleep(3.0)

        for _ in range(20):
            key = await _scrape_session_key(ctx)
            if key:
                session.session_key = key
                session.state = "success"
                session.message = "Đăng nhập Claude Web thành công"
                session.completed_at = time.time()
                return
            await asyncio.sleep(1.5)

        # Soft-fail: Google cookies present but claude.ai sessionKey not seen.
        if await _already_logged_in(ctx):
            session.state = "failed"
            session.error = "Google OK nhưng chưa lấy được sessionKey của claude.ai (thử lại / kiểm tra noVNC)"
        else:
            session.state = "failed"
            session.error = f"Không hoàn tất đăng nhập claude.ai (url={getattr(page, 'url', '?')})"
        session.completed_at = time.time()

    except asyncio.CancelledError:
        session.state = "failed"
        session.error = "Bị huỷ (có yêu cầu login mới)"
        session.completed_at = time.time()
        raise
    except Exception as exc:
        logger.exception("claude_web_login crashed profile=%s", session.profile)
        session.state = "failed"
        session.error = str(exc)
        session.completed_at = time.time()
