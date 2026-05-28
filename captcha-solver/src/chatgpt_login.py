"""ChatGPT onboard via Google OAuth.

Flow: chatgpt.com → click Login → click "Continue with Google"
→ Google OAuth (email/password/2FA) → redirect back to chatgpt.com
→ scrape JWT access_token.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from .browser_pool import pool

try:
    import pyotp
    _HAS_PYOTP = True
except ImportError:
    pyotp = None  # type: ignore
    _HAS_PYOTP = False

logger = logging.getLogger(__name__)

_CHATGPT_URL = "https://chatgpt.com/"
_AUTH_OPENAI = "https://auth.openai.com/"

# Selectors for "Continue with Google" on auth.openai.com
_GOOGLE_BTN_SELECTORS = (
    'button[data-provider="google"]',
    'button:has-text("Continue with Google")',
    'button:has-text("Tiếp tục với Google")',
    'a[href*="accounts.google.com"]',
    'button[aria-label*="Google"]',
    'form[action*="accounts.google.com"] button',
)

# Selectors for Google 2FA code input
_2FA_CODE_SELECTORS = (
    'input[type="tel"][autocomplete="one-time-code"]',
    'input[name="totpPin"]',
    'input[id="totpPin"]',
    'input[type="tel"]:not([disabled])',
)

# Cookies that prove a Google login completed
_GOOGLE_LOGIN_COOKIES = ("__Secure-1PSID", "__Secure-3PSID", "SID")


async def _type_human_like(locator, text: str) -> None:
    """Type text character-by-character with randomized delays."""
    # Random pause before starting (human looks at phone)
    await asyncio.sleep(random.uniform(0.4, 1.2))
    for i, ch in enumerate(text):
        await locator.press(ch, delay=random.randint(80, 350))
        # Occasionally pause mid-code
        if i == 2 and random.random() < 0.6:
            await asyncio.sleep(random.uniform(0.2, 0.6))


async def _safe_click(page, *selectors: str, timeout: int = 5000) -> bool:
    """Try each selector; click the first visible match. Returns True on success."""
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                await loc.click(timeout=timeout)
                return True
        except Exception:
            continue
    return False


# Selectors ONLY for Authenticator/TOTP — do NOT match SMS/phone.
# Google's 2FA method picker uses div.l5PPKe[jsname="fmcmS"] inside an <li>.
# We target the clickable element that CONTAINS "Google Authenticator" text.
_AUTHENTICATOR_OPTION_SELECTORS = (
    'li[data-challengetype="9"]',
    'div[data-challengetype="9"]',
    # Direct match for the Google Authenticator option div
    'div.l5PPKe:has(strong:has-text("Google Authenticator"))',
    'div[jsname="fmcmS"]:has(strong:has-text("Google Authenticator"))',
    # Fallback: any li containing Google Authenticator text
    'li:has(strong:has-text("Google Authenticator"))',
    'li:has(div:has-text("Google Authenticator"))',
    # Click parent li of the option div
    'li:has(div.l5PPKe:has-text("Google Authenticator"))',
)

_METHOD_SELECTOR_HINTS = (
    "choose how you", "chọn cách",
    "try another way", "thử cách khác",
    "get a verification code", "nhập mã xác minh",
    "google authenticator", "authenticator",
    "ứng dụng xác thực",
)


async def _pick_authenticator_method(page) -> bool:
    """When Google shows the method picker, click the Authenticator entry.
    NEVER clicks phone/SMS options — only Authenticator."""
    try:
        body_text = (await page.locator("body").inner_text(timeout=600)).lower()
    except Exception:
        return False

    if not any(h in body_text for h in _METHOD_SELECTOR_HINTS):
        return False

    # Log ALL available challenge options for debugging
    try:
        all_items = page.locator('li[data-challengetype], div[data-challengetype]')
        count = await all_items.count()
        opts = []
        for i in range(min(count, 8)):
            try:
                el = all_items.nth(i)
                ct = await el.get_attribute("data-challengetype")
                txt = (await el.inner_text())[:60]
                opts.append(f"type={ct} text='{txt}'")
            except Exception:
                pass
        # Also try to list all li elements in the method picker
        all_lis = page.locator('ul li, div[role="list"] li, section li')
        lic = await all_lis.count()
        if lic > 0:
            opts.append(f"[total {lic} li elements]")
        logger.info("chatgpt_login: method picker options: %s", " | ".join(opts) if opts else "none found")
    except Exception:
        pass

    # Strategy: find ANY element that contains "Google Authenticator" text and click it
    # Going from most specific to least specific
    authenticator_selectors = (
        'li:has(strong:text-is("Google Authenticator"))',
        'li:has(strong:has-text("Google Authenticator"))',
        'div[jsname="fmcmS"]:has(strong:has-text("Google Authenticator"))',
        'div.l5PPKe:has(strong:has-text("Google Authenticator"))',
        'div[jsname="fmcmS"]:has-text("Google Authenticator")',
        'div.l5PPKe:has-text("Google Authenticator")',
    )

    for sel in authenticator_selectors:
        try:
            # Get ALL matching elements (there may be duplicates)
            locs = page.locator(sel)
            cnt = await locs.count()
            if cnt == 0:
                continue
            # Click the first visible one
            for i in range(cnt):
                el = locs.nth(i)
                try:
                    if await el.is_visible(timeout=500):
                        await el.click(timeout=2500)
                        logger.info("chatgpt_login: picked Authenticator via %s[%d]", sel, i)
                        return True
                except Exception:
                    continue
        except Exception:
            continue

    # Last resort: find by iterating all <li> elements
    try:
        all_lis = page.locator('li')
        lic = await all_lis.count()
        for i in range(min(lic, 12)):
            li = all_lis.nth(i)
            try:
                txt = (await li.inner_text(timeout=300)).lower()
                if "google authenticator" in txt:
                    await li.click(timeout=2500)
                    logger.info("chatgpt_login: picked Authenticator via li[%d], text='%s'", i, txt[:50])
                    return True
            except Exception:
                continue
    except Exception:
        pass

    logger.warning("chatgpt_login: Authenticator option NOT found on method picker!")
    return False


async def _wait_for_google_login(page, timeout: int = 60) -> bool:
    """Wait until Google login cookies appear (meaning OAuth completed)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        cookies = await page.context.cookies()
        names = {c["name"] for c in cookies}
        if any(ck in names for ck in _GOOGLE_LOGIN_COOKIES):
            return True
        await asyncio.sleep(1.0)
    return False


async def _scrape_chatgpt_token(page) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """After redirect back to chatgpt.com, scrape the JWT and email.

    Returns (access_token, captured_email, access_token_preview).
    """
    access_token = None
    captured_email = None
    access_token_preview = None

    # Try localStorage / sessionStorage first
    for storage_key in ("localStorage", "sessionStorage"):
        try:
            token = await page.evaluate(
                f"""(() => {{
                    const keys = Object.keys({storage_key});
                    for (const k of keys) {{
                        const v = {storage_key}.getItem(k);
                        if (v && v.startsWith('eyJ') && v.length > 100) return v;
                    }}
                    return null;
                }})()"""
            )
            if token:
                logger.info("chatgpt_login: found JWT in %s", storage_key)
                access_token = token
                access_token_preview = token[:40] + "..." if len(token) > 40 else token
                break
        except Exception:
            continue

    # Fallback: check cookies for __Secure-next-auth.session-token
    if not access_token:
        try:
            cookies = await page.context.cookies()
            for c in cookies:
                if c.get("name", "").startswith("__Secure-next-auth") and len(c.get("value", "")) > 50:
                    access_token = c["value"]
                    access_token_preview = access_token[:40] + "..."
                    logger.info("chatgpt_login: found token in cookie %s", c["name"])
                    break
        except Exception:
            pass

    # Try to get email from the page
    try:
        captured_email = await page.evaluate("""(() => {
            const el = document.querySelector('[data-testid="account-email"], [title*="@"]');
            if (el) return el.textContent?.trim() || el.getAttribute("title");
            const userEl = document.querySelector('[class*="user"], [class*="profile"], [class*="account"]');
            if (userEl) {
                const text = userEl.textContent || '';
                const match = text.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}/);
                return match ? match[0] : null;
            }
            return null;
        })()""")
    except Exception:
        pass

    return access_token, captured_email, access_token_preview


@dataclass
class ChatGPTOnboardSession:
    profile: str = "chatgpt-default"
    email: str = ""
    state: str = "none"  # none, starting, running, need_tap, need_code, success, failed
    message: str = ""
    tap_number: Optional[str] = None
    elapsed_sec: float = 0.0
    error: Optional[str] = None
    access_token: Optional[str] = None
    expires: Optional[str] = None
    captured_email: Optional[str] = None
    access_token_preview: Optional[str] = None
    totp_secret: str = ""
    pending_code: Optional[str] = None
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "email": self.email,
            "state": self.state,
            "message": self.message,
            "tap_number": self.tap_number,
            "elapsed_sec": int(time.time() - self.started_at),
            "error": self.error,
            "access_token": self.access_token,
            "expires": self.expires,
            "captured_email": self.captured_email,
            "access_token_preview": self.access_token_preview,
        }


# In-memory session store
_sessions: dict[str, ChatGPTOnboardSession] = {}


def get_session(profile: str) -> Optional[ChatGPTOnboardSession]:
    return _sessions.get(profile)


async def start_chatgpt_onboard(
    profile: str = "chatgpt-default",
    email: str = "",
    password: str = "",
    totp_secret: str = "",
) -> ChatGPTOnboardSession:
    """Launch ChatGPT onboard in background. Returns immediately with initial state."""
    session = ChatGPTOnboardSession(
        profile=profile,
        email=email,
        totp_secret=totp_secret,
        state="starting",
        message="Khoi dong trinh duyet...",
    )
    _sessions[profile] = session

    # Kill any existing browser context so we start fresh
    await pool.close_profile(profile)
    await asyncio.sleep(0.5)

    asyncio.create_task(_run_onboard(session, password))
    return session


async def _run_onboard(session: ChatGPTOnboardSession, password: str) -> None:
    """Main onboard orchestration."""
    started_at = time.time()
    try:
        async with pool.page(profile=session.profile, headless=False) as page:
            # ── Step 1: Navigate to chatgpt.com ──
            session.state = "running"
            session.message = "Dang mo chatgpt.com..."
            logger.info("chatgpt_login: navigating to %s", _CHATGPT_URL)
            await page.goto(_CHATGPT_URL, wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(2.0)

            # ── Step 2: Click "Log in" on chatgpt.com ──
            session.message = "Dang tim nut Login..."
            login_clicked = False
            login_selectors = (
                'button:has-text("Log in")',
                'button:has-text("Đăng nhập")',
                'a[href*="/auth/login"]',
                'a[href*="auth.openai.com"]',
                'button[data-testid="login-button"]',
            )
            for sel in login_selectors:
                try:
                    loc = page.locator(sel).first
                    if await loc.count() > 0:
                        await loc.click(timeout=5_000)
                        login_clicked = True
                        logger.info("chatgpt_login: clicked login via %s", sel)
                        break
                except Exception:
                    continue

            if login_clicked:
                await asyncio.sleep(3.0)

            # ── Step 3: Click "Continue with Google" ──
            # May be on auth.openai.com or a popup
            session.message = "Dang tim nut Google..."

            # Check all open pages for the Google button
            google_clicked = False
            for pg in page.context.pages:
                try:
                    if pg.is_closed():
                        continue
                except Exception:
                    continue

                for sel in _GOOGLE_BTN_SELECTORS:
                    try:
                        loc = pg.locator(sel).first
                        if await loc.count() > 0:
                            # Wait for any navigation triggered by click
                            await loc.click(timeout=5_000)
                            google_clicked = True
                            logger.info("chatgpt_login: clicked Google via %s on %s", sel, pg.url)
                            break
                    except Exception:
                        continue
                if google_clicked:
                    break

            if not google_clicked:
                # Fallback: navigate directly to Google OAuth
                logger.warning("chatgpt_login: couldn't find Google button, trying direct Google login")
                session.message = "Khong tim thay nut Google, thu direct login..."
                await page.goto(
                    "https://accounts.google.com/signin/v2/identifier"
                    "?hl=vi&service=accountsettings",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )

            await asyncio.sleep(2.0)

            # ── Step 4: Google login (email) ──
            session.message = "Dang nhap Google email..."
            for pg in page.context.pages:
                try:
                    if pg.is_closed():
                        continue
                except Exception:
                    continue

                # Google email input
                email_filled = False
                email_selectors = (
                    'input[type="email"]',
                    'input[name="identifier"]',
                    '#identifierId',
                )
                for sel in email_selectors:
                    try:
                        loc = pg.locator(sel).first
                        if await loc.count() > 0:
                            await loc.click(timeout=3000)
                            await loc.fill("")
                            await asyncio.sleep(random.uniform(0.2, 0.5))
                            await _type_human_like(loc, session.email)
                            email_filled = True
                            logger.info("chatgpt_login: filled email via %s", sel)
                            break
                    except Exception:
                        continue

                if email_filled:
                    # Click "Next" / "Tiếp theo"
                    await asyncio.sleep(0.3)
                    await _safe_click(
                        pg,
                        'button:has-text("Next")',
                        'button:has-text("Tiếp theo")',
                        'span[jsname="V67aGc"]',
                        '#identifierNext',
                        'button[jsname="LgbsSe"]:visible',
                    )
                    break

            # ── Step 5: Google login (password) ──
            await asyncio.sleep(2.5)
            session.message = "Dang nhap mat khau..."

            for pg in page.context.pages:
                try:
                    if pg.is_closed():
                        continue
                except Exception:
                    continue

                pw_filled = False
                pw_selectors = (
                    'input[type="password"]',
                    'input[name="Passwd"]',
                    '#password input[type="password"]',
                )
                for sel in pw_selectors:
                    try:
                        loc = pg.locator(sel).first
                        if await loc.count() > 0:
                            await loc.click(timeout=3000)
                            await loc.fill("")
                            await asyncio.sleep(random.uniform(0.2, 0.5))
                            await _type_human_like(loc, password)
                            pw_filled = True
                            logger.info("chatgpt_login: filled password")
                            break
                    except Exception:
                        continue

                if pw_filled:
                    await asyncio.sleep(0.3)
                    await _safe_click(
                        pg,
                        'button:has-text("Next")',
                        'button:has-text("Tiếp theo")',
                        '#passwordNext',
                        'button[jsname="LgbsSe"]:visible',
                    )
                    break

            # ── Step 6: Handle 2FA if needed ──
            session.elapsed_sec = time.time() - started_at
            auth_picked = False       # Only pick Authenticator once
            code_attempt = 0          # Track retries for TOTP
            for _ in range(40):       # up to ~4 min polling
                await asyncio.sleep(3.0)
                session.elapsed_sec = time.time() - started_at

                # Check if we're already on chatgpt.com (login succeeded without 2FA)
                for pg in page.context.pages:
                    try:
                        if pg.is_closed():
                            continue
                    except Exception:
                        continue
                    if "chatgpt.com" in (pg.url or "") and "auth" not in (pg.url or ""):
                        session.state = "success"
                        session.message = "Da dang nhap ChatGPT (khong can 2FA)"
                        break

                if session.state == "success":
                    break

                # Check for 2FA on any page
                for pg in page.context.pages:
                    try:
                        if pg.is_closed():
                            continue
                    except Exception:
                        continue

                    # Auto-pick Authenticator — only ONCE per session
                    if not auth_picked and await _pick_authenticator_method(pg):
                        auth_picked = True
                        session.message = "Da chon Google Authenticator, dang cho code..."
                        logger.info("chatgpt_login: auto-picked Authenticator")
                        await asyncio.sleep(3.0)
                        continue

                    # Detect phone tap prompt
                    try:
                        tap_el = pg.locator(
                            'div[jsname="RjfePd"] span, '
                            '[data-tap-target], '
                            'text=/tap.*\\d{2}/i'
                        ).first
                        if await tap_el.count() > 0:
                            text = await tap_el.text_content() or ""
                            import re
                            nums = re.findall(r"\d{2}", text)
                            if nums:
                                session.tap_number = nums[0]
                                session.state = "need_tap"
                                session.message = f"Mo app Google tren dien thoai, bam so {nums[0]}"
                                logger.info("chatgpt_login: need_tap=%s", nums[0])
                                continue
                    except Exception:
                        pass

                    # Detect TOTP/2FA code input
                    for sel in _2FA_CODE_SELECTORS:
                        try:
                            loc = pg.locator(sel).first
                            if await loc.count() > 0:
                                visible = await loc.is_visible()
                                if not visible:
                                    continue

                                # Generate code — retry with fresh code if previous attempt failed
                                if session.totp_secret and _HAS_PYOTP:
                                    secret = session.totp_secret.replace(" ", "")
                                    code = pyotp.TOTP(secret).now()
                                    logger.info("chatgpt_login: TOTP code=%s (attempt %d)", code, code_attempt + 1)
                                    session.state = "need_code"
                                    session.message = f"Da tu sinh ma 2FA (lan {code_attempt + 1})"
                                    # Wait for fresh window if retrying
                                    if code_attempt > 0:
                                        remaining = 30 - (int(time.time()) % 30)
                                        if remaining < 5:
                                            await asyncio.sleep(remaining + 2)
                                            code = pyotp.TOTP(secret).now()
                                            logger.info("chatgpt_login: refreshed TOTP code=%s", code)
                                    await asyncio.sleep(1.0)
                                else:
                                    session.state = "need_code"
                                    session.message = "Can ma 2FA"
                                    code_deadline = time.time() + 180
                                    while time.time() < code_deadline and not session.pending_code:
                                        await asyncio.sleep(0.5)
                                    if not session.pending_code:
                                        session.state = "failed"
                                        session.error = "Khong nhan duoc ma 2FA trong 3 phut"
                                        return
                                    code = session.pending_code
                                    session.pending_code = None

                                # Fill code - use fill() for reliability on input[type=tel]
                                await loc.click(timeout=2000)
                                await loc.fill("")
                                await asyncio.sleep(random.uniform(0.3, 0.6))
                                await loc.fill(code)
                                code_attempt += 1
                                # Verify code was actually typed
                                try:
                                    actual = await loc.input_value()
                                    logger.info("chatgpt_login: typed=%s actual=%s (attempt %d)", code, actual, code_attempt)
                                except Exception:
                                    logger.info("chatgpt_login: filled 2FA code (attempt %d)", code_attempt)
                                await asyncio.sleep(random.uniform(0.4, 0.8))
                                await _safe_click(
                                    pg,
                                    'button:has-text("Next")',
                                    'button:has-text("Tiếp theo")',
                                    'span[jsname="V67aGc"]',
                                    '#totpNext button',
                                    '#totpNext span',
                                    'button[jsname="LgbsSe"]:visible',
                                )
                                session.state = "running"
                                session.message = f"Da gui ma (lan {code_attempt}), dang xac minh..."
                                await asyncio.sleep(5.0)

                                # Check immediately if code was accepted
                                try:
                                    body = (await pg.locator("body").inner_text(timeout=1000)).lower()
                                    if any(e in body for e in ("wrong", "sai", "incorrect", "không đúng", "invalid", "khong hop le")):
                                        logger.warning("chatgpt_login: code rejected, will retry")
                                        session.state = "running"  # reset to allow re-detection
                                        if code_attempt >= 3:
                                            session.state = "failed"
                                            session.error = "Sai ma TOTP 3 lan lien tiep"
                                            return
                                        await asyncio.sleep(3.0)
                                        break
                                except Exception:
                                    pass

                                break  # break for sel loop
                        except Exception:
                            continue

            # ── Step 7: Wait for redirect back to chatgpt.com ──
            session.elapsed_sec = time.time() - started_at
            for _ in range(20):  # up to ~60s
                await asyncio.sleep(3.0)
                session.elapsed_sec = time.time() - started_at
                for pg in page.context.pages:
                    try:
                        if pg.is_closed():
                            continue
                    except Exception:
                        continue
                    url = pg.url or ""
                    if "chatgpt.com" in url and "auth" not in url:
                        session.state = "success"
                        session.message = "Da redirect ve chatgpt.com"
                        break
                if session.state == "success":
                    break

            # ── Step 8: Scrape JWT token ──
            if session.state == "success":
                token, captured, preview = await _scrape_chatgpt_token(page)
                if token:
                    session.access_token = token
                    session.captured_email = captured or session.email
                    session.access_token_preview = preview
                    session.message = f"Lay token thanh cong ({captured or session.email})"
                    logger.info("chatgpt_login: scraped token preview=%s", preview)
                else:
                    session.state = "failed"
                    session.error = "Login OK nhung khong scrape duoc JWT"
                    logger.warning("chatgpt_login: login OK but no JWT found")

            if session.state not in ("success", "failed"):
                session.state = "failed"
                session.error = "Het thoi gian ma chua hoan tat"

    except Exception as exc:
        logger.exception("chatgpt_login: onboard failed")
        session.state = "failed"
        session.error = str(exc)[:200]
