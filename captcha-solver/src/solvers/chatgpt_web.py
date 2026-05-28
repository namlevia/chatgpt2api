'''ChatGPT Web (chatgpt.com) DOM-scrape helpers.

Uses the same browser_pool as the rest of the captcha-solver so profiles
that were logged in via /v1/chatgpt/onboard can be reused for direct
chat / image-gen / vision calls against the chatgpt.com SPA.
'''

from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
import time
from typing import Any

import httpx

from ..browser_pool import pool

logger = logging.getLogger(__name__)

_CHATGPT_HOME = 'https://chatgpt.com/'


async def _scrape_models(page) -> list[dict[str, Any]]:
    '''Fetch /backend-api/models inside the page context.'''
    try:
        result = await page.evaluate('''
            async () => {
                const r = await fetch('/backend-api/models', { credentials: 'include' });
                const text = await r.text();
                try { return {status: r.status, json: JSON.parse(text)}; }
                catch { return {status: r.status, text: text.slice(0, 500)}; }
            }
        ''')
        if isinstance(result, dict) and result.get('status') == 200:
            data = result.get('json', {})
            models = data.get('models', []) if isinstance(data, dict) else []
            return models
        return []
    except Exception:
        return []


async def _wait_for_ready(page, timeout: int = 20) -> None:
    """Wait until the chatgpt.com page is interactive (prompt editor visible)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            editor = page.locator('#prompt-textarea, [data-testid=chat-input], div[contenteditable=true]').first
            if await editor.count() > 0:
                try:
                    await editor.wait_for(state='visible', timeout=3000)
                except Exception:
                    await asyncio.sleep(0.5)
                    continue
                return
        except Exception:
            pass
        await asyncio.sleep(0.5)


async def _resolve_image_to_file(image: str) -> tuple[str, str]:
    """Resolve a data: URL or http(s) URL to a temp file. Returns (filepath, mime)."""
    if image.startswith("data:"):
        header, b64 = image.split(",", 1)
        mime = header.split(";")[0].replace("data:", "")
        data = base64.b64decode(b64)
    elif image.startswith(("http://", "https://")):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/130.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8",
        }
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
            r = await client.get(image)
            r.raise_for_status()
        data = r.content
        mime = (r.headers.get("content-type") or "image/png").split(";")[0].strip()
    else:
        raise ValueError("image must be data: URL or http(s) URL")

    ext = mime.split("/")[1] if "/" in mime else "png"
    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name, mime


async def list_models(
    profile: str = 'chatgpt-default',
    headless: bool = True,
    timeout: int = 30,
) -> dict[str, Any]:
    started = time.time()
    async with pool.page(profile=profile, headless=headless) as page:
        try:
            await page.goto(_CHATGPT_HOME, wait_until='domcontentloaded', timeout=timeout * 1000)
            await asyncio.sleep(2.0)
        except Exception:
            pass
        models = await _scrape_models(page)
        elapsed_ms = int((time.time() - started) * 1000)
        return {
            'profile': profile,
            'count': len(models),
            'models': models,
            'elapsed_ms': elapsed_ms,
        }


async def chat(
    profile: str = 'chatgpt-default',
    prompt: str = '',
    timeout: int = 90,
    headless: bool = False,
) -> dict[str, Any]:
    started = time.time()
    async with pool.page(profile=profile, headless=headless) as page:
        try:
            await page.goto(_CHATGPT_HOME, wait_until='domcontentloaded', timeout=20_000)
            await asyncio.sleep(2.0)
        except Exception:
            pass
        await _wait_for_ready(page, timeout=15)
        # Try to type into the prompt editor and send
        try:
            editor = page.locator('#prompt-textarea, [data-testid=chat-input], div[contenteditable=true]').first
            await editor.wait_for(state='visible', timeout=15_000)
            await editor.click()
            await editor.fill(prompt)
            await asyncio.sleep(0.3)
            send_btn = page.locator('button[data-testid=send-button], button:has(svg)').first
            await send_btn.click(timeout=5_000)
        except Exception as exc:
            logger.warning('chatgpt_web chat type/send failed: %s', str(exc)[:120])

        # Wait for response
        await asyncio.sleep(min(timeout, 60))
        try:
            reply_el = page.locator('[data-message-author-role=assistant]').last
            reply_text = await reply_el.inner_text()
        except Exception:
            reply_text = ''
        elapsed_ms = int((time.time() - started) * 1000)
        return {
            'profile': profile,
            'prompt': prompt,
            'reply': reply_text,
            'elapsed_ms': elapsed_ms,
        }


async def generate_image(
    profile: str = 'chatgpt-default',
    prompt: str = '',
    timeout: int = 120,
    headless: bool = False,
) -> dict[str, Any]:
    return {
        'profile': profile,
        'prompt': prompt,
        'images': [],
        'error': 'chatgpt_web generate_image not implemented',
    }


async def analyze_image(
    profile: str = 'chatgpt-default',
    image: str = '',
    prompt: str = 'Phan tich noi dung anh nay mot cach chi tiet.',
    timeout: int = 120,
    headless: bool = False,
) -> dict[str, Any]:
    """Upload an image to chatgpt.com and ask a question about it.

    Args:
        image: either a `data:image/<mime>;base64,<...>` data URL or an
               https URL to an image.
        prompt: text question to ask alongside the image.
        timeout: max seconds to wait for the full response.

    Returns:
        {"text": <ChatGPT's analysis>, "elapsed_ms": int}
    """
    started = time.time()
    stages: dict[str, Any] = {}

    # 1. Resolve image → temp file on disk
    tmp_path, mime = await _resolve_image_to_file(image)
    stages['image_resolved'] = True
    stages['image_mime'] = mime

    try:
        async with pool.page(profile=profile, headless=headless) as page:
            # 2. Navigate to chatgpt.com if not already there
            if 'chatgpt.com' not in (page.url or ''):
                await page.goto(_CHATGPT_HOME, wait_until='domcontentloaded', timeout=30_000)
            await _wait_for_ready(page, timeout=30)
            stages['page_ready'] = True

            # 3. Upload image via file chooser.
            #    chatgpt.com has a "+" button (aria-label="Thêm tệp và nhiều tính năng khác")
            #    Clicking it opens a popover with "Tải ảnh/tệp lên" which triggers the
            #    native file dialog.
            try:
                # Open the + menu
                add_btn = page.locator(
                    'button[aria-label*="Thêm tệp"], '
                    'button[aria-label*="Attach"], '
                    'button[aria-label*="Add file"]'
                ).first
                await add_btn.click(timeout=5_000)
                await asyncio.sleep(0.5)
            except Exception as exc:
                logger.warning('chatgpt_web: could not click + button: %s', str(exc)[:120])
                stages['attach_btn_error'] = str(exc)[:120]

            # 4. Use expect_file_chooser to intercept the native file dialog
            try:
                async with page.expect_file_chooser(timeout=10_000) as fc_info:
                    # Click "Upload file/image" option in the popover
                    upload_opt = page.locator(
                        'button:has-text("Tải ảnh"), '
                        'button:has-text("Tải tệp"), '
                        'button:has-text("Upload"), '
                        'div[role="menuitem"]:has-text("file"), '
                        'div[role="menuitem"]:has-text("Upload"), '
                        'div[role="menuitem"]:has-text("ảnh"), '
                        'div[role="menuitem"]:has-text("tệp")'
                    ).first
                    if await upload_opt.count() > 0:
                        await upload_opt.click(timeout=3_000)
                    else:
                        # Fallback: click the hidden file input directly
                        file_input = page.locator('input[type="file"][accept*="image"]').first
                        await file_input.set_input_files(tmp_path)
                        stages['upload_method'] = 'direct_input'
                        # Skip file_chooser flow
                        raise Exception('used direct input')
                file_chooser = await fc_info.value
                await file_chooser.set_files(tmp_path)
                stages['upload_method'] = 'file_chooser'
                logger.info('chatgpt_web: uploaded image %s (mime=%s) via file chooser',
                            tmp_path, mime)
            except Exception as exc:
                msg = str(exc)[:120]
                if 'used direct input' in msg:
                    pass  # Already handled above
                elif 'file_chooser' not in stages:
                    # File chooser didn't open — try direct file input as fallback
                    logger.warning('chatgpt_web: file_chooser failed (%s), trying direct input', msg)
                    try:
                        file_input = page.locator('input[type="file"][accept*="image"]').first
                        await file_input.set_input_files(tmp_path)
                        stages['upload_method'] = 'direct_input_fallback'
                    except Exception as exc2:
                        logger.error('chatgpt_web: direct input also failed: %s', str(exc2)[:120])
                        stages['upload_error'] = str(exc2)[:120]

            # 5. Wait for image to process (upload + thumbnail appears)
            await asyncio.sleep(3.0)
            stages['image_uploaded'] = True

            # 6. Type the prompt
            try:
                editor = page.locator(
                    '#prompt-textarea, '
                    '[data-testid=chat-input], '
                    'div[contenteditable=true]'
                ).first
                await editor.wait_for(state='visible', timeout=10_000)
                await editor.click()
                await asyncio.sleep(0.3)
                await editor.fill(prompt)
                await asyncio.sleep(0.5)
                stages['prompt_typed'] = True
            except Exception as exc:
                logger.warning('chatgpt_web vision: type prompt failed: %s', str(exc)[:120])
                stages['prompt_error'] = str(exc)[:120]

            # 7. Click send
            try:
                send_btn = page.locator(
                    'button[data-testid=send-button], '
                    'button[aria-label*="Gửi"], '
                    'button[aria-label*="Send"], '
                    'button:has(svg)'
                ).first
                await send_btn.click(timeout=5_000)
                stages['send_clicked'] = True
            except Exception as exc:
                logger.warning('chatgpt_web vision: send click failed: %s', str(exc)[:120])
                stages['send_error'] = str(exc)[:120]

            # 8. Wait for assistant response
            remaining = timeout - (time.time() - started)
            wait_secs = max(10, min(remaining, 120))
            await asyncio.sleep(wait_secs)

            # 9. Extract response text
            reply_text = ''
            try:
                reply_el = page.locator('[data-message-author-role=assistant]').last
                if await reply_el.count() > 0:
                    reply_text = await reply_el.inner_text()
            except Exception:
                pass

            if not reply_text:
                # Try broader selectors
                try:
                    body_text = await page.locator('body').inner_text()
                    # Look for error messages
                    if 'Unable to' in body_text or 'unable to' in body_text:
                        reply_text = f'[ChatGPT could not process: {body_text[:300]}]'
                    else:
                        reply_text = body_text[-2000:] if len(body_text) > 2000 else body_text
                except Exception:
                    pass

            elapsed_ms = int((time.time() - started) * 1000)
            return {
                'profile': profile,
                'prompt': prompt,
                'text': reply_text,
                'elapsed_ms': elapsed_ms,
                'stages': stages,
            }
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
