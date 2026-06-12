import asyncio
import imaplib
import email
import re
import time
import email.utils
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel
from .browser_pool import pool

logger = logging.getLogger(__name__)

class CodexOnboardReq(BaseModel):
    auth_url: str
    github_email: str
    github_password: str
    gmail_email: str
    gmail_app_password: str

async def _fetch_imap_code(gmail_email: str, gmail_app_password: str, since_timestamp: float, target_email: str, max_wait=120) -> Optional[str]:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            def do_imap():
                mail = imaplib.IMAP4_SSL('imap.gmail.com')
                mail.login(gmail_email, gmail_app_password)
                mail.select('inbox')
                # Search for recent unread emails from GitHub or OpenAI
                status, messages = mail.search(None, '(UNSEEN)')
                if not messages or not messages[0]:
                    mail.logout()
                    return None
                
                # Check from newest to oldest
                msg_ids = messages[0].split()[::-1]
                code_match = None
                for msg_id in msg_ids[:5]:  # Check top 5 unread
                    status, data = mail.fetch(msg_id, '(RFC822)')
                    msg = email.message_from_bytes(data[0][1])
                    
                    # Verify the email is actually new
                    date_header = msg.get('Date')
                    if date_header:
                        try:
                            msg_date = email.utils.parsedate_to_datetime(date_header)
                            if msg_date.timestamp() < since_timestamp - 30: # 30s clock skew tolerance
                                continue
                        except Exception:
                            pass

                    sender = str(msg.get('From', '')).lower()
                    if 'github.com' not in sender and 'openai.com' not in sender:
                        continue
                    
                    content = ''
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == 'text/plain':
                                content = part.get_payload(decode=True).decode()
                                break
                    else:
                        content = msg.get_payload(decode=True).decode()
                    
                    # Verify that the email is intended for the target_email
                    # Forwarded emails usually keep the target_email in the To header or body
                    if target_email.lower() not in str(msg).lower():
                        continue
                    
                    match = re.search(r'\b\d{6}\b', content)
                    if match:
                        code_match = match.group(0)
                        break
                
                mail.logout()
                return code_match
            
            code = await asyncio.to_thread(do_imap)
            if code:
                return code
        except Exception as e:
            logger.warning('IMAP Error: %s', str(e))
        await asyncio.sleep(5)
    return None

async def run_codex_onboard(req: CodexOnboardReq) -> dict[str, Any]:
    profile = f'codex-{req.github_email.split("@")[0]}'
    ctx = await pool.get(profile=profile, headless=False, force_recreate=True)
    pages = ctx.pages
    page = pages[0] if pages else await ctx.new_page()

    try:
        await page.goto(req.auth_url, wait_until='domcontentloaded', timeout=60000)
        
        # Give it a moment to redirect
        await asyncio.sleep(3.0)

        # Handle OpenAI login screen if presented
        if 'auth.openai.com' in page.url:
            is_microsoft = 'outlook.com' in req.github_email.lower() or 'hotmail.com' in req.github_email.lower()
            try:
                if is_microsoft:
                    logger.info('At OpenAI login screen, attempting Passwordless Email Flow...')
                    
                    # If it's the "Welcome Back" screen, check if our account is already in the list!
                    content = await page.content()
                    if 'Chào mừng trở lại' in content or 'Welcome back' in content or 'choose-an-account' in page.url or await page.locator('button:has-text("Đăng nhập vào tài khoản khác"), button:has-text("Log in to another account")').count() > 0:
                        # Try to find the account email in the list
                        account_locator = page.locator(f'text="{req.github_email}"')
                        if await account_locator.count() > 0:
                            logger.info(f'Account {req.github_email} is already logged in! Clicking it to bypass 6-digit code...')
                            await account_locator.first.click()
                            await asyncio.sleep(4.0)
                        else:
                            # Account not found in the quick-login list, click "Log in to another account"
                            if await page.locator('button:has-text("Đăng nhập vào tài khoản khác"), button:has-text("Log in to another account")').count() > 0:
                                await page.click('button:has-text("Đăng nhập vào tài khoản khác"), button:has-text("Log in to another account")')
                                await asyncio.sleep(2.0)
                    
                    # Fill email
                    if await page.locator('input[type="email"], input[name="email"]').count() > 0:
                        request_time = time.time()
                        await page.fill('input[type="email"], input[name="email"]', req.github_email)
                        await page.click('button[type="submit"], button:has-text("Tiếp tục"), button:has-text("Continue")')
                        await asyncio.sleep(5.0)
                        
                    # Now we should be on the Verification Code screen
                    content = await page.content()
                    if 'Kiểm tra hộp thư' in content or 'Check your email' in content or 'Check your inbox' in content or await page.locator('input[inputmode="numeric"], input[type="text"]').count() > 0:
                        logger.info('Waiting for OpenAI 6-digit code from Gmail...')
                        code = await _fetch_imap_code(req.gmail_email, req.gmail_app_password, request_time, req.github_email)
                        if not code:
                            return {'state': 'failed', 'error': 'Could not fetch OpenAI code from Gmail IMAP'}
                        
                        logger.info(f'Fetched OpenAI code: {code}')
                        # Fill the code
                        await page.fill('input[inputmode="numeric"], input[type="text"]', code)
                        await page.click('button[type="submit"], button:has-text("Tiếp tục"), button:has-text("Continue")')
                        await asyncio.sleep(5.0)
                else:
                    logger.info('At OpenAI login screen, clicking Continue with GitHub...')
                    btn = page.locator('button:has-text("GitHub"), a:has-text("GitHub"), button:has-text("github")').first
                    await btn.wait_for(state='visible', timeout=15000)
                    await btn.click()
                    await page.wait_for_load_state('domcontentloaded')
                    await asyncio.sleep(3.0)
            except Exception as e:
                logger.warning(f"Could not complete Auth0 flow: {e}")

        # Handle GitHub Login
        if 'github.com/login' in page.url:
            await page.fill('input[name="login"]', req.github_email)
            await page.fill('input[name="password"]', req.github_password)
            await page.click('input[name="commit"]')
            await page.wait_for_load_state('domcontentloaded')
            await asyncio.sleep(2.0)
            
            content = await page.content()
            title = await page.title()
            if 'Device Verification' in content or 'device verification' in title.lower() or await page.locator('#otp').count() > 0:
                logger.info('GitHub requires device verification code')
                request_time = time.time() - 30 # For github, email is triggered on previous step
                code = await _fetch_imap_code(req.gmail_email, req.gmail_app_password, request_time, req.github_email)
                if not code:
                    return {'state': 'failed', 'error': 'Could not fetch verification code from Gmail IMAP'}
                
                await page.fill('#otp', code)
                await asyncio.sleep(1.0)
                if await page.locator('button:has-text("Verify")').count() > 0:
                    await page.click('button:has-text("Verify")')
                await page.wait_for_load_state('domcontentloaded')
                await asyncio.sleep(2.0)

        # Handle Microsoft Login
        elif 'login.live.com' in page.url or 'login.microsoftonline.com' in page.url:
            logger.info('Handling Microsoft Login...')
            if await page.locator('input[type="email"]').count() > 0:
                await page.fill('input[type="email"]', req.github_email)
                await page.click('input[type="submit"], button:has-text("Next")')
                await asyncio.sleep(2.0)
            
            if await page.locator('input[type="password"]').count() > 0:
                await page.fill('input[type="password"]', req.github_password)
                await page.click('input[type="submit"], button:has-text("Sign in")')
                await asyncio.sleep(2.0)
                
            # Handle "Stay signed in?"
            if await page.locator('input[type="submit"][value="Yes"], button:has-text("Yes")').count() > 0:
                await page.click('input[type="submit"][value="Yes"], button:has-text("Yes")')
                await asyncio.sleep(2.0)

        # Authorize App (GitHub or OpenAI Consent)
        title = await page.title()
        content = await page.content()
        # GitHub authorize
        if 'Authorize' in title or await page.locator('button:has-text("Authorize")').count() > 0 or await page.locator('button[name="authorize"]').count() > 0:
            auth_btn = page.locator('button:has-text("Authorize"), button[name="authorize"]').first
            if await auth_btn.count() > 0:
                await auth_btn.click()
                await page.wait_for_load_state('domcontentloaded')
                await asyncio.sleep(2.0)
                
        # OpenAI Codex Consent (Tiếp tục)
        if 'Codex' in content and await page.locator('button:has-text("Tiếp tục"), button:has-text("Continue")').count() > 0:
            logger.info("Clicking Continue on Codex Consent screen...")
            await page.click('button:has-text("Tiếp tục"), button:has-text("Continue")')
            await page.wait_for_load_state('domcontentloaded')
            await asyncio.sleep(2.0)

        # Wait for redirect to localhost with code and state
        deadline = time.time() + 15
        while time.time() < deadline:
            if 'code=' in page.url and 'state=' in page.url:
                break
            await asyncio.sleep(0.5)

        final_url = page.url
        logger.info(f"Codex onboard success, final_url: {final_url}")
        return {'state': 'success', 'redirect_url': final_url}

    except Exception as e:
        logger.exception('Codex onboard error')
        return {'state': 'failed', 'error': str(e)}
    finally:
        await pool.close_profile(profile)
