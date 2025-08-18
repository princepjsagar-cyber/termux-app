import asyncio
import logging
import os
import re
import json
import base64
from datetime import datetime
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    Defaults,
    MessageHandler,
    filters,
)

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore


if load_dotenv is not None:
    load_dotenv()


BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
ADMIN_ID_RAW = os.getenv("ADMIN_ID", "").strip()
ADMIN_ID = int(ADMIN_ID_RAW) if ADMIN_ID_RAW.isdigit() else None


# Configure logging early
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("growth_bot")


def _require_token() -> None:
    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is not set. Provide it via environment variable or a .env file."
        )


def _bold(text: str) -> str:
    return f"<b>{text}</b>"


def _code(text: str) -> str:
    return f"<code>{text}</code>"


def _extract_url_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    url_pattern = r"(?i)\b((?:https?://)?(?:[\w-]+\.)+[\w-]+(?:/[\w\-./?%&=+#]*)?)\b"
    match = re.search(url_pattern, text)
    return match.group(1) if match else None


def _normalize_url(candidate: str) -> Optional[str]:
    if not candidate:
        return None
    candidate = candidate.strip()
    if not candidate:
        return None
    if not candidate.startswith(("http://", "https://")):
        candidate = f"https://{candidate}"
    try:
        parsed = urlparse(candidate)
        if not parsed.scheme or not parsed.netloc:
            return None
        return candidate
    except Exception:
        return None


def _base64url_no_padding(text: str) -> str:
    encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


async def _http_get_json(url: str, *, headers: Optional[Dict[str, str]] = None, timeout_s: float = 10.0) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json(), None
    except Exception as exc:
        return None, str(exc)


async def _http_post_json(url: str, payload: Dict, *, headers: Optional[Dict[str, str]] = None, timeout_s: float = 10.0) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json(), None
    except Exception as exc:
        return None, str(exc)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Handle deep-link referral payloads: /start ref_<inviter_id>
    try:
        if update.message and context.args:
            payload = context.args[0]
            if isinstance(payload, str):
                if payload.startswith("ref_"):
                    inviter_id_str = payload.replace("ref_", "", 1)
                    if inviter_id_str.isdigit():
                        inviter_id = int(inviter_id_str)
                        await credit_referral_if_applicable(inviter_id=inviter_id, new_user_id=update.effective_user.id if update.effective_user else 0)
                elif payload.startswith("camp_"):
                    # camp_<code>_ref_<inviter>
                    try:
                        rest = payload.replace("camp_", "", 1)
                        parts = rest.split("_ref_")
                        if len(parts) == 2:
                            camp_code = parts[0]
                            inviter_id_str = parts[1]
                            if inviter_id_str.isdigit() and camp_code:
                                inviter_id = int(inviter_id_str)
                                await credit_campaign_referral_if_applicable(
                                    campaign_code=camp_code,
                                    inviter_id=inviter_id,
                                    new_user_id=update.effective_user.id if update.effective_user else 0,
                                )
                    except Exception as exc:
                        logger.warning("Failed to process campaign payload: %s", exc)
    except Exception as exc:
        logger.warning("Failed to process referral payload: %s", exc)
    keyboard = [
        [InlineKeyboardButton("🚀 Growth Tips", callback_data="growth_tips")],
        [InlineKeyboardButton("📊 Member Analytics", callback_data="analytics")],
        [InlineKeyboardButton("📣 Promotion Tools", callback_data="promotion")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.message:
        await update.message.reply_text(
            f"🌟 {_bold('Community Growth Bot')} 🌟\n\n"
            "I help grow Telegram communities organically! Choose an option:",
            reply_markup=reply_markup,
        )


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()

    if query.data == "growth_tips":
        await query.edit_message_text(
            "📈 "
            + _bold("Organic Growth Strategies")
            + "\n\n"
            "1. Post daily valuable content\n"
            "2. Use relevant hashtags: #yourniche #telegram\n"
            "3. Collaborate with similar channels\n"
            "4. Run engaging contests and polls\n"
            "5. Share exclusive content for members\n\n"
            "Real communities beat fake numbers every time! 🏆",
        )
    elif query.data == "analytics":
        chat = await context.bot.get_chat(update.effective_chat.id)  # type: ignore[arg-type]
        try:
            member_count = await context.bot.get_chat_member_count(chat.id)
        except TelegramError:
            member_count = "Unavailable"

        username_display = f"@{chat.username}" if getattr(chat, "username", None) else "N/A"

        await query.edit_message_text(
            "📊 "
            + _bold("Channel Analytics")
            + "\n\n"
            f"Chat: {getattr(chat, 'title', 'N/A')}\n"
            f"Members: {member_count}\n"
            f"Username: {username_display}\n\n"
            "Track growth with: /stats",
        )
    elif query.data == "promotion":
        await query.edit_message_text(
            "📣 "
            + _bold("Promotion Toolkit")
            + "\n\n"
            "1. Use /invite to generate member invite link\n"
            "2. Create promotional posts with /createpost\n"
            "3. Schedule content with /schedule\n"
            "4. Analyze best posting times with /analytics\n\n"
            "Organic growth takes time but brings REAL engagement!",
        )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = await context.bot.get_chat(update.effective_chat.id)  # type: ignore[arg-type]
    try:
        member_count = await context.bot.get_chat_member_count(chat.id)
    except TelegramError:
        member_count = "Unavailable"

    if update.message:
        await update.message.reply_text(
            "📈 "
            + _bold("Growth Statistics")
            + "\n\n"
            f"Current members: {member_count}\n"
            "Daily growth: Calculating...\n\n"
            "Tip: Post during peak hours (8-10AM, 6-8PM local time) for maximum reach!",
        )


async def security_tips_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tips = (
        "🔐 "
        + _bold("Web Security Tips (OWASP Top 10)")
        + "\n\n"
        "1. Validate and sanitize all inputs (avoid injection).\n"
        "2. Enforce strong auth and MFA; protect sessions.\n"
        "3. Use proper access control; deny by default.\n"
        "4. Protect sensitive data (TLS, hashing, key mgmt).\n"
        "5. Handle security misconfig; least privilege.\n"
        "6. Keep dependencies patched (SCA).\n"
        "7. Prevent XSS with output encoding + CSP.\n"
        "8. Use secure deserialization patterns only.\n"
        "9. Log security events and monitor anomalies.\n"
        "10. Use SSRF-safe network rules and allowlists."
    )
    if update.message:
        await update.message.reply_text(tips)


async def _check_with_virustotal(url: str) -> Optional[str]:
    api_key = os.getenv("VT_API_KEY", "").strip()
    if not api_key:
        return None
    headers = {"x-apikey": api_key}
    url_id = _base64url_no_padding(url)
    data, err = await _http_get_json(f"https://www.virustotal.com/api/v3/urls/{url_id}", headers=headers)
    if err or not data:
        return None
    attrs = data.get("data", {}).get("attributes", {})
    stats = attrs.get("last_analysis_stats", {})
    malicious = stats.get("malicious", 0)
    suspicious = stats.get("suspicious", 0)
    harmless = stats.get("harmless", 0)
    undetected = stats.get("undetected", 0)
    total = sum([malicious, suspicious, harmless, undetected]) or 1
    return (
        f"VirusTotal: malicious={malicious}, suspicious={suspicious}, harmless={harmless}, undetected={undetected} (total={total})"
    )


async def _check_with_safe_browsing(url: str) -> Optional[str]:
    api_key = os.getenv("GSB_API_KEY", "").strip()
    if not api_key:
        return None
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
    payload = {
        "client": {"clientId": "growth-bot", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    data, err = await _http_post_json(endpoint, payload)
    if err:
        return None
    matches = data.get("matches", []) if data else []
    if not matches:
        return "Google Safe Browsing: no threats found"
    kinds = sorted({m.get("threatType", "UNKNOWN") for m in matches})
    return "Google Safe Browsing: threats found - " + ", ".join(kinds)


async def scan_url_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text if update.message else ""
    provided = _extract_url_from_text(text)
    url = _normalize_url(provided or (context.args[0] if context.args else ""))
    if not url:
        if update.message:
            await update.message.reply_text("Usage: /scanurl <url>")
        return
    vt = await _check_with_virustotal(url)
    gsb = await _check_with_safe_browsing(url)
    lines = [
        "🔎 " + _bold("URL Safety Check") + "\n",
        f"URL: {url}",
    ]
    if vt:
        lines.append(vt)
    if gsb:
        lines.append(gsb)
    if len(lines) == 2:
        lines.append("No reputation services configured. Set VT_API_KEY or GSB_API_KEY.")
    if update.message:
        await update.message.reply_text("\n".join(lines))


def _header_status(headers: Dict[str, str]) -> Tuple[str, str]:
    # Normalize header keys to lowercase
    lc = {k.lower(): v for k, v in headers.items()}
    findings: Dict[str, str] = {}

    # Check common security headers
    checks = {
        "content-security-policy": "Add a strict CSP to mitigate XSS (e.g., default-src 'self');",
        "strict-transport-security": "Enable HSTS to enforce HTTPS; include subdomains & preload;",
        "x-content-type-options": "Set X-Content-Type-Options: nosniff;",
        "x-frame-options": "Set X-Frame-Options: DENY or SAMEORIGIN;",
        "referrer-policy": "Set a strict Referrer-Policy (e.g., no-referrer or strict-origin-when-cross-origin);",
        "permissions-policy": "Configure Permissions-Policy to limit powerful features;",
        "cross-origin-opener-policy": "Set COOP to same-origin for isolation;",
        "cross-origin-resource-policy": "Set CORP (e.g., same-site) to protect resources;",
        "cross-origin-embedder-policy": "Consider COEP for stronger isolation;",
    }
    for key, advice in checks.items():
        if key not in lc:
            findings[key] = f"Missing: {advice}"
    summary_lines = []
    for key in checks.keys():
        present = key in lc
        summary_lines.append(f"{key}: {'present' if present else 'missing'}")
    advice_lines = [v for v in findings.values()]
    return "\n".join(summary_lines), ("\n".join(advice_lines) if advice_lines else "All key headers present.")


async def headers_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text if update.message else ""
    provided = _extract_url_from_text(text)
    url = _normalize_url(provided or (context.args[0] if context.args else ""))
    if not url:
        if update.message:
            await update.message.reply_text("Usage: /headers <url>")
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url)
        hdrs = {k: v for k, v in resp.headers.items()}
        summary, advice = _header_status(hdrs)
        preview = "\n".join([f"{k}: {v}" for k, v in hdrs.items() if k.lower() in {"content-security-policy","strict-transport-security","x-content-type-options","x-frame-options","referrer-policy","permissions-policy"}])
        message = (
            "🧾 "
            + _bold("Security Headers Report")
            + "\n\n"
            + f"Target: {url}\n"
            + summary
            + "\n\n"
            + _bold("Recommendations")
            + "\n"
            + advice
            + (f"\n\n" + _bold("Preview") + "\n" + _code(preview) if preview else "")
        )
        if update.message:
            await update.message.reply_text(message)
    except Exception as exc:
        if update.message:
            await update.message.reply_text(f"Could not fetch headers: {exc}")


async def security_news_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Try CISA KEV JSON feed
    urls = [
        "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json.gz",
    ]
    feed = None
    err_msg = None
    for u in urls:
        feed, err_msg = await _http_get_json(u, timeout_s=12.0)
        if feed:
            break
    items = []
    if feed and isinstance(feed, dict) and isinstance(feed.get("vulnerabilities"), list):
        try:
            vulns = feed["vulnerabilities"]
            def parse_date(s: str) -> datetime:
                try:
                    return datetime.fromisoformat(s)
                except Exception:
                    return datetime.min
            vulns.sort(key=lambda v: parse_date(v.get("dateAdded", "")), reverse=True)
            for v in vulns[:5]:
                cve = v.get("cveID", "Unknown")
                name = v.get("vulnerabilityName", "")
                date_added = v.get("dateAdded", "")
                short = v.get("shortDescription", "")
                items.append(f"{cve} - {name} ({date_added})\n{short}")
        except Exception:
            items = []
    if not items:
        note = f"Could not fetch CISA KEV feed: {err_msg or 'unknown error'}"
        items = [note]
    text = "📰 " + _bold("Security News (CISA KEV)") + "\n\n" + "\n\n".join(items)
    if update.message:
        await update.message.reply_text(text)


async def disclose_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "✅ "
        + _bold("Responsible Disclosure Guide")
        + "\n\n"
        "1. Confirm scope and legality; never test without permission.\n"
        "2. Document steps to reproduce safely.\n"
        "3. Contact the vendor/security email; use encryption if offered.\n"
        "4. Share only necessary details; avoid public disclosure until fixed.\n"
        "5. Agree on a remediation timeline; coordinate updates.\n\n"
        + _bold("Email Template")
        + "\n"
        "Subject: Security vulnerability report for <project>\n\n"
        "Hello Security Team,\n\n"
        "I would like to report a potential security issue found in <project>.\n"
        "Summary: <brief description>\n"
        "Impact: <affected components/users>\n"
        "Steps to reproduce: <step-by-step>\n"
        "Recommendation: <remediation idea>\n\n"
        "I am happy to provide more details or collaborate on validation.\n\n"
        "Regards,\n<your name>"
    )
    if update.message:
        await update.message.reply_text(msg)


async def invite_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if not chat:
        return

    try:
        invite_link = await context.bot.create_chat_invite_link(
            chat_id=chat.id,
            name="GrowthBot Invite",
        )
    except TelegramError as exc:
        logger.warning("Failed to create invite link: %s", exc)
        if update.message:
            await update.message.reply_text(
                "I could not create an invite link. Ensure the bot is an admin with the right to manage invite links."
            )
        return

    if update.message:
        await update.message.reply_text(
            "🔗 "
            + _bold("Invite New Members")
            + "\n\n"
            "Share this link to grow your community:\n"
            f"{invite_link.invite_link}\n\n"
            "Pro Tip: Offer value to new members for higher conversion!",
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.new_chat_members:
        for member in update.message.new_chat_members:
            welcome_msg = (
                f"👋 Welcome {member.full_name}!\n\n"
                "We're excited to have you here! "
                "Be sure to:\n"
                "1. Introduce yourself\n"
                "2. Check pinned messages\n"
                "3. Engage with our community\n\n"
                "Enjoy your stay! 🎉"
            )
            await update.message.reply_text(welcome_msg)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Exception while handling an update: %s", context.error)


def build_application() -> Application:
    _require_token()
    defaults = Defaults(parse_mode=ParseMode.HTML)
    return Application.builder().token(BOT_TOKEN).defaults(defaults).build()


# --- Simple JSON-file based referral store (demo-grade) ---
REFERRAL_STORE_PATH = os.path.join(os.path.dirname(__file__), "referrals.json")
CAMPAIGN_STORE_PATH = os.path.join(os.path.dirname(__file__), "campaigns.json")
PREFS_STORE_PATH = os.path.join(os.path.dirname(__file__), "prefs.json")


def _load_referrals() -> Dict[str, Dict[str, int]]:
    try:
        with open(REFERRAL_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_referrals(data: Dict[str, Dict[str, int]]) -> None:
    tmp = REFERRAL_STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, REFERRAL_STORE_PATH)


def _load_campaigns() -> Dict:
    try:
        with open(CAMPAIGN_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"campaigns": {}, "referrals": {}}


def _save_campaigns(data: Dict) -> None:
    tmp = CAMPAIGN_STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CAMPAIGN_STORE_PATH)


def _load_prefs() -> Dict[str, Dict[str, str]]:
    try:
        with open(PREFS_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_prefs(data: Dict[str, Dict[str, str]]) -> None:
    tmp = PREFS_STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PREFS_STORE_PATH)


async def credit_referral_if_applicable(inviter_id: int, new_user_id: int) -> None:
    if inviter_id <= 0 or new_user_id <= 0 or inviter_id == new_user_id:
        return
    data = _load_referrals()
    inviter_key = str(inviter_id)
    invited_set_key = f"invited_{inviter_key}"
    invited = set(data.get(invited_set_key, {}))  # keys only
    if str(new_user_id) in invited:
        return
    # Update inviter stats
    stats = data.get(inviter_key, {"count": 0})
    stats["count"] = int(stats.get("count", 0)) + 1
    data[inviter_key] = stats
    # Remember this invitee so we don't count twice
    invited.add(str(new_user_id))
    data[invited_set_key] = {k: 1 for k in invited}
    _save_referrals(data)


async def credit_campaign_referral_if_applicable(campaign_code: str, inviter_id: int, new_user_id: int) -> None:
    if not campaign_code or inviter_id <= 0 or new_user_id <= 0 or inviter_id == new_user_id:
        return
    data = _load_campaigns()
    refs = data.setdefault("referrals", {}).setdefault(campaign_code, {})
    inviter_key = str(inviter_id)
    invited_set_key = f"invited_{inviter_key}"
    invited = set(refs.get(invited_set_key, {}))
    if str(new_user_id) in invited:
        return
    stats = refs.get(inviter_key, {"count": 0})
    stats["count"] = int(stats.get("count", 0)) + 1
    refs[inviter_key] = stats
    invited.add(str(new_user_id))
    refs[invited_set_key] = {k: 1 for k in invited}
    _save_campaigns(data)


async def referral_link_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not update.message:
        return
    # Deep link format: https://t.me/<bot_username>?start=ref_<user_id>
    try:
        me = await context.bot.get_me()
        bot_username = me.username
    except Exception:
        bot_username = None
    if not bot_username:
        await update.message.reply_text("Could not determine bot username.")
        return
    link = f"https://t.me/{bot_username}?start=ref_{user.id}"
    await update.message.reply_text(
        "🔗 "
        + _bold("Your referral link")
        + "\n\nShare this to invite friends.\n"
        + link
    )


async def campaign_link_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: /campaign <code>")
        return
    camp_code = context.args[0].strip()
    if not camp_code:
        await update.message.reply_text("Usage: /campaign <code>")
        return
    try:
        me = await context.bot.get_me()
        bot_username = me.username
    except Exception:
        bot_username = None
    if not bot_username:
        await update.message.reply_text("Could not determine bot username.")
        return
    link = f"https://t.me/{bot_username}?start=camp_{camp_code}_ref_{user.id}"
    await update.message.reply_text(
        "🎯 "
        + _bold("Campaign referral link")
        + f"\n\nCampaign: {camp_code}\n"
        + link
    )


async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    data = _load_referrals()
    entries = [(int(uid), v.get("count", 0)) for uid, v in data.items() if uid.isdigit()]
    entries.sort(key=lambda x: x[1], reverse=True)
    top = entries[:10]
    if not top:
        await update.message.reply_text("No referrals yet. Use /referral to get your link.")
        return
    lines = ["🏆 " + _bold("Referral Leaderboard") + "\n"]
    for rank, (uid, count) in enumerate(top, start=1):
        lines.append(f"{rank}. User {uid}: {count}")
    await update.message.reply_text("\n".join(lines))


async def campaign_leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if not context.args:
        await update.message.reply_text("Usage: /campaignboard <code>")
        return
    camp_code = context.args[0].strip()
    if not camp_code:
        await update.message.reply_text("Usage: /campaignboard <code>")
        return
    data = _load_campaigns()
    refs = data.get("referrals", {}).get(camp_code, {})
    entries = [(int(uid), v.get("count", 0)) for uid, v in refs.items() if uid.isdigit()]
    entries.sort(key=lambda x: x[1], reverse=True)
    top = entries[:10]
    if not top:
        await update.message.reply_text("No referrals yet for this campaign.")
        return
    lines = ["🏁 " + _bold(f"Campaign Leaderboard: {camp_code}") + "\n"]
    for rank, (uid, count) in enumerate(top, start=1):
        lines.append(f"{rank}. User {uid}: {count}")
    await update.message.reply_text("\n".join(lines))


async def setlang_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    if not context.args:
        await update.message.reply_text("Usage: /setlang <en|es|hi|ar|fr|de>")
        return
    lang = context.args[0].strip().lower()
    if lang not in {"en", "es", "hi", "ar", "fr", "de"}:
        await update.message.reply_text("Unsupported language. Use: en, es, hi, ar, fr, de")
        return
    prefs = _load_prefs()
    prefs[str(update.effective_user.id)] = {"lang": lang}
    _save_prefs(prefs)
    await update.message.reply_text(f"Language set to {lang}.")


async def verifyjoin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not context.args:
        if update.message:
            await update.message.reply_text("Usage: /verifyjoin <channel_username>")
        return
    channel = context.args[0].strip()
    if channel.startswith("@"):  # normalize
        channel = channel
    else:
        channel = f"@{channel}"
    try:
        member = await context.bot.get_chat_member(chat_id=channel, user_id=update.effective_user.id)  # type: ignore[arg-type]
        status = getattr(member, "status", "unknown")
        if status in {"creator", "administrator", "member"}:
            await update.message.reply_text("✅ You are a member. Thanks!")
        else:
            await update.message.reply_text("❌ Not a member yet. Please join and try again.")
    except Exception as exc:
        await update.message.reply_text(f"Could not verify membership: {exc}")


def main() -> None:
    application = build_application()

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("invite", invite_command))
    application.add_handler(CommandHandler("securitytips", security_tips_command))
    application.add_handler(CommandHandler("scanurl", scan_url_command))
    application.add_handler(CommandHandler("headers", headers_command))
    application.add_handler(CommandHandler("securitynews", security_news_command))
    application.add_handler(CommandHandler("disclose", disclose_command))
    application.add_handler(CommandHandler("referral", referral_link_command))
    application.add_handler(CommandHandler("leaderboard", leaderboard_command))
    application.add_handler(CommandHandler("campaign", campaign_link_command))
    application.add_handler(CommandHandler("campaignboard", campaign_leaderboard_command))
    application.add_handler(CommandHandler("setlang", setlang_command))
    application.add_handler(CommandHandler("verifyjoin", verifyjoin_command))

    # Buttons
    application.add_handler(CallbackQueryHandler(handle_button))

    # New member welcome
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_message))

    # Error logging
    application.add_error_handler(error_handler)

    logger.info("Starting legitimate growth bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

