import os
import io
import base64
import asyncio
import logging
import json
import hashlib
import time
from typing import List, Dict, Any
from telegram import Update, InputFile, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    AIORateLimiter,
    MessageHandler,
    filters,
    InlineQueryHandler,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

BOT_TOKEN = os.environ.get("BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to Neon Bot! I can help you with various tasks. "
        "Try /help to see what I can do."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    lang = (getattr(user, "language_code", "") or "").lower()
    is_hi = lang.startswith("hi")

    if is_hi:
        user_help = (
            "उपलब्ध कमांड्स:\n\n"
            "मुख्य:\n"
            "/start — स्वागत संदेश\n"
            "/help, /commands — यह सहायता\n"
            "/status, /health — बॉट की स्थिति\n\n"
            "AI:\n"
            "/ai <प्रॉम्प्ट> — AI चैट (स्ट्रीमिंग)\n"
            "/ask <सवाल> — ताज़ा वेब संदर्भ + उत्तर\n"
            "/img <प्रॉम्प्ट> — इमेज जनरेशन (Gemini)\n\n"
            "खोज और समाचार:\n"
            "/web <क्वेरी> — वेब सर्च\n"
            "/news [SYMS] — ताज़ा सुर्खियाँ\n"
            "/subscribe_news [SYMS] [मिनट] — नियमित अपडेट\n"
            "/unsubscribe_news — अपडेट बंद\n"
            "/newsportal — न्यूज़ पोर्टल लिंक\n\n"
            "आवाज़ और विज़न:\n"
            "/tts <टेक्स्ट> — टेक्स्ट से आवाज़\n"
            "/ocr — इमेज से टेक्स्ट निकालें\n\n"
            "पर्सोना:\n"
            "/persona set <नाम> | /persona list — मोड बदलें\n"
        )
        admin_extra = (
            "\nएडमिन/कन्फ़िग:\n"
            "/feature <name> on|off — फीचर टॉगल\n"
            "/analytics — उपयोग काउंटर\n"
            "/setkey <NAME> <VALUE> — API की/कन्फ़िग सेट करें\n"
        )
    else:
        user_help = (
            "Available commands:\n\n"
            "Core:\n"
            "/start — Welcome message\n"
            "/help, /commands — This help\n"
            "/status, /health — Bot status & liveness\n\n"
            "AI:\n"
            "/ai <prompt> — Chat with AI (streaming)\n"
            "/ask <question> — Fresh web context + answer\n"
            "/img <prompt> — Image generation (Gemini)\n\n"
            "Search & News:\n"
            "/web <query> — Web search (CSE→Tavily→SerpAPI)\n"
            "/news [SYMS] — Latest headlines (e.g., AAPL,TSLA)\n"
            "/subscribe_news [SYMS] [minutes] — Push updates\n"
            "/unsubscribe_news — Stop pushes\n"
            "/newsportal — Portal link\n\n"
            "Voice & Vision:\n"
            "/tts <text> — Text to speech\n"
            "/ocr — Send/reply with an image to extract text\n\n"
            "Personalization:\n"
            "/persona set <name> | /persona list — Persona modes\n"
        )
        admin_extra = (
            "\nAdmin & Config:\n"
            "/feature <name> on|off — Toggle features\n"
            "/analytics — Usage counters\n"
            "/setkey <NAME> <VALUE> — Set API keys/config\n"
        )

    text = user_help + admin_extra
    await update.message.reply_text(text)


async def setkey_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Please provide a key: /setkey <name> <value>")
        return
    if len(args) < 2:
        await update.message.reply_text("Usage: /setkey <OPENAI_API_KEY|TAVILY_API_KEY|SERPAPI_KEY|GOOGLE_CSE_API_KEY|GOOGLE_CSE_CX|GEMINI_API_KEY|NEWS_API_KEY|NGROK_AUTHTOKEN|BOT_DATA_KEY> <value>")
        return
    name = args[0].strip().upper()
    value = " ".join(args[1:]).strip()
    allowed = {"OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_KEY", "GOOGLE_CSE_API_KEY", "GOOGLE_CSE_CX", "GEMINI_API_KEY", "NEWS_API_KEY", "NGROK_AUTHTOKEN", "BOT_DATA_KEY"}
    if name not in allowed:
        await update.message.reply_text("Key must be one of: " + ", ".join(sorted(allowed)))
        return
    context.application.bot_data[name] = value
    await update.message.reply_text("✅ Set " + name + " (in-memory)")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bot Status:\n"
        f"• Token: {'Set' if BOT_TOKEN else 'Not Set'}\n"
        f"• Data Key: {'Set' if context.application.bot_data.get('BOT_DATA_KEY') else 'Not Set'}\n"
        f"• Store Path: {_get_store_path()}\n"
        f"• OPENAI_API_KEY: {'Set' if _get_runtime_key(context, 'OPENAI_API_KEY') else 'Not Set'}\n"
        f"• TAVILY_API_KEY: {'Set' if _get_runtime_key(context, 'TAVILY_API_KEY') else 'Not Set'}\n"
        f"• SERPAPI_KEY: {'Set' if _get_runtime_key(context, 'SERPAPI_KEY') else 'Not Set'}\n"
        f"• GOOGLE_CSE_API_KEY: {'Set' if _get_runtime_key(context, 'GOOGLE_CSE_API_KEY') else 'Not Set'}\n"
        f"• GOOGLE_CSE_CX: {'Set' if _get_runtime_key(context, 'GOOGLE_CSE_CX') else 'Not Set'}\n"
        f"• GEMINI_API_KEY: {'Set' if _get_runtime_key(context, 'GEMINI_API_KEY') else 'Not Set'}\n"
        f"• NEWS_API_KEY: {'Set' if _get_runtime_key(context, 'NEWS_API_KEY') else 'Not Set'}\n"
        f"• NGROK_AUTHTOKEN: {'Set' if _get_runtime_key(context, 'NGROK_AUTHTOKEN') else 'Not Set'}"
    )


async def ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /ai <prompt>")
        return
    await handle_ai_chat(update, context, prompt)


async def img_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    gkey = get_gemini_key(context)
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /img <prompt>")
        return
    if not gkey:
        await update.message.reply_text("Set GEMINI_API_KEY with /setkey GEMINI_API_KEY <key>.")
        return

    try:
        import google.generativeai as genai
        genai.configure(api_key=gkey)
        model_name = os.environ.get("GEMINI_IMAGE_MODEL", "imagen-3.0-generate-001")
        size = os.environ.get("GEMINI_IMAGE_SIZE", "1024x1024")
        # Try official image generation method first
        try:
            model = genai.GenerativeModel(model_name)
            result = model.generate_images(prompt=prompt, number_of_images=1, size=size)
        except Exception:
            # Fallback to generate_content for older SDKs
            model = genai.GenerativeModel(model_name)
            result = model.generate_content(prompt)

        b64 = None
        # Extract image base64 from various possible result shapes
        if hasattr(result, "generated_images") and result.generated_images:
            gi = result.generated_images[0]
            if hasattr(gi, "image"):
                b64 = getattr(gi.image, "base64_data", None) or getattr(gi.image, "data", None)
            b64 = b64 or getattr(gi, "base64_image", None)
        if not b64 and hasattr(result, "images") and result.images:
            img = result.images[0]
            b64 = getattr(img, "base64_data", None) or getattr(img, "data", None)
        if not b64 and hasattr(result, "candidates"):
            for cand in (result.candidates or []):
                for part in getattr(cand.content, "parts", []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        b64 = inline.data
                        break
                if b64:
                    break

        if not b64:
            await update.message.reply_text("❌ Gemini did not return an image.")
            return

        raw = base64.b64decode(b64)
        bio = io.BytesIO(raw)
        bio.name = "image.png"
        await update.message.reply_photo(photo=InputFile(bio, filename="image.png"), caption=f"🎨 {prompt[:200]}")
    except Exception as e:
        logging.exception("Gemini image generation error: %s", e)
        await update.message.reply_text("❌ Image generation failed.")


async def web_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import httpx
    tc = get_tavily_client(context)
    serpapi_key = get_serpapi_key(context)
    gkey = get_google_cse_key(context)
    gcx = get_google_cse_cx(context)
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /web <query>")
        return
    try:
        if gkey and gcx:
            params = {"key": gkey, "cx": gcx, "q": query, "num": 5}
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get("https://www.googleapis.com/customsearch/v1", params=params)
                resp.raise_for_status()
                data = resp.json()
            items = data.get("items", [])
            lines = []
            for item in items[:5]:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                lines.append(f"- {title}\n{snippet}\n{link}")
            body = "\n\n".join(lines) or "No results."
            await update.message.reply_text(f"🔎 {query}\n\n{body}"[:4096], disable_web_page_preview=True)
            return
        if tc:
            results = tc.search(query=query, search_depth="advanced", max_results=5)
            sources = results.get("results", [])
            summary = results.get("answer", "") or results.get("raw_content", "")
            if not summary:
                summary = "\n".join([s.get("content", "") for s in sources])[:1500]
            text = f"🔎 {query}\n\n{summary[:3500]}\n\n" + "\n".join([f"- {s.get('title','')}" for s in sources[:5]])
            await update.message.reply_text(text[:4096], disable_web_page_preview=True)
            return
        if serpapi_key:
            params = {"engine": "google", "q": query, "api_key": serpapi_key, "num": "5"}
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get("https://serpapi.com/search.json", params=params)
                resp.raise_for_status()
                data = resp.json()
            organic = data.get("organic_results", [])
            lines = []
            for item in organic[:5]:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                lines.append(f"- {title}\n{snippet}\n{link}")
            body = "\n\n".join(lines) or "No results."
            await update.message.reply_text(f"🔎 {query}\n\n{body}"[:4096], disable_web_page_preview=True)
            return
        await update.message.reply_text("Set GOOGLE_CSE_API_KEY+GOOGLE_CSE_CX or TAVILY_API_KEY or SERPAPI_KEY to enable web search.")
    except Exception as e:
        logging.exception("Web search error: %s", e)
        await update.message.reply_text("❌ Web search failed.")


async def code_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please provide a description for code generation.")
    context.user_data["code_description"] = update.message.text
    await update.message.reply_text("I'm ready to generate! Type /code again to continue.")


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file_id = photo.file_id
    file = await context.bot.get_file(file_id)
    file_bytes = await file.download_as_bytearray()
    bio = io.BytesIO(file_bytes)
    bio.name = "photo.jpg"
    try:
        # Assuming a placeholder for image generation logic
        # In a real bot, you'd use a DALL-E client here
        await update.message.reply_text("🖼️ Image generation is not yet implemented.")
        # await update.message.reply_photo(InputFile(bio)) # Uncomment to send photo
    except Exception as e:
        logging.exception("Photo handler error: %s", e)
        await update.message.reply_text("❌ Failed to process photo.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error("Exception while handling update:", exc_info=context.error)
    if update and hasattr(update, "message") and update.message:
        try:
            await update.message.reply_text("⚠️ An error occurred. Please try again later.")
        except Exception:
            pass

# --------- Secure persistent store (encrypted, opt-in) ---------
def _get_store_path() -> str:
    return os.environ.get("BOT_STORE_PATH", "/workspace/.neon_store.enc")


def _get_data_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "BOT_DATA_KEY")


def _derive_fernet(key_str: str):
    from cryptography.fernet import Fernet
    # Derive a stable 32-byte key from any passphrase
    digest = hashlib.sha256(key_str.encode()).digest()
    fkey = base64.urlsafe_b64encode(digest)
    return Fernet(fkey)


def _ensure_store_root(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    app_data = context.application.bot_data
    if "STORE" not in app_data or not isinstance(app_data["STORE"], dict):
        app_data["STORE"] = {"users": {}}
    return app_data["STORE"]


def _get_user_entry(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> Dict[str, Any]:
    store = _ensure_store_root(context)
    users = store.setdefault("users", {})
    entry = users.setdefault(str(user_id), {"items": [], "history": []})
    return entry


def load_store_eager(application) -> None:
    try:
        key = os.environ.get("BOT_DATA_KEY", "").strip()
        path = _get_store_path()
        store: Dict[str, Any] = {"users": {}}
        if key and os.path.exists(path):
            with open(path, "rb") as f:
                token = f.read()
            fernet = _derive_fernet(key)
            data = fernet.decrypt(token)
            store = json.loads(data.decode())
        application.bot_data["STORE"] = store
    except Exception as e:
        logging.exception("Failed to load store: %s", e)
        application.bot_data["STORE"] = {"users": {}}


def save_store(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        key = _get_data_key(context)
        if not key:
            return
        store = _ensure_store_root(context)
        data = json.dumps(store).encode()
        fernet = _derive_fernet(key)
        token = fernet.encrypt(data)
        path = _get_store_path()
        with open(path, "wb") as f:
            f.write(token)
    except Exception as e:
        logging.exception("Failed to save store: %s", e)


async def periodic_save_job(context: ContextTypes.DEFAULT_TYPE):
    save_store(context)


def _get_runtime_key(context: ContextTypes.DEFAULT_TYPE, name: str) -> str:
    try:
        app_data = context.application.bot_data
        val = app_data.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def append_user_message(history: List[Dict[str, Any]], text: str) -> None:
    history.append({"role": "user", "content": text})


def get_openai_client(context: ContextTypes.DEFAULT_TYPE):
    try:
        from openai import OpenAI
    except Exception:
        return None
    api_key = _get_runtime_key(context, "OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_tavily_client(context: ContextTypes.DEFAULT_TYPE):
    try:
        from tavily import TavilyClient
    except Exception:
        return None
    api_key = _get_runtime_key(context, "TAVILY_API_KEY")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


def get_serpapi_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "SERPAPI_KEY")

def get_google_cse_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "GOOGLE_CSE_API_KEY")

def get_google_cse_cx(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "GOOGLE_CSE_CX")

def get_gemini_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "GEMINI_API_KEY")

def get_news_api_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    return _get_runtime_key(context, "NEWS_API_KEY")

def get_news_endpoint() -> str:
    return os.environ.get("NEWS_API_ENDPOINT", "https://api.example-news.io/v3/realtime").strip()

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import httpx
    api_key = get_news_api_key(context)
    if not api_key:
        await update.message.reply_text("Set NEWS_API_KEY with /setkey NEWS_API_KEY <key>.")
        return
    # Symbols can be provided as args or from env; default AAPL,MSFT
    symbols_arg = "".join(context.args).strip() if context.args else ""
    if symbols_arg:
        symbols = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    else:
        symbols = [s.strip().upper() for s in os.environ.get("NEWS_SYMBOLS", "AAPL,MSFT").split(",") if s.strip()]
    limit = int(os.environ.get("NEWS_MAX_RESULTS", "8") or 8)
    params = {"symbols": ",".join(symbols), "limit": str(limit), "sort": "newest"}
    headers = {"X-Api-Key": api_key}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(get_news_endpoint(), params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        # Accept both shapes: {articles: [...]} or {news: [...]} or direct list
        items = []
        if isinstance(data, dict):
            items = data.get("articles") or data.get("news") or []
        elif isinstance(data, list):
            items = data
        if not items:
            await update.message.reply_text("No news found.")
            return
        lines = []
        for it in items[:limit]:
            title = (it.get("title") if isinstance(it, dict) else str(it)) or "(untitled)"
            url = (it.get("url") if isinstance(it, dict) else "") or ""
            src = ""
            if isinstance(it, dict):
                src_obj = it.get("source") or {}
                if isinstance(src_obj, dict):
                    src = src_obj.get("name", "")
                elif isinstance(src_obj, str):
                    src = src_obj
            line = f"• {title}"
            if src:
                line += f" — {src}"
            if url:
                line += f"\n{url}"
            lines.append(line)
        text = "\n\n".join(lines)
        await update.message.reply_text(text[:4096], disable_web_page_preview=False)
    except Exception as e:
        logging.exception("News fetch error: %s", e)
        await update.message.reply_text("❌ Failed to fetch news.")


async def subscribe_news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Usage: /subscribe_news [AAPL,TSLA] [minutes]
    symbols = None
    interval_min = 30
    if context.args:
        parts = " ".join(context.args).split()
        if parts:
            if "," in parts[0] or parts[0].isalpha():
                symbols = [s.strip().upper() for s in parts[0].split(",") if s.strip()]
                parts = parts[1:]
        if parts:
            try:
                interval_min = max(5, int(parts[0]))
            except Exception:
                pass
    entry = _get_user_entry(context, update.effective_user.id)
    entry.setdefault("news_sub", {})
    if symbols:
        entry["news_sub"]["symbols"] = symbols
    entry["news_sub"]["interval_min"] = interval_min
    save_store(context)
    await update.message.reply_text(f"✅ Subscribed to news every {interval_min}m for symbols: {','.join(symbols or entry['news_sub'].get('symbols', ['AAPL','MSFT']))}")


async def unsubscribe_news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entry = _get_user_entry(context, update.effective_user.id)
    if "news_sub" in entry:
        entry.pop("news_sub", None)
        save_store(context)
        await update.message.reply_text("🛑 Unsubscribed from news updates.")
    else:
        await update.message.reply_text("You are not subscribed.")


async def _push_news_job(context: ContextTypes.DEFAULT_TYPE):
    import httpx
    app = context.application
    store = _ensure_store_root(context)
    api_key = _get_runtime_key(context, "NEWS_API_KEY")
    if not api_key:
        return
    for uid, udata in store.get("users", {}).items():
        sub = udata.get("news_sub")
        if not sub:
            continue
        symbols = sub.get("symbols") or [s.strip().upper() for s in os.environ.get("NEWS_SYMBOLS", "AAPL,MSFT").split(",") if s.strip()]
        limit = int(os.environ.get("NEWS_MAX_RESULTS", "5") or 5)
        params = {"symbols": ",".join(symbols), "limit": str(limit), "sort": "newest"}
        headers = {"X-Api-Key": api_key}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(get_news_endpoint(), params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            items = []
            if isinstance(data, dict):
                items = data.get("articles") or data.get("news") or []
            elif isinstance(data, list):
                items = data
            if not items:
                continue
            first = items[0]
            title = first.get("title", "(untitled)") if isinstance(first, dict) else str(first)
            url = (first.get("url") if isinstance(first, dict) else "") or ""
            msg = f"📰 Latest: {title}\n{url}" if url else f"📰 Latest: {title}"
            try:
                await app.bot.send_message(chat_id=int(uid), text=msg)
            except Exception:
                pass
        except Exception:
            continue


def ensure_history(context: ContextTypes.DEFAULT_TYPE) -> List[Dict[str, Any]]:
    user_data = context.user_data
    if "history" not in user_data:
        user_data["history"] = []
    # Trim to last 10 exchanges
    if len(user_data["history"]) > 20:
        user_data["history"] = user_data["history"][-20:]
    return user_data["history"]


def _aggregate_web_content(context: ContextTypes.DEFAULT_TYPE, query: str) -> str:
    import httpx
    gkey = get_google_cse_key(context)
    gcx = get_google_cse_cx(context)
    tc = get_tavily_client(context)
    serpapi_key = get_serpapi_key(context)
    try:
        if gkey and gcx:
            params = {"key": gkey, "cx": gcx, "q": query, "num": 5}
            with httpx.Client(timeout=15) as client:
                resp = client.get("https://www.googleapis.com/customsearch/v1", params=params)
                resp.raise_for_status()
                data = resp.json()
            items = data.get("items", [])
            snippets = []
            for it in items[:5]:
                title = it.get("title", "")
                snippet = it.get("snippet", "")
                link = it.get("link", "")
                snippets.append(f"{title}\n{snippet}\n{link}")
            return "\n\n".join(snippets)
        if tc:
            results = tc.search(query=query, search_depth="advanced", max_results=5)
            sources = results.get("results", [])
            if sources:
                return "\n\n".join([s.get("content", "") for s in sources[:5]])
        if serpapi_key:
            params = {"engine": "google", "q": query, "api_key": serpapi_key, "num": "5"}
            with httpx.Client(timeout=15) as client:
                resp = client.get("https://serpapi.com/search.json", params=params)
                resp.raise_for_status()
                data = resp.json()
            organic = data.get("organic_results", [])
            lines = []
            for it in organic[:5]:
                title = it.get("title", "")
                snippet = it.get("snippet", "")
                link = it.get("link", "")
                lines.append(f"{title}\n{snippet}\n{link}")
            return "\n\n".join(lines)
    except Exception:
        pass
    return ""


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _rate_limit_ok(update.effective_user.id):
        return
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /ask <question>")
        return
    if not _ask_quota_ok(context, update.effective_user.id):
        await update.message.reply_text("Daily /ask quota reached. Try tomorrow.")
        return
    flags = _get_flags(context)
    key = f"ask::{prompt.strip().lower()}"
    if flags.get("ask_cache"):
        cached = _cache_get(context, key)
        if cached:
            await update.message.reply_text(cached[:4096])
            _inc_usage(context, "ask_cache_hit")
            return
    fresh_context = _aggregate_web_content(context, prompt)
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY (and optionally web keys) to enable /ask.")
        return
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You answer with up-to-date info. Prefer the provided snippets; if unsure, say so."},
    ]
    if fresh_context:
        messages.append({"role": "system", "content": f"Snippets:\n{fresh_context}"})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        if flags.get("ask_cache"):
            _cache_set(context, key, text)
        await update.message.reply_text(text[:4096])
        _inc_usage(context, "ask")
    except Exception as e:
        logging.exception("/ask error: %s", e)
        await update.message.reply_text("❌ Failed to answer.")

# Hook moderation and persona in AI chat
async def handle_ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    if not _rate_limit_ok(update.effective_user.id):
        return
    if not await _moderate_if_enabled(context, prompt):
        await update.message.reply_text("❌ Content blocked by moderation.")
        return
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY to enable AI chat.")
        return

    # Persona
    system_prompt = "You are a helpful assistant."
    try:
        persona = _get_user_entry(context, update.effective_user.id).get("persona")
        if persona == "tutor":
            system_prompt = "You are a patient tutor who explains step by step."
        elif persona == "coder":
            system_prompt = "You are a senior software engineer; respond with clear code-first solutions."
        elif persona == "analyst":
            system_prompt = "You are a data analyst; use bullet points and numbers."
        elif persona == "friendly":
            system_prompt = "You are friendly and concise."
    except Exception:
        pass

    try:
        user_id = update.effective_user.id
        entry = _get_user_entry(context, user_id)
        history = entry.setdefault("history", [])
        if len(history) > 20:
            entry["history"] = history[-20:]
            history = entry["history"]
    except Exception:
        history = ensure_history(context)

    # Prepend system
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": prompt}]

    placeholder = await update.message.reply_text("🤖 Thinking…")
    content_chunks: List[str] = []

    try:
        stream = client.chat.completions.create(
            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=messages,
            stream=True,
            temperature=0.4,
        )

        last_edit = ""
        async def periodic_edit():
            nonlocal last_edit
            while True:
                await asyncio.sleep(0.6)
                joined = "".join(content_chunks).strip()
                if joined and joined != last_edit:
                    try:
                        await placeholder.edit_text(joined[:4096])
                        last_edit = joined
                    except Exception:
                        pass

        edit_task = asyncio.create_task(periodic_edit())
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                content_chunks.append(delta)
        edit_task.cancel()

        final_text = "".join(content_chunks).strip()
        entry = _get_user_entry(context, update.effective_user.id)
        hist = entry.setdefault("history", [])
        hist.append({"role": "user", "content": prompt})
        hist.append({"role": "assistant", "content": final_text})
        save_store(context)

        await placeholder.edit_text((final_text or "(no content)")[:4096])
        _inc_usage(context, "ai")
    except Exception as e:
        logging.exception("AI chat error: %s", e)
        try:
            await placeholder.edit_text("❌ AI error. Try again later.")
        except Exception:
            pass


async def add_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data
    args = context.args

    if not args:
        await update.message.reply_text("⚠️ Please provide data: /add <your_data>")
        return

    new_data = " ".join(args)
    if "items" not in user_data:
        user_data["items"] = []

    user_data["items"].append(new_data)

    # Also persist securely per user if BOT_DATA_KEY is set
    try:
        entry = _get_user_entry(context, update.effective_user.id)
        items = entry.setdefault("items", [])
        items.append(new_data)
        save_store(context)
    except Exception:
        pass

    await update.message.reply_text(f"✅ Added: {new_data}")


async def get_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Prefer encrypted store if available
    try:
        entry = _get_user_entry(context, update.effective_user.id)
        items = entry.get("items", [])
    except Exception:
        items = []

    if not items:
        user_data = context.user_data
        items = user_data.get("items", [])

    if not items:
        await update.message.reply_text("ℹ️ You haven't added any data yet!")
        return

    items_list = "\n".join(f"• {item}" for item in items)
    await update.message.reply_text(f"📦 Your stored data:\n{items_list}")


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    client = get_openai_client(context)
    if not client:
        return
    voice = update.message.voice or update.message.audio
    if not voice:
        return
    tf = await context.bot.get_file(voice.file_id)
    # Download into memory
    file_bytes = await tf.download_as_bytearray()
    bio = io.BytesIO(file_bytes)
    bio.name = "audio.ogg"
    try:
        tr = client.audio.transcriptions.create(
            model=os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
            file=bio,
            response_format="text",
        )
        text = tr.strip() if isinstance(tr, str) else getattr(tr, "text", "").strip()
        if not text:
            text = "(empty transcription)"
        await update.message.reply_text(f"🗣️ {text[:4000]}")
    except Exception as e:
        logging.exception("Transcription error: %s", e)
        await update.message.reply_text("❌ Transcription failed.")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    if update.message.text.strip().startswith("/"):
        return
    await handle_ai_chat(update, context, update.message.text.strip())


async def newsportal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    public_url_path = "/workspace/news_portal_url.txt"
    public_url = None
    try:
        if os.path.exists(public_url_path):
            with open(public_url_path, "r") as f:
                public_url = f.read().strip()
    except Exception:
        public_url = None
    if public_url:
        await update.message.reply_text(f"📰 News Portal: {public_url}")
        return
    host = os.environ.get("NEWS_PORTAL_HOST", "http://127.0.0.1:8080")
    await update.message.reply_text(f"📰 News Portal: {host}")


async def inline_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query or ""
    if not query.strip():
        return
    # Simple AI-backed suggestion: echo + web hint
    suggestions = [
        InlineQueryResultArticle(
            id=str(int(time.time() * 1000)),
            title=f"Ask: {query[:50]}",
            input_message_content=InputTextMessageContent(f"/ask {query}")
        )
    ]
    try:
        await update.inline_query.answer(suggestions, cache_time=5, is_personal=True)
    except Exception:
        pass

async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("OK")


async def referral_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    base = os.environ.get("REFERRAL_BASE", "http://127.0.0.1:8090")
    await update.message.reply_text(f"🎯 Referral API base: {base}\nPOST {base}/register (json: {{username}})\nPOST {base}/reward (json: {{referral_link}})")


# Advanced config and helpers
OWNER_ID = int(os.environ.get("OWNER_ID", "0") or 0)

def _get_flags(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, bool]:
    flags = context.application.bot_data.setdefault("FEATURE_FLAGS", {})
    # defaults
    defaults = {
        "moderation": False,
        "tts": True,
        "ocr": True,
        "ask_cache": True,
        "quota": False,
    }
    for k, v in defaults.items():
        flags.setdefault(k, v)
    # Env overrides: FEATURE_<NAME>=on|off
    def env_on(name: str, default: bool) -> bool:
        val = os.environ.get(name)
        if not val:
            return default
        val = val.strip().lower()
        if val in ("1", "true", "on", "yes"): return True
        if val in ("0", "false", "off", "no"): return False
        return default
    flags["moderation"] = env_on("FEATURE_MODERATION", flags["moderation"]) 
    flags["quota"] = env_on("FEATURE_QUOTA", flags["quota"]) 
    flags["ask_cache"] = env_on("FEATURE_ASK_CACHE", flags["ask_cache"]) 
    flags["tts"] = env_on("FEATURE_TTS", flags["tts"]) 
    flags["ocr"] = env_on("FEATURE_OCR", flags["ocr"]) 
    return flags

# Simple per-user rate limiting
_last_msg_ts = {}

def _rate_limit_ok(user_id: int, min_interval_s: float = 0.5) -> bool:
    now = time.time()
    prev = _last_msg_ts.get(user_id, 0.0)
    if now - prev < min_interval_s:
        return False
    _last_msg_ts[user_id] = now
    return True

# Moderation
async def _moderate_if_enabled(context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
    flags = _get_flags(context)
    if not flags.get("moderation"):
        return True
    try:
        client = get_openai_client(context)
        if not client:
            return True
        res = client.moderations.create(model=os.environ.get("OPENAI_MODERATION_MODEL", "omni-moderation-latest"), input=text)
        out = getattr(res, "results", [{}])[0]
        if out.get("flagged"):
            return False
    except Exception:
        return True
    return True

# Persona management
async def persona_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    if not args:
        entry = _get_user_entry(context, update.effective_user.id)
        persona = entry.get("persona") or "default"
        await update.message.reply_text(f"Current persona: {persona}\nUsage: /persona set <name> | /persona list")
        return
    if args[0].lower() == "list":
        await update.message.reply_text("Personas: default, tutor, coder, analyst, friendly")
        return
    if args[0].lower() == "set" and len(args) >= 2:
        name = args[1].strip().lower()
        entry = _get_user_entry(context, update.effective_user.id)
        entry["persona"] = name
        save_store(context)
        await update.message.reply_text(f"✅ Persona set: {name}")
        return
    await update.message.reply_text("Usage: /persona set <name> | /persona list")

# Analytics
def _inc_usage(context: ContextTypes.DEFAULT_TYPE, key: str):
    usage = context.application.bot_data.setdefault("USAGE", {})
    usage[key] = usage.get(key, 0) + 1

async def analytics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if OWNER_ID and update.effective_user.id != OWNER_ID:
        await update.message.reply_text("Unauthorized")
        return
    usage = context.application.bot_data.get("USAGE", {})
    lines = [f"{k}: {v}" for k, v in sorted(usage.items())]
    await update.message.reply_text("Usage counts:\n" + ("\n".join(lines) or "(empty)"))

# Admin feature toggles
async def feature_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if OWNER_ID and update.effective_user.id != OWNER_ID:
        await update.message.reply_text("Unauthorized")
        return
    args = context.args or []
    flags = _get_flags(context)
    if len(args) < 2:
        await update.message.reply_text("Usage: /feature <name> on|off\n" + "\n".join([f"{k}={v}" for k, v in flags.items()]))
        return
    name, state = args[0], args[1].lower()
    if name not in flags:
        await update.message.reply_text("Unknown flag. Available: " + ", ".join(flags.keys()))
        return
    flags[name] = state == "on"
    await update.message.reply_text(f"✅ {name} set to {flags[name]}")

# Caching and quota for /ask
def _get_cache(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Dict[str, Any]]:
    return context.application.bot_data.setdefault("ASK_CACHE", {})

def _cache_get(context: ContextTypes.DEFAULT_TYPE, key: str, ttl_s: int = 600):
    cache = _get_cache(context)
    entry = cache.get(key)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > ttl_s:
        cache.pop(key, None)
        return None
    return entry.get("val")

def _cache_set(context: ContextTypes.DEFAULT_TYPE, key: str, val: str):
    cache = _get_cache(context)
    cache[key] = {"ts": time.time(), "val": val}

# Daily quota tracking (per user)
from datetime import datetime

def _ask_quota_ok(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> bool:
    flags = _get_flags(context)
    if not flags.get("quota"):
        return True
    limit = int(os.environ.get("DAILY_ASK_LIMIT", "50") or 50)
    entry = _get_user_entry(context, user_id)
    q = entry.setdefault("ask_quota", {})
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if q.get("day") != today:
        q["day"], q["count"] = today, 0
    if q["count"] >= limit:
        return False
    q["count"] += 1
    save_store(context)
    return True

# TTS
async def tts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    flags = _get_flags(context)
    if not flags.get("tts"):
        await update.message.reply_text("TTS disabled")
        return
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text("Usage: /tts <text>")
        return
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY to use TTS.")
        return
    try:
        speech = client.audio.speech.create(
            model=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=os.environ.get("OPENAI_TTS_VOICE", "alloy"),
            input=text,
            format=os.environ.get("OPENAI_TTS_FORMAT", "mp3"),
        )
        audio_bytes = speech.read() if hasattr(speech, "read") else getattr(speech, "content", b"")
        if not audio_bytes:
            audio_bytes = speech  # some SDKs return bytes
        bio = io.BytesIO(audio_bytes)
        bio.name = "speech.mp3"
        await update.message.reply_voice(voice=InputFile(bio, filename="speech.mp3"))
    except Exception as e:
        logging.exception("TTS error: %s", e)
        await update.message.reply_text("❌ TTS failed.")

# OCR
async def ocr_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    flags = _get_flags(context)
    if not flags.get("ocr"):
        await update.message.reply_text("OCR disabled")
        return
    if not update.message or not (update.message.photo or update.message.document):
        await update.message.reply_text("Reply to an image with /ocr or send an image with /ocr in caption.")
        return
    photo = None
    if update.message.photo:
        photo = update.message.photo[-1]
    elif update.message.document and str(update.message.document.mime_type or "").startswith("image/"):
        photo = update.message.document
    if not photo:
        await update.message.reply_text("No image found.")
        return
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY to use OCR.")
        return
    tf = await context.bot.get_file(photo.file_id)
    image_url = tf.file_path
    messages = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Extract all visible text from this image."},
            {"type": "input_image", "image_url": image_url},
        ]
    }]
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        await update.message.reply_text(text[:4096])
    except Exception as e:
        logging.exception("OCR error: %s", e)
        await update.message.reply_text("❌ OCR failed.")


def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN environment variable is not set")

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .rate_limiter(AIORateLimiter())
        .concurrent_updates(True)
        .build()
    )

    load_store_eager(application)
    try:
        application.job_queue.run_repeating(periodic_save_job, interval=60, first=60)
        application.job_queue.run_repeating(_push_news_job, interval=300, first=120)
    except Exception:
        pass

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("commands", help_command))
    application.add_handler(CommandHandler("add", add_data))
    application.add_handler(CommandHandler("get", get_data))
    application.add_handler(CommandHandler("setkey", setkey_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("ai", ai_command))
    application.add_handler(CommandHandler("img", img_command))
    application.add_handler(CommandHandler("web", web_command))
    application.add_handler(CommandHandler("news", news_command))
    application.add_handler(CommandHandler("newsportal", newsportal_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("subscribe_news", subscribe_news_command))
    application.add_handler(CommandHandler("unsubscribe_news", unsubscribe_news_command))
    application.add_handler(CommandHandler("code", code_command))
    application.add_handler(CommandHandler("health", health_command))
    application.add_handler(CommandHandler("persona", persona_command))
    application.add_handler(CommandHandler("analytics", analytics_command))
    application.add_handler(CommandHandler("feature", feature_command))
    application.add_handler(CommandHandler("tts", tts_command))
    application.add_handler(CommandHandler("ocr", ocr_command))
    application.add_handler(CommandHandler("referral", referral_command))

    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    application.add_handler(InlineQueryHandler(inline_query_handler))

    application.add_error_handler(error_handler)

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )

if __name__ == "__main__":
    main()