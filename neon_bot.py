import os
import io
import base64
import asyncio
import logging
import json
import hashlib
from typing import List, Dict, Any
from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    AIORateLimiter,
    MessageHandler,
    filters,
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
    await update.message.reply_text(
        "Here are the commands you can use:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/add <your_data> - Add data to your secure store\n"
        "/get - Get your stored data\n"
        "/setkey <your_key> - Set your secure key (optional)\n"
        "/status - Show bot status\n"
        "/ai - Start AI chat\n"
        "/img - Generate image (requires DALL-E)\n"
        "/web - Search the web (requires Google Custom Search)\n"
        "/code - Generate code (requires GitHub)\n"
        "/voice - Transcribe voice message\n"
        "You can also send photos or text messages."
    )


async def setkey_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Please provide a key: /setkey <your_key>")
        return
    key = " ".join(args)
    context.application.bot_data["BOT_DATA_KEY"] = key
    await update.message.reply_text(f"✅ Key set. Your data will be encrypted with: {key[:4]}...")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bot Status:\n"
        f"• Token: {'Set' if BOT_TOKEN else 'Not Set'}\n"
        f"• Data Key: {'Set' if context.application.bot_data.get('BOT_DATA_KEY') else 'Not Set'}\n"
        f"• Store Path: {_get_store_path()}\n"
        f"• OpenAI API Key: {'Set' if os.environ.get('OPENAI_API_KEY') else 'Not Set'}\n"
        f"• DALL-E API Key: {'Set' if os.environ.get('DALL_E_API_KEY') else 'Not Set'}\n"
        f"• Google Custom Search API Key: {'Set' if os.environ.get('GOOGLE_CSE_API_KEY') else 'Not Set'}\n"
        f"• GitHub API Key: {'Set' if os.environ.get('GITHUB_API_KEY') else 'Not Set'}"
    )


async def ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /ai <prompt>")
        return
    await handle_ai_chat(update, context, prompt)


async def img_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY to enable image generation.")
        return
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /img <prompt>")
        return
    try:
        img = client.images.generate(
            model=os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            prompt=prompt,
            size=os.environ.get("OPENAI_IMAGE_SIZE", "1024x1024"),
        )
        b64 = img.data[0].b64_json
        data = base64.b64decode(b64)
        bio = io.BytesIO(data)
        bio.name = "image.png"
        await update.message.reply_photo(photo=InputFile(bio, filename="image.png"), caption=f"🎨 {prompt[:200]}")
    except Exception as e:
        logging.exception("Image generation error: %s", e)
        await update.message.reply_text("❌ Image generation failed.")


async def web_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import httpx
    tc = get_tavily_client(context)
    serpapi_key = get_serpapi_key(context)
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /web <query>")
        return
    try:
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
        await update.message.reply_text("Set TAVILY_API_KEY or SERPAPI_KEY to enable web search.")
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


def ensure_history(context: ContextTypes.DEFAULT_TYPE) -> List[Dict[str, Any]]:
    user_data = context.user_data
    if "history" not in user_data:
        user_data["history"] = []
    # Trim to last 10 exchanges
    if len(user_data["history"]) > 20:
        user_data["history"] = user_data["history"][-20:]
    return user_data["history"]


async def handle_ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY to enable AI chat.")
        return

    # Use encrypted persistent history per user if available
    try:
        user_id = update.effective_user.id
        entry = _get_user_entry(context, user_id)
        history = entry.setdefault("history", [])
        if len(history) > 20:
            entry["history"] = history[-20:]
            history = entry["history"]
    except Exception:
        history = ensure_history(context)

    append_user_message(history, prompt)

    # Send placeholder message for streaming edits
    placeholder = await update.message.reply_text("🤖 Thinking…")

    content_chunks: List[str] = []

    try:
        stream = client.chat.completions.create(
            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=history,
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
        # Save assistant reply to history
        history.append({"role": "assistant", "content": final_text})

        if final_text:
            # Ensure final text is applied
            await placeholder.edit_text(final_text[:4096])
        else:
            await placeholder.edit_text("(no content)")
        # Persist updated history
        save_store(context)

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
    except Exception:
        pass

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("add", add_data))
    application.add_handler(CommandHandler("get", get_data))
    application.add_handler(CommandHandler("setkey", setkey_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("ai", ai_command))
    application.add_handler(CommandHandler("img", img_command))
    application.add_handler(CommandHandler("web", web_command))
    application.add_handler(CommandHandler("code", code_command))

    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    application.add_error_handler(error_handler)

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )

if __name__ == "__main__":
    main()