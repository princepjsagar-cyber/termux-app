import os
import io
import base64
import asyncio
import logging
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


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Bot credentials (read from environment; do not store on disk)
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
BOT_USERNAME = "NeonTakeshiBot"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_html(
        f"👋 Hi {user.mention_html()}! I'm {BOT_USERNAME}.\n"
        "Use /add <data> to store information (in-memory only)\n"
        "Use /get to retrieve your data (session-only)\n\n"
        "AI features:\n"
        "• /ai <prompt> — advanced AI chat (also replies to plain text)\n"
        "• Send an image with caption 'describe' — vision Q&A\n"
        "• /img <prompt> — image generation\n"
        "• Send a voice note — transcription\n"
        "• /code <lang> <code> — run code in sandbox\n"
        "• /web <query> — web search + summary"
    )


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
    await update.message.reply_text(f"✅ Added: {new_data}")


async def get_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data
    if "items" not in user_data or not user_data["items"]:
        await update.message.reply_text("ℹ️ You haven't added any data yet!")
        return

    items_list = "\n".join(f"• {item}" for item in user_data["items"])
    await update.message.reply_text(f"📦 Your stored data:\n{items_list}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error("Exception while handling update:", exc_info=context.error)
    if update and hasattr(update, "message") and update.message:
        try:
            await update.message.reply_text("⚠️ An error occurred. Please try again later.")
        except Exception:
            pass


# --------- AI provider utils ---------
def _get_runtime_key(context: ContextTypes.DEFAULT_TYPE, name: str) -> str:
    try:
        app_data = context.application.bot_data
        val = app_data.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


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


def append_user_message(history: List[Dict[str, Any]], text: str) -> None:
    history.append({"role": "user", "content": text})


def ensure_history(context: ContextTypes.DEFAULT_TYPE) -> List[Dict[str, Any]]:
    user_data = context.user_data
    if "history" not in user_data:
        user_data["history"] = []
    # Trim to last 10 exchanges
    if len(user_data["history"]) > 20:
        user_data["history"] = user_data["history"][-20:]
    return user_data["history"]


# --------- AI chat (with streaming edits) ---------
async def handle_ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    client = get_openai_client(context)
    if not client:
        await update.message.reply_text("Set OPENAI_API_KEY to enable AI chat.")
        return

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

    except Exception as e:
        logging.exception("AI chat error: %s", e)
        try:
            await placeholder.edit_text("❌ AI error. Try again later.")
        except Exception:
            pass


async def ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /ai <prompt>")
        return
    await handle_ai_chat(update, context, prompt)


# --------- Vision on photo ---------
async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    client = get_openai_client(context)
    if not client:
        return
    if not update.message or not update.message.photo:
        return
    # Prefer largest size
    file_id = update.message.photo[-1].file_id
    tf = await context.bot.get_file(file_id)
    # We can send the file URL directly to OpenAI vision
    image_url = tf.file_path

    question = update.message.caption or "Describe this image."

    messages = [
        {"role": "user", "content": [
            {"type": "input_text", "text": question},
            {"type": "input_image", "image_url": image_url},
        ]}
    ]

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
        await update.message.reply_text(answer[:4096])
    except Exception as e:
        logging.exception("Vision error: %s", e)
        await update.message.reply_text("❌ Vision failed.")


# --------- Image generation ---------
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


# --------- Voice transcription ---------
async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    client = get_openai_client()
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


# --------- Web search ---------
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
            params = {
                "engine": "google",
                "q": query,
                "api_key": serpapi_key,
                "num": "5",
            }
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


# --------- In-memory config management ---------
MASK = "••••••••"


def _mask_key(value: str) -> str:
    if not value:
        return "(missing)"
    if len(value) <= 8:
        return MASK
    return value[:4] + MASK + value[-4:]


async def setkey_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /setkey <OPENAI_API_KEY|TAVILY_API_KEY|SERPAPI_KEY> <value>"
        )
        return
    name = context.args[0].strip().upper()
    value = " ".join(context.args[1:]).strip()
    allowed = {"OPENAI_API_KEY", "TAVILY_API_KEY", "SERPAPI_KEY"}
    if name not in allowed:
        await update.message.reply_text(
            "Key must be one of: OPENAI_API_KEY, TAVILY_API_KEY, SERPAPI_KEY"
        )
        return
    context.application.bot_data[name] = value
    await update.message.reply_text(f"✅ Set {name} = {_mask_key(value)} (in-memory)")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    openai = _get_runtime_key(context, "OPENAI_API_KEY")
    tavily = _get_runtime_key(context, "TAVILY_API_KEY")
    serp = _get_runtime_key(context, "SERPAPI_KEY")
    lines = [
        f"OPENAI_API_KEY: {_mask_key(openai)}",
        f"TAVILY_API_KEY: {_mask_key(tavily)}",
        f"SERPAPI_KEY: {_mask_key(serp)}",
        "Features: AI chat, vision, images, voice, web search, code exec",
    ]
    await update.message.reply_text("\n".join(lines))


# --------- Code execution via Piston ---------
async def code_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import httpx
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usage: /code <lang> <code>")
        return
    lang = context.args[0]
    code = " ".join(context.args[1:])
    # Strip backticks if present
    if code.startswith("```"):
        code = code.strip("`\n ")
        # Remove possible language hint
        first_newline = code.find("\n")
        if first_newline != -1:
            head = code[:first_newline]
            if len(head) < 12:
                code = code[first_newline + 1 :]
    payload = {
        "language": lang,
        "files": [{"name": f"main.{lang}", "content": code}],
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post("https://emkc.org/api/v2/piston/execute", json=payload)
            resp.raise_for_status()
            data = resp.json()
        run = data.get("run", {})
        out = run.get("stdout", "")
        err = run.get("stderr", "")
        if not out and not err:
            out = "(no output)"
        text = (out + ("\n" + err if err else "")).strip()
        await update.message.reply_text(f"🧪 Output:\n{text[:4000]}")
    except Exception as e:
        logging.exception("Code exec error: %s", e)
        await update.message.reply_text("❌ Code execution failed.")


# --------- Fallback text handler to AI ---------
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    # Ignore commands (handled separately)
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

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
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

    # Start bot with low-latency long polling and no disk persistence
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()

