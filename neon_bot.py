import os
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    AIORateLimiter,
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
        "Use /get to retrieve your data (session-only)"
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
    application.add_handler(CommandHandler("add", add_data))
    application.add_handler(CommandHandler("get", get_data))
    application.add_error_handler(error_handler)

    # Start bot with low-latency long polling and no disk persistence
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()

