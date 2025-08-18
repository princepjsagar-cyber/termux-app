import asyncio
import logging
import os

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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


def main() -> None:
    application = build_application()

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("invite", invite_command))

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

