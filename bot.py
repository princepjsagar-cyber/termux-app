import logging
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

# Bot Configuration
BOT_TOKEN = "8002376288:AAE8R34Rv5QCC9A5d7IKqm8EzGFAdk46uYI"
CHANNEL_USERNAME = "@sallytheson0"
ADMIN_USER_ID = 1389741290

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("✨ Join Channel", url=f"https://t.me/{CHANNEL_USERNAME[1:]}")],
        [InlineKeyboardButton("📢 Share Invite Link", callback_data="share_invite")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"🌟 Welcome to our growth bot!\n\n"
        f"Join our channel: {CHANNEL_USERNAME}\n"
        "Click below to get your personal invite link:",
        reply_markup=reply_markup
    )

async def share_invite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user = query.from_user
    invite_link = (
        f"https://t.me/share/url?url=https://t.me/{CHANNEL_USERNAME[1:]}"
        f"&text=Join%20our%20awesome%20channel%20with%20@{user.username}!"
    )

    await query.edit_message_text(
        f"🔥 Your personal invite link:\n\n{invite_link}\n\n"
        "Share this link to invite friends to our channel!\n\n"
        "📊 You'll earn special rewards when 5 friends join!",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="back_to_main")]])
    )

async def back_to_main(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    keyboard = [
        [InlineKeyboardButton("✨ Join Channel", url=f"https://t.me/{CHANNEL_USERNAME[1:]}")],
        [InlineKeyboardButton("📢 Share Invite Link", callback_data="share_invite")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        f"🌟 Welcome to our growth bot!\n\n"
        f"Join our channel: {CHANNEL_USERNAME}\n"
        "Click below to get your personal invite link:",
        reply_markup=reply_markup
    )

async def track_join(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for member in update.message.new_chat_members:
        if member.id == context.bot.id:
            continue

        logging.info(f"New member joined: {member.full_name} (@{member.username})")
        try:
            total_members = await context.bot.get_chat_member_count(CHANNEL_USERNAME)
            await context.bot.send_message(
                chat_id=ADMIN_USER_ID,
                text=(
                    "🚀 New member joined!\n\n"
                    f"Name: {member.full_name}\n"
                    f"Username: @{member.username}\n"
                    f"Total members: {total_members}"
                )
            )
        except Exception as e:
            logging.error(f"Error sending notification: {e}")


def main():
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(share_invite, pattern="^share_invite$"))
    application.add_handler(CallbackQueryHandler(back_to_main, pattern="^back_to_main$"))

    application.add_handler(
        MessageHandler(
            filters.Chat(CHANNEL_USERNAME) & filters.StatusUpdate.NEW_CHAT_MEMBERS,
            track_join,
        )
    )

    application.run_polling()


if __name__ == '__main__':
    main()