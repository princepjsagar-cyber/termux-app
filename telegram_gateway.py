import logging
import os
import json
from typing import Optional, Set

from neon_bot import get_bot
from ai_services import AIServices

try:
	import redis  # type: ignore
except Exception:
	redis = None  # type: ignore

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from telegram.constants import ChatMemberStatus, ChatType

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram_gateway")

_ai = AIServices()


def get_redis_client():
	redis_url = os.getenv("REDIS_URL")
	if not redis_url or redis is None:
		return None
	try:
		return redis.from_url(redis_url, decode_responses=True)
	except Exception:
		return None


def get_admin_ids() -> Set[str]:
	raw = os.getenv("ADMIN_IDS", "").strip()
	if not raw:
		return set()
	return {p.strip() for p in raw.split(",") if p.strip()}


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	text: Optional[str] = update.effective_message.text if update.effective_message else None
	user_id: Optional[str] = str(update.effective_user.id) if update.effective_user else None
	if not text:
		return
	client = get_redis_client()
	if client is not None:
		client.rpush("queue:tasks", json.dumps({
			"type": "tg_message",
			"user_id": user_id,
			"text": text,
		}))
	else:
		bot = get_bot()
		await bot.send_message(f"TG({user_id}): {text}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	await update.effective_message.reply_text("Hello! I'm online 24/7. Try /help for commands.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	await update.effective_message.reply_text(
		"Commands:\n"
		"/ping - check if online\n"
		"/stats - basic stats\n"
		"/groupstats - member count for this chat\n"
		"/growth - show growth placeholder (admin)\n"
		"/invite - generate invite link (admin)\n"
		"/ai <prompt> - ask AI\n"
		"/summarize <text> - summarize\n"
		"/translate <lang> <text> - translate\n"
		"/broadcast <msg> - admin only"
	)


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	await update.effective_message.reply_text("pong")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	bot = get_bot()
	status = bot.get_status()
	msg = (
		f"Uptime: {status.get('uptime_seconds')}s\n"
		f"Sent: {status.get('messages_sent')}\n"
		f"Queue: {status.get('queue_depth')}\n"
	)
	await update.effective_message.reply_text(msg)


async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	user_id = str(update.effective_user.id) if update.effective_user else ""
	if user_id not in get_admin_ids():
		await update.effective_message.reply_text("Unauthorized")
		return
	text = " ".join(context.args).strip()
	if not text:
		await update.effective_message.reply_text("Usage: /broadcast <message>")
		return
	client = get_redis_client()
	if client is not None:
		client.rpush("queue:tasks", json.dumps({
			"type": "broadcast",
			"from": user_id,
			"text": text,
		}))
		await update.effective_message.reply_text("Broadcast enqueued")
	else:
		bot = get_bot()
		await bot.send_message(f"BROADCAST: {text}")
		await update.effective_message.reply_text("Broadcast sent")


async def cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	prompt = " ".join(context.args).strip()
	if not prompt:
		await update.effective_message.reply_text("Usage: /ai <prompt>")
		return
	reply = _ai.generate_text(prompt)
	await update.effective_message.reply_text(reply[:3500])


async def cmd_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	text = " ".join(context.args).strip()
	if not text:
		await update.effective_message.reply_text("Usage: /summarize <text>")
		return
	reply = _ai.summarize(text)
	await update.effective_message.reply_text(reply[:3500])


async def cmd_translate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	if not context.args:
		await update.effective_message.reply_text("Usage: /translate <lang> <text>")
		return
	target = context.args[0]
	text = " ".join(context.args[1:]).strip()
	if not text:
		await update.effective_message.reply_text("Usage: /translate <lang> <text>")
		return
	reply = _ai.translate(text, target)
	await update.effective_message.reply_text(reply[:3500])


async def cmd_groupstats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	if chat is None or chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL}:
		await update.effective_message.reply_text("This command only works in groups or channels.")
		return
	try:
		count = await context.bot.get_chat_member_count(chat.id)
		await update.effective_message.reply_text(f"👥 Current member count: {count}")
	except Exception as exc:
		logger.warning("get_chat_member_count failed: %s", exc)
		await update.effective_message.reply_text("❌ Could not get member count. Make sure I'm an admin if this is a channel.")


async def _ensure_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
	try:
		member = await context.bot.get_chat_member(chat_id, user_id)
		return member.status in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER}
	except Exception:
		return False


async def cmd_growth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	user = update.effective_user
	if chat is None or user is None:
		return
	if not await _ensure_admin(context, chat.id, user.id):
		await update.effective_message.reply_text("🚫 Admins only.")
		return
	try:
		count = await context.bot.get_chat_member_count(chat.id)
		await update.effective_message.reply_text(
			"📈 Growth tracking placeholder\n"
			f"Current members: {count}\n\n"
			"To properly track growth:\n1) Add me as admin\n2) I will record daily counts (when enabled)\n3) View graphs with /groupstats"
		)
	except Exception as exc:
		logger.warning("growth failed: %s", exc)
		await update.effective_message.reply_text("❌ An error occurred. Make sure I'm an admin.")


async def cmd_invite(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	user = update.effective_user
	if chat is None or user is None:
		return
	if not await _ensure_admin(context, chat.id, user.id):
		await update.effective_message.reply_text("🚫 Admins only.")
		return
	try:
		link = await context.bot.create_chat_invite_link(chat_id=chat.id)
		await update.effective_message.reply_text(
			f"🔗 Invite link:\n\n{link.invite_link}",
			disable_web_page_preview=True,
		)
	except Exception as exc:
		logger.warning("invite failed: %s", exc)
		await update.effective_message.reply_text("❌ Could not generate invite link. Make sure I'm an admin.")


def main() -> None:
	token = os.getenv("TELEGRAM_BOT_TOKEN")
	if not token:
		raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
	app = ApplicationBuilder().token(token).build()
	app.add_handler(CommandHandler("start", cmd_start))
	app.add_handler(CommandHandler("help", cmd_help))
	app.add_handler(CommandHandler("ping", cmd_ping))
	app.add_handler(CommandHandler("stats", cmd_stats))
	app.add_handler(CommandHandler("groupstats", cmd_groupstats))
	app.add_handler(CommandHandler("growth", cmd_growth))
	app.add_handler(CommandHandler("invite", cmd_invite))
	app.add_handler(CommandHandler("broadcast", cmd_broadcast))
	app.add_handler(CommandHandler("ai", cmd_ai))
	app.add_handler(CommandHandler("summarize", cmd_summarize))
	app.add_handler(CommandHandler("translate", cmd_translate))
	app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
	logger.info("Telegram gateway started (polling mode)")
	app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":  # pragma: no cover
	main()