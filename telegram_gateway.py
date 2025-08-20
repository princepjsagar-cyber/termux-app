import logging
import os
import json
from typing import Optional, Set

from neon_bot import get_bot

try:
	import redis  # type: ignore
except Exception:
	redis = None  # type: ignore

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram_gateway")


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
	await update.effective_message.reply_text("Hello! I'm online 24/7. Send a message to interact.")


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


def main() -> None:
	token = os.getenv("TELEGRAM_BOT_TOKEN")
	if not token:
		raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
	app = ApplicationBuilder().token(token).build()
	app.add_handler(CommandHandler("start", cmd_start))
	app.add_handler(CommandHandler("ping", cmd_ping))
	app.add_handler(CommandHandler("stats", cmd_stats))
	app.add_handler(CommandHandler("broadcast", cmd_broadcast))
	app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
	logger.info("Telegram gateway started (polling mode)")
	# Blocking; recommended pattern for PTB v21
	app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":  # pragma: no cover
	main()