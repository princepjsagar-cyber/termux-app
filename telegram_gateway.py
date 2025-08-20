import asyncio
import logging
import os
import json
from typing import Optional

from neon_bot import get_bot

try:
	import redis  # type: ignore
except Exception:
	redis = None  # type: ignore

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

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


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	text: Optional[str] = update.effective_message.text if update.effective_message else None
	user_id: Optional[str] = str(update.effective_user.id) if update.effective_user else None
	if not text:
		return
	# Enqueue as a task if Redis is present; else handle via NeonBot directly
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


async def main() -> None:
	token = os.getenv("TELEGRAM_BOT_TOKEN")
	if not token:
		raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
	app = ApplicationBuilder().token(token).build()
	app.add_handler(MessageHandler(filters.ALL & (~filters.COMMAND), on_message))
	logger.info("Telegram gateway started (polling mode)")
	await app.initialize()
	await app.start()
	try:
		await app.updater.start_polling()
		await asyncio.Event().wait()
	finally:
		await app.updater.stop()
		await app.stop()
		await app.shutdown()


if __name__ == "__main__":  # pragma: no cover
	asyncio.run(main())