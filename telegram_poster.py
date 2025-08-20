import asyncio
import logging
import os
from typing import List

from ai_services import AIServices

try:
	import redis  # type: ignore
except Exception:
	redis = None  # type: ignore

from telegram import Bot

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram_poster")


def get_redis_client():
	redis_url = os.getenv("REDIS_URL")
	if not redis_url or redis is None:
		return None
	try:
		return redis.from_url(redis_url, decode_responses=True)
	except Exception:
		return None


async def post_loop() -> None:
	bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
	channel = os.getenv("POST_CHANNEL")  # e.g., @yourchannel or chat id
	interval = int(os.getenv("POST_INTERVAL_SECONDS", "86400"))
	if not bot_token:
		raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
	ai = AIServices()
	bot = Bot(token=bot_token)
	client = get_redis_client()
	logger.info("telegram poster started interval=%s channel=%s", interval, channel)
	while True:
		try:
			content = ai.generate_text(os.getenv("POST_PROMPT", "Share a concise daily tip"))
			if channel:
				await bot.send_message(chat_id=channel, text=content[:4000])
			elif client is not None:
				# Send to opted-in users
				for uid in list(client.smembers("optin:daily")):
					try:
						await bot.send_message(chat_id=int(uid), text=content[:4000])
					except Exception:
						continue
			else:
				logger.info("No channel or opt-in set; skipping post")
		except Exception as exc:
			logger.warning("post failed: %s", exc)
		await asyncio.sleep(interval)


if __name__ == "__main__":  # pragma: no cover
	asyncio.run(post_loop())