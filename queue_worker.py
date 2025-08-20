import asyncio
import json
import logging
import os
import time
from typing import Optional

from neon_bot import get_bot

try:
	import redis  # type: ignore
except Exception:  # pragma: no cover
	redis = None  # type: ignore

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("queue_worker")


def get_redis_client():
	redis_url = os.getenv("REDIS_URL")
	if not redis_url or redis is None:
		return None
	try:
		return redis.from_url(redis_url, decode_responses=True)
	except Exception as exc:  # pragma: no cover
		logger.warning("Redis connection failed: %s", exc)
		return None


async def process_task(task: dict) -> None:
	bot = get_bot()
	if task.get("type") == "smm_order":
		message = f"SMM order: service={task.get('service')} target={task.get('target')} qty={task.get('quantity')}"
		await bot.send_message(message)
	else:
		await bot.send_message(f"Task: {task}")


async def worker_loop() -> None:
	client = get_redis_client()
	if client is None:
		logger.info("No Redis configured. Worker will idle.")
	while True:
		try:
			if client is None:
				await asyncio.sleep(5)
				continue
			item = client.blpop(["queue:tasks"], timeout=5)
			if not item:
				continue
			_, data = item
			task = json.loads(data)
			await process_task(task)
		except Exception as exc:  # pragma: no cover
			logger.exception("Worker error: %s", exc)
			await asyncio.sleep(2)


if __name__ == "__main__":  # pragma: no cover
	asyncio.run(worker_loop())