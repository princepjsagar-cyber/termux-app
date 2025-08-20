import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

# Optional: lazy redis import to avoid hard dependency when REDIS_URL is not set
try:
	import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	redis = None  # type: ignore


logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("neon_bot")


class NeonBot:
	"""Minimal bot core with safe defaults.

	- Avoids touching local data; uses environment variables only.
	- Uses Redis if REDIS_URL is provided; otherwise operates in memory.
	"""

	def __init__(self) -> None:
		self.start_time = time.time()
		self.messages_sent = 0
		self.redis_client = None
		redis_url = os.getenv("REDIS_URL")
		if redis_url and redis is not None:
			try:
				self.redis_client = redis.from_url(redis_url, decode_responses=True)
				logger.info("Connected to Redis")
			except Exception as exc:  # pragma: no cover
				logger.warning("Failed to connect to Redis: %s", exc)

	@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
	async def send_message(self, message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
		"""Simulate sending a message; customize to your real transport/API."""
		await asyncio.sleep(0.01)
		self.messages_sent += 1
		payload = {"message": message, "user_id": user_id, "ts": time.time()}
		if self.redis_client is not None:
			try:
				self.redis_client.lpush("neon:sent", json.dumps(payload))
			except Exception as exc:  # pragma: no cover
				logger.debug("Redis push failed: %s", exc)
		return {"ok": True, "echo": payload}

	def get_status(self) -> Dict[str, Any]:
		uptime = time.time() - self.start_time
		queue_depth = None
		if self.redis_client is not None:
			try:
				queue_depth = self.redis_client.llen("queue:tasks")
			except Exception:
				queue_depth = None
		return {
			"uptime_seconds": round(uptime, 2),
			"messages_sent": self.messages_sent,
			"queue_depth": queue_depth,
			"environment": {
				"has_redis": bool(self.redis_client),
				"region": os.getenv("RENDER_REGION"),
				"env": os.getenv("ENV", "prod"),
			},
		}


_bot_singleton: Optional[NeonBot] = None


def get_bot() -> NeonBot:
	global _bot_singleton
	if _bot_singleton is None:
		_bot_singleton = NeonBot()
	return _bot_singleton


async def main_loop() -> None:
	"""Optional CLI loop for local testing."""
	bot = get_bot()
	logger.info("Neon bot started")
	while True:
		await bot.send_message(f"heartbeat-{random.randint(1000, 9999)}")
		await asyncio.sleep(float(os.getenv("BOT_HEARTBEAT_SECONDS", "30")))


if __name__ == "__main__":  # pragma: no cover
	asyncio.run(main_loop())