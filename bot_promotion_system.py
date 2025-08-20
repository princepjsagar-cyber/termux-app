import asyncio
import logging
import os
from neon_bot import get_bot

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("promotion_system")


async def promotion_loop() -> None:
	interval = int(os.getenv("PROMOTION_INTERVAL_SECONDS", "300"))
	message = os.getenv("PROMO_MESSAGE", "Check out our new features! 💡")
	logger.info("Promotion worker started. interval=%ss", interval)
	bot = get_bot()
	while True:
		try:
			await bot.send_message(message)
		except Exception as exc:  # pragma: no cover
			logger.warning("Promo send failed: %s", exc)
		await asyncio.sleep(interval)


if __name__ == "__main__":  # pragma: no cover
	asyncio.run(promotion_loop())