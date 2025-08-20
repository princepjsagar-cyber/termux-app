import os
import time
import logging
from typing import List

import requests

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("keepalive_worker")


def parse_urls(env_value: str) -> List[str]:
	urls: List[str] = []
	for raw in env_value.split(","):
		u = raw.strip()
		if not u:
			continue
		if not (u.startswith("http://") or u.startswith("https://")):
			continue
		urls.append(u)
	return urls


def run() -> None:
	urls_env = os.getenv("KEEPALIVE_URLS", "").strip()
	if not urls_env:
		logger.info("No KEEPALIVE_URLS provided; idling")
		while True:
			time.sleep(600)
			continue
	urls = parse_urls(urls_env)
	interval_seconds = int(os.getenv("KEEPALIVE_INTERVAL_SECONDS", "240"))
	logger.info("Keepalive worker started: %d URLs, interval=%ss", len(urls), interval_seconds)
	while True:
		for url in urls:
			try:
				resp = requests.get(url, timeout=8)
				logger.info("ping url=%s status=%s", url, resp.status_code)
			except Exception as exc:  # pragma: no cover
				logger.warning("ping failed url=%s err=%s", url, exc)
		time.sleep(interval_seconds)


if __name__ == "__main__":  # pragma: no cover
	run()