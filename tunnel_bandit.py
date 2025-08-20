import logging
import os
import random
import time
from typing import Dict, List, Optional

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("tunnel_bandit")


class EpsilonGreedyBandit:
	def __init__(self, arms: List[str], epsilon: float = 0.1) -> None:
		self.epsilon = epsilon
		self.arms = arms
		self.successes: Dict[str, int] = {a: 1 for a in arms}
		self.trials: Dict[str, int] = {a: 2 for a in arms}

	def select(self) -> str:
		if not self.arms:
			raise RuntimeError("No proxy arms configured")
		if random.random() < self.epsilon:
			return random.choice(self.arms)
		return max(self.arms, key=lambda a: self.successes[a] / self.trials[a])

	def update(self, arm: str, success: bool) -> None:
		self.trials[arm] += 1
		if success:
			self.successes[arm] += 1


def load_proxies() -> List[str]:
	proxies_str = os.getenv("PROXY_LIST", "").strip()
	if not proxies_str:
		return []
	return [p.strip() for p in proxies_str.split(",") if p.strip()]


def probe_proxy(proxy: str, timeout: float = 2.0) -> bool:
	# Placeholder: implement real probing logic (HTTP request or TCP connect)
	time.sleep(0.05)
	return random.random() > 0.2


def run_selector_loop() -> None:
	proxies = load_proxies()
	if not proxies:
		logger.info("No proxies configured; selector will idle")
		while True:
			time.sleep(10)
			continue
	bandit = EpsilonGreedyBandit(proxies, epsilon=float(os.getenv("BANDIT_EPSILON", "0.1")))
	logger.info("Tunnel bandit started with %d proxies", len(proxies))
	while True:
		chosen = bandit.select()
		success = probe_proxy(chosen)
		bandit.update(chosen, success)
		logger.info("proxy=%s success=%s score=%.3f", chosen, success, bandit.successes[chosen] / bandit.trials[chosen])
		time.sleep(float(os.getenv("BANDIT_INTERVAL_SECONDS", "15")))


if __name__ == "__main__":  # pragma: no cover
	run_selector_loop()