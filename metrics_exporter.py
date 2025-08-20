import os
import time
import logging
from typing import Dict

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

from neon_bot import get_bot

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("metrics_exporter")

app = FastAPI(title="Metrics Exporter", version="1.0.0")

BOT_UPTIME = Gauge("bot_uptime_seconds", "Neon bot uptime in seconds")
BOT_MESSAGES = Gauge("bot_messages_sent", "Total messages sent by the bot")
QUEUE_DEPTH = Gauge("queue_depth", "Queue depth if available")


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
	return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
	bot = get_bot()
	status = bot.get_status()
	BOT_UPTIME.set(status.get("uptime_seconds", 0.0))
	BOT_MESSAGES.set(status.get("messages_sent", 0))
	qd = status.get("queue_depth")
	if qd is not None:
		QUEUE_DEPTH.set(qd)
	return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root() -> Dict[str, str]:
	return {"name": "metrics-exporter", "status": "running"}