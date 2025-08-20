import os
import time
import json
import logging
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Optional Redis
try:
	import redis  # type: ignore
except Exception:  # pragma: no cover
	redis = None  # type: ignore

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("smm_panel")

app = FastAPI(title="Enhanced SMM Panel", version="1.0.0")


class OrderRequest(BaseModel):
	service: str = Field(..., description="Service name")
	target: str = Field(..., description="Target username/link")
	quantity: int = Field(..., ge=1, description="Quantity to deliver")
	note: Optional[str] = Field(default=None)


class OrderResponse(BaseModel):
	order_id: str
	enqueued: bool
	queue_depth: Optional[int] = None


@app.get("/healthz")
async def healthz() -> dict:
	return {"status": "ok", "panel": "smm"}


def _get_redis_client():
	redis_url = os.getenv("REDIS_URL")
	if not redis_url or redis is None:
		return None
	try:
		return redis.from_url(redis_url, decode_responses=True)
	except Exception as exc:  # pragma: no cover
		logger.warning("Redis connection failed: %s", exc)
		return None


@app.post("/orders", response_model=OrderResponse)
async def create_order(req: OrderRequest) -> OrderResponse:
	order_id = str(uuid4())
	task = {
		"id": order_id,
		"type": "smm_order",
		"service": req.service,
		"target": req.target,
		"quantity": req.quantity,
		"note": req.note,
		"ts": time.time(),
	}
	client = _get_redis_client()
	queue_depth: Optional[int] = None
	if client is not None:
		try:
			client.rpush("queue:tasks", json.dumps(task))
			queue_depth = client.llen("queue:tasks")
		except Exception as exc:  # pragma: no cover
			logger.warning("Failed to enqueue task: %s", exc)
	else:
		logger.info("Redis not configured; order accepted but not queued")
	return OrderResponse(order_id=order_id, enqueued=client is not None, queue_depth=queue_depth)


@app.get("/")
async def root() -> dict:
	return {"name": "enhanced-smm-panel", "status": "running"}