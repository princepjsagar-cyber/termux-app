import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from neon_bot import get_bot

app = FastAPI(title="Akka Bot Server", version="1.0.0")


class SendRequest(BaseModel):
	message: str
	user_id: Optional[str] = None


@app.get("/healthz")
async def healthz() -> dict:
	return {"status": "ok", "env": os.getenv("ENV", "prod")}


@app.get("/")
async def root() -> dict:
	return {"name": "akka-bot-server", "status": "running"}


@app.post("/bot/send")
async def send(req: SendRequest) -> dict:
	bot = get_bot()
	result = await bot.send_message(req.message, req.user_id)
	return result


@app.get("/bot/status")
async def status() -> dict:
	bot = get_bot()
	return bot.get_status()