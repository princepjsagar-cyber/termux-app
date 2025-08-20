import os
from typing import Optional

from fastapi import FastAPI, Response, Query
from pydantic import BaseModel
import requests

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


@app.get("/news/realtime")
async def news_realtime(
	symbols: str = Query("AAPL,MSFT"),
	limit: int = Query(8, ge=1, le=50),
	sort: str = Query("publishedAt"),
	language: Optional[str] = Query("en"),
) -> dict:
	"""Server-side proxy for news to avoid exposing API keys in the browser.

	Defaults to NewsAPI org /v2/everything using NEWS_API_KEY.
	Env:
	- NEWS_API_KEY: required
	- NEWS_API_ENDPOINT: optional, defaults to https://newsapi.org/v2/everything
	"""
	api_key = os.getenv("NEWS_API_KEY")
	endpoint = os.getenv("NEWS_API_ENDPOINT", "https://newsapi.org/v2/everything")
	if not api_key:
		return {"articles": [], "error": "NEWS_API_KEY not set"}
	query = " OR ".join([s.strip() for s in symbols.split(",") if s.strip()]) or "AI"
	params = {
		"q": query,
		"pageSize": limit,
		"sortBy": sort,
		"language": language,
	}
	try:
		resp = requests.get(endpoint, params=params, headers={"X-Api-Key": api_key}, timeout=10)
		data = resp.json()
		# Normalize to { articles: [...] }
		articles = data.get("articles") or data.get("news") or []
		return {"articles": articles}
	except Exception as exc:
		return {"articles": [], "error": str(exc)}


@app.get("/news")
async def news_widget() -> Response:
	try:
		with open("news_widget.html", "r", encoding="utf-8") as f:
			content = f.read()
		return Response(content, media_type="text/html")
	except Exception:
		return Response("<h1>News widget not found</h1>", media_type="text/html", status_code=404)