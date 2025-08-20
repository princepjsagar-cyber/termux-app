import os
import logging
import asyncio
from typing import Dict, List

import discord
from discord.ext import commands
import requests
from bs4 import BeautifulSoup  # noqa: F401  # placeholder if needed later

from ai_services import AIServices

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("discord_gateway")

PREFIX = os.getenv("DISCORD_PREFIX", "!")
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

ai = AIServices()

# In-memory conversation history; does not touch disk
user_history: Dict[int, List[Dict[str, str]]] = {}


def get_google_results(query: str) -> str:
	api_key = os.getenv("GOOGLE_API_KEY")
	cx = os.getenv("SEARCH_ENGINE_ID")
	if not api_key or not cx:
		return "[web search unavailable]"
	url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={requests.utils.quote(query)}"
	try:
		resp = requests.get(url, timeout=10)
		data = resp.json()
		items = data.get("items", [])[:3]
		if not items:
			return "No web results found."
		return "\n\n".join([f"🔍 [{it['title']}]({it['link']})\n{it.get('snippet','')}" for it in items])
	except Exception as exc:
		logger.warning("google search failed: %s", exc)
		return "[web search error]"


async def get_ai_response(user_id: int, message: str, standalone: bool = False) -> str:
	if user_id not in user_history:
		user_history[user_id] = []
	if standalone:
		return ai.generate_text(message)
	user_history[user_id].append({"role": "user", "content": message})
	reply = ai.generate_text(message)
	user_history[user_id].append({"role": "assistant", "content": reply})
	if len(user_history[user_id]) > 10:
		user_history[user_id] = user_history[user_id][-10:]
	return reply


@bot.event
async def on_ready():
	logger.info("Discord bot logged in as %s", bot.user.name)


@bot.event
async def on_message(message: discord.Message):
	if message.author == bot.user:
		return
	await bot.process_commands(message)


@bot.command()
async def ask(ctx: commands.Context, *, query: str):
	"""Ask AI with optional web search augmentation"""
	resp = await get_ai_response(ctx.author.id, query, standalone=True)
	if ("I don't know" in resp) or ("search" in query.lower()):
		web = get_google_results(query)
		resp = f"{resp}\n\n**Web Results:**\n{web}"
	await ctx.reply(resp[:2000])


@bot.command()
async def image(ctx: commands.Context, *, prompt: str):
	"""Generate an image via AI services"""
	url = ai.generate_image(prompt)
	await ctx.reply(url[:2000])


@bot.command()
async def history(ctx: commands.Context):
	"""Show conversation history (in-memory)"""
	hist = user_history.get(ctx.author.id, [])
	if not hist:
		await ctx.send("No history yet")
		return
	text = []
	for item in hist[-10:]:
		role = item.get("role")
		content = item.get("content", "")
		prefix = "You:" if role == "user" else "Bot:"
		text.append(f"{prefix} {content}")
	await ctx.send("\n".join(text)[:2000])


@bot.command()
async def search(ctx: commands.Context, *, query: str):
	"""Perform web search via Google CSE (optional)"""
	await ctx.send(f"**Search results for '{query}':**\n{get_google_results(query)}")


def main() -> None:
	token = os.getenv("DISCORD_BOT_TOKEN")
	if not token:
		raise RuntimeError("DISCORD_BOT_TOKEN not set")
	bot.run(token)


if __name__ == "__main__":  # pragma: no cover
	main()