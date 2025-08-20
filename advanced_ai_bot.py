import os
import logging
from typing import Dict, List, Tuple
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ai_services import AIServices

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("advanced_ai_bot")


class AdvancedAIBot:
	"""High-level AI bot helper.

	- Uses environment variables for configuration (BOT_USERNAME optional)
	- Relies on AIServices for text and image generation
	- Maintains an in-memory conversation map per user
	- No persistence; does not touch existing data
	"""

	def __init__(self) -> None:
		self.user_id_to_session: Dict[int, Dict] = {}
		self.ai = AIServices()
		self.bot_username = os.getenv("BOT_USERNAME", "Assistant")

	def _ensure_session(self, user_id: int) -> Dict:
		if user_id not in self.user_id_to_session:
			self.user_id_to_session[user_id] = {
				"conversation": [],
				"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
				"temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
				"max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "512")),
			}
		return self.user_id_to_session[user_id]

	def get_welcome(self, first_name: str) -> Tuple[str, InlineKeyboardMarkup]:
		text = (
			f"👋 Hello {first_name}! I'm {self.bot_username}, your advanced AI assistant.\n\n"
			"I can help you with:\n"
			"- Answering questions\n"
			"- Generating creative content\n"
			"- Analyzing data\n"
			"- And much more!\n\n"
			"Try these commands:\n"
			"/chat - Start a conversation\n"
			"/image - Generate AI images\n"
			"/summarize - Summarize text\n"
			"/translate - Translate text\n"
			"/code - Generate or explain code\n"
			"/settings - Configure bot behavior"
		)
		keyboard = [
			[InlineKeyboardButton("💬 Start Chat", callback_data="start_chat")],
			[InlineKeyboardButton("🖼 Generate Image", callback_data="generate_image")],
			[InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
		]
		return text, InlineKeyboardMarkup(keyboard)

	def help_text(self) -> str:
		return (
			"🤖 Advanced AI Bot Help\n\n"
			"Available Commands:\n"
			"/start - Welcome message\n"
			"/help - Show this help\n"
			"/chat - Start an AI conversation\n"
			"/image [prompt] - Generate AI images\n"
			"/summarize [text] - Summarize text\n"
			"/translate [lang] [text] - Translate text\n"
			"/code [prompt] - Generate or explain code\n"
			"/settings - Configure bot behavior\n\n"
			"You can also just type your message to chat!"
		)

	def generate_reply(self, user_id: int, user_message: str, standalone: bool = False) -> str:
		session = self._ensure_session(user_id)
		if standalone:
			prompt = user_message
			return self.ai.generate_text(prompt)
		# With conversation context
		session["conversation"].append({"role": "user", "content": user_message})
		reply = self.ai.generate_text(user_message)
		session["conversation"].append({"role": "assistant", "content": reply})
		if len(session["conversation"]) > 10:
			session["conversation"] = session["conversation"][-10:]
		return reply

	def generate_image(self, prompt: str) -> str:
		return self.ai.generate_image(prompt)

	def clear_history(self, user_id: int) -> None:
		self.user_id_to_session.get(user_id, {}).update({"conversation": []})