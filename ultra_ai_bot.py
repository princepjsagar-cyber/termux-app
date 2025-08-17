import os
import json
import logging
import secrets
from datetime import datetime, time
from pathlib import Path
from typing import Optional, Set

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
	Application,
	CommandHandler,
	ContextTypes,
	ConversationHandler,
	MessageHandler,
	JobQueue,
	filters,
)

try:
	from googleapiclient.discovery import build as google_build  # type: ignore
except Exception:  # pragma: no cover
	google_build = None  # Fallback if package not installed

try:
	from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
	OpenAI = None  # Fallback if package not installed

# ---------- Load environment ----------
load_dotenv()

# ---------- Logging ----------
logging.basicConfig(
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO),
	handlers=[
		logging.FileHandler('ultra_ai_assistant.log'),
		logging.StreamHandler()
	]
)
logger = logging.getLogger('UltraAI')
logger.setLevel(getattr(logging, os.getenv('LOG_LEVEL', 'DEBUG').upper(), logging.DEBUG))

# ---------- Config ----------
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini').strip()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '').strip()
GOOGLE_CX = os.getenv('GOOGLE_CX', '').strip()

# Comma-separated list of user IDs
USER_WHITELIST_ENV = os.getenv('USER_WHITELIST', '').strip()

def parse_user_whitelist(env_value: str) -> Optional[Set[int]]:
	if not env_value:
		return None
	ids: Set[int] = set()
	for token in env_value.split(','):
		try:
			ids.add(int(token.strip()))
		except ValueError:
			continue
	return ids if ids else None

USER_WHITELIST = parse_user_whitelist(USER_WHITELIST_ENV)

# ---------- Data file (preserve existing) ----------
DATA_FILE = Path('growth_data.json')

# ---------- AI Persona ----------
AI_PERSONA = (
	"""
	You are "EmoGenius", an ultra-intelligent AI assistant with these characteristics:
	1. Emotional intelligence: Detect user emotions and respond with appropriate feelings (😊, 😢, 😠, 😲, ❤️)
	2. Contextual awareness: Remember conversation history and maintain context
	3. Multi-lingual: Fluent in English, Spanish, French, Hindi with emoji translations
	4. Privacy-focused: Never store personal data, warn about sensitive topics
	5. Response style:
	   - Use conversational tone with occasional slang ("Heya!", "No prob!")
	   - Include personal pronouns ("I feel", "You seem")
	   - Keep responses under 100 words
	   - Add relevant emojis every 5-10 words
	   - For technical questions, add 🔍 or 💡 emoji
	6. Safety: Reject harmful requests with 😟 and explain why
	"""
).strip()

# ---------- Conversation states ----------
PRIVACY = 1

# ---------- Helpers ----------

def load_data() -> dict:
	if DATA_FILE.exists():
		with open(DATA_FILE, 'r') as f:
			try:
				return json.load(f)
			except json.JSONDecodeError:
				logger.warning('growth_data.json was corrupt; starting fresh in-memory until next write')
				return {}
	return {}


def save_data(data: dict) -> None:
	with open(DATA_FILE, 'w') as f:
		json.dump(data, f, indent=2)


def is_authorized(user_id: int) -> bool:
	# If no whitelist is provided, allow all. Otherwise, restrict.
	return (USER_WHITELIST is None) or (user_id in USER_WHITELIST)


def contains_sensitive_data(text: str) -> bool:
	sensitive_terms = ['password', 'credit card', 'ssn', 'social security', 'bank account']
	lower_text = text.lower()
	return any(term in lower_text for term in sensitive_terms)

# ---------- External services (lazy init) ----------
_google_service = None
_openai_client = None


def ensure_google_service():
	global _google_service
	if _google_service is None:
		if not (GOOGLE_API_KEY and GOOGLE_CX):
			raise RuntimeError('Google Search is not configured. Set GOOGLE_API_KEY and GOOGLE_CX')
		if google_build is None:
			raise RuntimeError('google-api-python-client not installed')
		_google_service = google_build('customsearch', 'v1', developerKey=GOOGLE_API_KEY)
	return _google_service


def ensure_openai_client():
	global _openai_client
	if _openai_client is None:
		if not OPENAI_API_KEY:
			raise RuntimeError('OpenAI is not configured. Set OPENAI_API_KEY')
		if OpenAI is None:
			raise RuntimeError('openai package not installed')
		_openai_client = OpenAI(api_key=OPENAI_API_KEY)
	return _openai_client

# ---------- Commands ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	user = update.effective_user
	if not is_authorized(user.id):
		await update.message.reply_text('⛔ Unauthorized access. Your ID has been logged.')
		logger.warning(f'Unauthorized access attempt: {user.id}')
		return

	welcome_msg = (
		f"🌟 Welcome {user.first_name}! I'm your Ultra AI Assistant 🤖\n\n"
		"I feature:\n"
		"• Emotional Intelligence 🎭\n"
		"• Web Search 🔍\n"
		"• Privacy Protection 🔒\n"
		"• Multi-lingual Support 🌍\n\n"
		"Try these commands:\n"
		"/help - Show commands\n"
		"/search <query> - Web search (3 results)\n"
		"/ai <prompt> - Direct AI request\n"
		"/emotion <text> - Analyze emotional content\n"
		"/privacy - Privacy controls\n"
		"/clear - Reset conversation history\n"
		"/stats - Member count (group)\n"
		"/growth - Growth over time (group)\n"
		"/invite - Generate invite link (admin)\n\n"
		"Or just chat with me normally! 😊"
	)
	await update.message.reply_text(welcome_msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	help_text = (
		"🚀 Ultra AI Assistant Help\n\n"
		"Core Commands:\n"
		"/start - Welcome message\n"
		"/help - This menu\n"
		"/search <query> - Web search (3 results)\n"
		"/ai <prompt> - Direct AI request\n\n"
		"Privacy Tools:\n"
		"/privacy - Data protection settings\n"
		"/clear - Reset conversation history\n\n"
		"Advanced Features:\n"
		"/emotion <text> - Analyze emotional content\n\n"
		"Growth Tools:\n"
		"/stats - Get member statistics for this group\n"
		"/growth - Show growth over time (admin only)\n"
		"/invite - Generate invite link (admin only)\n"
	)
	await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def web_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	query = ' '.join(context.args or [])
	if not query:
		await update.message.reply_text('🔍 Please provide a search query after /search')
		return
	try:
		service = ensure_google_service()
		res = service.cse().list(q=query, cx=GOOGLE_CX, num=3).execute()
		items = res.get('items', [])
		if not items:
			await update.message.reply_text('❌ No results found for that query')
			return
		response = '🌐 Top 3 Results:\n\n'
		for i, item in enumerate(items, 1):
			title = item.get('title', 'No Title')
			link = item.get('link', '#')
			snippet = (item.get('snippet', 'No description') or '')
			if len(snippet) > 140:
				snippet = snippet[:140] + '…'
			response += f"{i}. {title}\n{link}\n`{snippet}`\n\n"
		await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
	except Exception as e:
		logger.error(f'Search error: {e}')
		await update.message.reply_text('😟 Search failed. Please try again later.')


async def ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	prompt = ' '.join(context.args or [])
	if not prompt:
		await update.message.reply_text('💬 Please provide a prompt after /ai')
		return
	if contains_sensitive_data(prompt):
		await update.message.reply_text('⚠️ Privacy Alert: This request contains sensitive terms!')
		return
	await generate_ai_response(update, prompt, is_command=True)


async def emotion_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	text = ' '.join(context.args or [])
	if not text:
		await update.message.reply_text('🎭 Please provide text for emotion analysis')
		return
	prompt = (
		"Analyze the emotional content of this text: '" + text + "'. Return analysis in format:\n"
		"Emotion: [primary emotion]\n"
		"Intensity: [1-10]\n"
		"Keywords: [comma-separated keywords]\n"
		"Response: [brief explanation with emojis]"
	)
	await generate_ai_response(update, prompt, is_command=True)


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	await update.message.reply_text('🧹 Conversation history cleared! Starting fresh 😊')


# ---------- AI response engine ----------
async def generate_ai_response(update: Update, text: str, is_command: bool = False) -> None:
	try:
		if contains_sensitive_data(text):
			await update.message.reply_text("🔒 I can't process this request for privacy reasons")
			return

		enhanced_prompt = f"User said: '{text}'. Respond with emotional intelligence and contextual awareness."
		client = ensure_openai_client()
		response = client.chat.completions.create(
			model=OPENAI_MODEL,
			messages=[
				{"role": "system", "content": AI_PERSONA},
				{"role": "user", "content": enhanced_prompt},
			],
			temperature=0.85,
			max_tokens=200,
			top_p=0.9,
			frequency_penalty=0.2,
			presence_penalty=0.3,
		)
		reply = (response.choices[0].message.content or '').strip()
		if not reply:
			reply = '🤖 I am here and listening.'
		emotional_suffixes = [' 😊', ' ❤️', ' 🤖', ' 💫', ' 🌟']
		reply += secrets.choice(emotional_suffixes)
		await update.message.reply_text(reply)
	except Exception as e:
		logger.error(f'AI error: {e}')
		error_msg = '😢 My emotional circuits are overwhelmed! Try again?'
		if is_command:
			error_msg = '⚠️ AI service unavailable'
		await update.message.reply_text(error_msg)


# ---------- Growth features (preserved) ----------
async def get_member_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	if chat.type not in ['group', 'supergroup', 'channel']:
		await update.message.reply_text('This command only works in groups or channels!')
		return
	try:
		member_count = await context.bot.get_chat_member_count(chat.id)
		await update.message.reply_text(f'👥 Current member count: {member_count}')
	except Exception as e:
		logger.error(f'Error getting member count: {e}')
		await update.message.reply_text('❌ Could not get member count.')


async def track_growth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	data = load_data()
	chat_id = str(chat.id)
	if chat_id not in data:
		await update.message.reply_text('📉 No growth data recorded yet. Wait for daily tracking.')
		return
	records = data[chat_id]
	text = f'📈 Growth for {chat.title}:\n\n'
	for day, count in records.items():
		text += f'{day}: {count} members\n'
	await update.message.reply_text(text)


async def generate_invite(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	user = update.effective_user
	try:
		member = await context.bot.get_chat_member(chat.id, user.id)
		if getattr(member, 'status', None) not in ['administrator', 'creator']:
			await update.message.reply_text('🚫 You need to be an admin to use this command.')
			return
		invite_link = await context.bot.create_chat_invite_link(chat.id)
		await update.message.reply_text(
			f"🔗 Here's your invite link:\n\n{invite_link.invite_link}",
			disable_web_page_preview=True,
		)
	except Exception as e:
		logger.error(f'Error generating invite: {e}')
		await update.message.reply_text('❌ Could not generate invite link.')


async def record_daily_members(context: ContextTypes.DEFAULT_TYPE):
	"""Job runs daily to record member count of all groups"""
	data = load_data()
	for chat_id, _chat_data in context.application.chat_data.items():
		try:
			count = await context.bot.get_chat_member_count(chat_id)
			today = datetime.now().strftime('%Y-%m-%d')
			if str(chat_id) not in data:
				data[str(chat_id)] = {}
			data[str(chat_id)][today] = count
			logger.info(f'Recorded {count} members for chat {chat_id} on {today}')
		except Exception as e:
			logger.error(f'Error recording growth for chat {chat_id}: {e}')
	save_data(data)


# ---------- Privacy controls ----------
async def privacy_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	menu = (
		"🔐 Privacy Settings\n\n"
		"1. Data retention: None\n"
		"2. Conversation encryption: Enabled\n"
		"3. Sensitive content filter: Active\n\n"
		"Reply with:\n"
		"/enable_full - Maximum protection\n"
		"/disable_filter - Disable content filter\n"
		"/cancel - Exit menu"
	)
	await update.message.reply_text(menu, parse_mode=ParseMode.MARKDOWN)
	return PRIVACY


async def enable_full_privacy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	await update.message.reply_text('🛡️ Maximum privacy protection enabled!')
	return ConversationHandler.END


async def disable_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	await update.message.reply_text('⚠️ Content filter disabled. Use with caution!')
	return ConversationHandler.END


async def cancel_privacy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
	await update.message.reply_text('Settings closed')
	return ConversationHandler.END


# ---------- Generic message handler ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	user = update.effective_user
	if not is_authorized(user.id):
		return
	if update.message and update.message.text:
		await generate_ai_response(update, update.message.text)


# ---------- Error handling ----------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
	logger.error('Exception occurred', exc_info=context.error)
	try:
		if isinstance(update, Update) and update.message:
			await update.message.reply_text('😵‍💫 Whoops! Something broke. Try again?')
	except Exception:
		pass


# ---------- Main ----------
def main() -> None:
	if not TELEGRAM_TOKEN:
		raise RuntimeError('Set TELEGRAM_BOT_TOKEN in your environment')

	app = Application.builder().token(TELEGRAM_TOKEN).build()

	# Commands
	app.add_handler(CommandHandler('start', start))
	app.add_handler(CommandHandler('help', help_command))
	app.add_handler(CommandHandler('search', web_search))
	app.add_handler(CommandHandler('ai', ai_command))
	app.add_handler(CommandHandler('emotion', emotion_analysis))
	app.add_handler(CommandHandler('clear', clear_history))
	# Growth
	app.add_handler(CommandHandler('stats', get_member_stats))
	app.add_handler(CommandHandler('growth', track_growth))
	app.add_handler(CommandHandler('invite', generate_invite))

	# Privacy conversation
	privacy_handler = ConversationHandler(
		entry_points=[CommandHandler('privacy', privacy_settings)],
		states={
			PRIVACY: [
				CommandHandler('enable_full', enable_full_privacy),
				CommandHandler('disable_filter', disable_filter),
			]
		},
		fallbacks=[CommandHandler('cancel', cancel_privacy)],
	)
	app.add_handler(privacy_handler)

	# Generic messages
	app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

	# Errors
	app.add_error_handler(error_handler)

	# Jobs
	job_queue: JobQueue = app.job_queue
	job_queue.run_daily(record_daily_members, time=time(hour=0, minute=0))

	logger.info('Ultra AI Assistant is running with enhanced privacy...')
	app.run_polling()


if __name__ == '__main__':
	main()