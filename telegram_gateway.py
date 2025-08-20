import logging
import os
import json
from typing import Optional, Set, Dict

from neon_bot import get_bot
from ai_services import AIServices

try:
	import redis  # type: ignore
except Exception:
	redis = None  # type: ignore

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, CallbackQueryHandler, filters
from telegram.constants import ChatMemberStatus, ChatType
from telegram import InlineKeyboardMarkup, InlineKeyboardButton

logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram_gateway")

_ai = AIServices()


def get_redis_client():
	redis_url = os.getenv("REDIS_URL")
	if not redis_url or redis is None:
		return None
	try:
		return redis.from_url(redis_url, decode_responses=True)
	except Exception:
		return None


def get_admin_ids() -> Set[str]:
	raw = os.getenv("ADMIN_IDS", "").strip()
	if not raw:
		return set()
	return {p.strip() for p in raw.split(",") if p.strip()}


# Localization minimal
LOCALE: Dict[str, Dict[str, str]] = {
	"en": {
		"welcome": "Hello! I'm online 24/7. Try /help for commands.",
		"help": (
			"Commands:\n"
			"/ping - check if online\n"
			"/stats - group member count in groups; bot stats in private\n"
			"/groupstats - member count for this chat\n"
			"/growth - show growth placeholder (admin)\n"
			"/invite - generate invite link (admin)\n"
			"/myinvite - your referral link\n"
			"/leaderboard - top referrers\n"
			"/optin - receive daily content\n"
			"/optout - stop daily content\n"
			"/ai <prompt> - ask AI\n"
			"/summarize <text> - summarize\n"
			"/translate <lang> <text> - translate\n"
			"/broadcast <msg> - admin only"
		),
	},
	"hi": {
		"welcome": "नमस्ते! मैं 24/7 ऑनलाइन हूँ। कमांड के लिए /help टाइप करें।",
		"help": (
			"कमांड:\n"
			"/ping - ऑनलाइन जाँचें\n"
			"/stats - समूह में सदस्य गिनती; निजी में बॉट स्टेटस\n"
			"/groupstats - इस चैट के सदस्य\n"
			"/growth - वृद्धि (एडमिन)\n"
			"/invite - इनवाइट लिंक (एडमिन)\n"
			"/myinvite - आपका रेफरल लिंक\n"
			"/leaderboard - शीर्ष रेफरल्स\n"
			"/optin - दैनिक सामग्री पाएं\n"
			"/optout - बंद करें\n"
			"/ai <प्रॉम्प्ट> - AI से पूछें\n"
			"/summarize <टेक्स्ट> - सारांश\n"
			"/translate <भाषा> <टेक्स्ट> - अनुवाद\n"
			"/broadcast <संदेश> - सिर्फ़ एडमिन"
		),
	},
}


def t(key: str, update: Update) -> str:
	lang = (update.effective_user.language_code or "en").split("-")[0] if update.effective_user else "en"
	return LOCALE.get(lang, LOCALE["en"]).get(key, LOCALE["en"].get(key, ""))


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	text: Optional[str] = update.effective_message.text if update.effective_message else None
	user_id: Optional[str] = str(update.effective_user.id) if update.effective_user else None
	if not text:
		return
	client = get_redis_client()
	if client is not None:
		client.rpush("queue:tasks", json.dumps({
			"type": "tg_message",
			"user_id": user_id,
			"text": text,
		}))
	else:
		bot = get_bot()
		await bot.send_message(f"TG({user_id}): {text}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	# Deep link referral handling: /start ref_<id>
	args = context.args or []
	if args and args[0].startswith("ref_"):
		ref_id = args[0][4:]
		client = get_redis_client()
		if client is not None and update.effective_user:
			uid = str(update.effective_user.id)
			if not client.sismember(f"referral:referred", uid):
				client.sadd("referral:referred", uid)
				client.zincrby("referral:leaders", 1, ref_id)
				client.hsetnx(f"referral:user:{ref_id}", "username", update.effective_user.username or "")
	# Localized welcome with quick actions
	keyboard = [[InlineKeyboardButton("My Invite", callback_data="my_invite")]]
	await update.effective_message.reply_text(t("welcome", update), reply_markup=InlineKeyboardMarkup(keyboard))


async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	q = update.callback_query
	await q.answer()
	if q.data == "my_invite":
		await cmd_myinvite(update, context)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	msg = t("help", update)
	if chat and chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}:
		msg = "Group Help:\n" + msg
	else:
		msg = "Direct Help:\n" + msg
	await update.effective_message.reply_text(msg)


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	await update.effective_message.reply_text("pong")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	if chat and chat.type in {ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL}:
		try:
			count = await context.bot.get_chat_member_count(chat.id)
			await update.effective_message.reply_text(f"👥 Current member count: {count}")
		except Exception as exc:
			logger.warning("/stats member count failed: %s", exc)
			await update.effective_message.reply_text("❌ Could not get member count. Make sure I'm an admin if this is a channel.")
		return
	bot = get_bot()
	status = bot.get_status()
	msg = (
		f"Uptime: {status.get('uptime_seconds')}s\n"
		f"Sent: {status.get('messages_sent')}\n"
		f"Queue: {status.get('queue_depth')}\n"
	)
	await update.effective_message.reply_text(msg)


async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	user_id = str(update.effective_user.id) if update.effective_user else ""
	if user_id not in get_admin_ids():
		await update.effective_message.reply_text("Unauthorized")
		return
	text = " ".join(context.args).strip()
	if not text:
		await update.effective_message.reply_text("Usage: /broadcast <message>")
		return
	client = get_redis_client()
	if client is not None:
		client.rpush("queue:tasks", json.dumps({
			"type": "broadcast",
			"from": user_id,
			"text": text,
		}))
		await update.effective_message.reply_text("Broadcast enqueued")
	else:
		bot = get_bot()
		await bot.send_message(f"BROADCAST: {text}")
		await update.effective_message.reply_text("Broadcast sent")


async def cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	prompt = " ".join(context.args).strip()
	if not prompt:
		await update.effective_message.reply_text("Usage: /ai <prompt>")
		return
	reply = _ai.generate_text(prompt)
	await update.effective_message.reply_text(reply[:3500])


async def cmd_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	text = " ".join(context.args).strip()
	if not text:
		await update.effective_message.reply_text("Usage: /summarize <text>")
		return
	reply = _ai.summarize(text)
	await update.effective_message.reply_text(reply[:3500])


async def cmd_translate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	if not context.args:
		await update.effective_message.reply_text("Usage: /translate <lang> <text>")
		return
	target = context.args[0]
	text = " ".join(context.args[1:]).strip()
	if not text:
		await update.effective_message.reply_text("Usage: /translate <lang> <text>")
		return
	reply = _ai.translate(text, target)
	await update.effective_message.reply_text(reply[:3500])


async def cmd_groupstats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	if chat is None or chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL}:
		await update.effective_message.reply_text("This command only works in groups or channels.")
		return
	try:
		count = await context.bot.get_chat_member_count(chat.id)
		await update.effective_message.reply_text(f"👥 Current member count: {count}")
	except Exception as exc:
		logger.warning("get_chat_member_count failed: %s", exc)
		await update.effective_message.reply_text("❌ Could not get member count. Make sure I'm an admin if this is a channel.")


async def _ensure_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
	try:
		member = await context.bot.get_chat_member(chat_id, user_id)
		return member.status in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.OWNER}
	except Exception:
		return False


async def cmd_growth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	user = update.effective_user
	if chat is None or user is None:
		return
	if not await _ensure_admin(context, chat.id, user.id):
		await update.effective_message.reply_text("🚫 Admins only.")
		return
	try:
		count = await context.bot.get_chat_member_count(chat.id)
		await update.effective_message.reply_text(
			"📈 Growth tracking placeholder\n"
			f"Current members: {count}\n\n"
			"To properly track growth:\n1) Add me as admin\n2) I will record daily counts (when enabled)\n3) View graphs with /groupstats"
		)
	except Exception as exc:
		logger.warning("growth failed: %s", exc)
		await update.effective_message.reply_text("❌ An error occurred. Make sure I'm an admin.")


async def cmd_invite(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	chat = update.effective_chat
	user = update.effective_user
	if chat is None or user is None:
		return
	if not await _ensure_admin(context, chat.id, user.id):
		await update.effective_message.reply_text("🚫 Admins only.")
		return
	try:
		link = await context.bot.create_chat_invite_link(chat_id=chat.id)
		await update.effective_message.reply_text(
			f"🔗 Invite link:\n\n{link.invite_link}",
			disable_web_page_preview=True,
		)
	except Exception as exc:
		logger.warning("invite failed: %s", exc)
		await update.effective_message.reply_text("❌ Could not generate invite link. Make sure I'm an admin.")


async def cmd_myinvite(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	bot_username = os.getenv("BOT_USERNAME")
	if not bot_username:
		me = await context.bot.get_me()
		bot_username = me.username
	uid = str(update.effective_user.id) if update.effective_user else ""
	deep = f"https://t.me/{bot_username}?start=ref_{uid}"
	await update.effective_message.reply_text(f"Your referral link:\n{deep}")
	client = get_redis_client()
	if client is not None and update.effective_user:
		client.hset(f"referral:user:{uid}", mapping={"username": update.effective_user.username or ""})


async def cmd_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	client = get_redis_client()
	if client is None:
		await update.effective_message.reply_text("[leaderboard unavailable]")
		return
	leaders = client.zrevrange("referral:leaders", 0, 9, withscores=True)
	lines = []
	for member, score in leaders:
		info = client.hgetall(f"referral:user:{member}") or {}
		u = info.get("username")
		label = f"@{u}" if u else member
		lines.append(f"{int(score)} - {label}")
	await update.effective_message.reply_text("Top referrers:\n" + ("\n".join(lines) or "None yet"))


async def cmd_optin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	client = get_redis_client()
	if client is None or update.effective_user is None:
		await update.effective_message.reply_text("[opt-in unavailable]")
		return
	client.sadd("optin:daily", str(update.effective_user.id))
	await update.effective_message.reply_text("You are now opted-in for daily content.")


async def cmd_optout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	client = get_redis_client()
	if client is None or update.effective_user is None:
		await update.effective_message.reply_text("[opt-out unavailable]")
		return
	client.srem("optin:daily", str(update.effective_user.id))
	await update.effective_message.reply_text("You have opted-out of daily content.")


def main() -> None:
	token = os.getenv("TELEGRAM_BOT_TOKEN")
	if not token:
		raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
	app = ApplicationBuilder().token(token).build()
	app.add_handler(CallbackQueryHandler(on_cb))
	app.add_handler(CommandHandler("start", cmd_start))
	app.add_handler(CommandHandler("help", cmd_help))
	app.add_handler(CommandHandler("ping", cmd_ping))
	app.add_handler(CommandHandler("stats", cmd_stats))
	app.add_handler(CommandHandler("groupstats", cmd_groupstats))
	app.add_handler(CommandHandler("growth", cmd_growth))
	app.add_handler(CommandHandler("invite", cmd_invite))
	app.add_handler(CommandHandler("myinvite", cmd_myinvite))
	app.add_handler(CommandHandler("leaderboard", cmd_leaderboard))
	app.add_handler(CommandHandler("optin", cmd_optin))
	app.add_handler(CommandHandler("optout", cmd_optout))
	app.add_handler(CommandHandler("broadcast", cmd_broadcast))
	app.add_handler(CommandHandler("ai", cmd_ai))
	app.add_handler(CommandHandler("summarize", cmd_summarize))
	app.add_handler(CommandHandler("translate", cmd_translate))
	app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
	logger.info("Telegram gateway started (polling mode)")
	app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":  # pragma: no cover
	main()