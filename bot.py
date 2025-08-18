import os
import html
import time
import json
import threading
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from telebot import TeleBot
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, CallbackQuery


# Load .env if present
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")  # Google CSE 'cx' value
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Optional fallback

DATA_DIR = os.getenv("DATA_DIR", "/workspace/data")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing environment variable: TELEGRAM_TOKEN")

# Require at least one search provider configured
has_google = bool(GOOGLE_API_KEY and SEARCH_ENGINE_ID)
has_serpapi = bool(SERPAPI_KEY)
if not (has_google or has_serpapi):
    raise RuntimeError(
        "Configure at least one search provider: either GOOGLE_API_KEY & SEARCH_ENGINE_ID or SERPAPI_KEY"
    )


bot = TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")


# In-memory state per chat for pagination and lightweight rate limiting
_state_lock = threading.Lock()
_chat_state: Dict[int, Dict[str, object]] = {}
_last_request_ts: Dict[int, float] = {}


def _rate_limited(chat_id: int, min_interval_seconds: float = 1.5) -> bool:
    now = time.time()
    last = _last_request_ts.get(chat_id, 0.0)
    if now - last < min_interval_seconds:
        return True
    _last_request_ts[chat_id] = now
    return False


def google_search(query: str, start: int = 1, num: int = 3) -> Tuple[List[dict], Optional[int]]:
    """Call Google Custom Search API.

    Returns (items, next_start) where next_start is the next page start index if available.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num,
        "start": start,
        "safe": "active",
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", []) or []
        # Determine next page start, if provided by API
        next_start: Optional[int] = None
        queries = data.get("queries", {})
        next_page = queries.get("nextPage", [])
        if next_page and isinstance(next_page, list):
            next_start = next_page[0].get("startIndex")
        return items, next_start
    except Exception:
        return [], None


def serpapi_search(query: str, start: int = 1, num: int = 3) -> Tuple[List[dict], Optional[int]]:
    """Optional SerpAPI fallback. Returns items in the same shape as google_search.
    Pagination may be limited depending on plan.
    """
    if not SERPAPI_KEY:
        return [], None
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "num": num,
        "start": max(0, start - 1),  # SerpAPI is 0-based
        "api_key": SERPAPI_KEY,
        "safe": "active",
        "hl": "en",
        "gl": "us",
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        organic = data.get("organic_results", []) or []
        items: List[dict] = []
        for r in organic[:num]:
            items.append({
                "title": r.get("title") or "No Title",
                "link": r.get("link") or "#",
                "snippet": r.get("snippet") or r.get("about_this_result", {}).get("source", {}).get("description") or "No description available",
            })
        # SerpAPI next page calculation may require paid plan; keep simple.
        next_start = None
        return items, next_start
    except Exception:
        return [], None


def search_web(query: str, start: int = 1, num: int = 3) -> Tuple[List[dict], Optional[int], str]:
    """Try Google CSE first, then SerpAPI if configured. Returns (items, next_start, provider)."""
    items, next_start = google_search(query, start=start, num=num)
    if items:
        return items, next_start, "google"
    items2, next_start2 = serpapi_search(query, start=start, num=num)
    if items2:
        return items2, next_start2, "serpapi"
    return [], None, "none"


def _format_results_message(items: List[dict]) -> str:
    lines: List[str] = ["🌐 Here are top results:", ""]
    for item in items[:3]:
        title = html.escape(item.get("title", "No Title"))
        link = item.get("link", "#")
        snippet = html.escape(item.get("snippet", "No description available"))
        # Telegram HTML links
        lines.append(f"🔹 <a href=\"{link}\">{title}</a>\n{snippet}\n")
    return "\n".join(lines)


def _build_more_keyboard(has_more: bool) -> Optional[InlineKeyboardMarkup]:
    if not has_more:
        return None
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton(text="More results ▶️", callback_data="more"))
    return kb


@bot.message_handler(commands=["start", "help"])  # type: ignore[arg-type]
def handle_start(message: Message) -> None:
    bot.send_message(
        message.chat.id,
        (
            "👋 Hi! Send me any question or keywords and I'll search the web for you.\n\n"
            "Tips:\n"
            "- Be specific (e.g., 'python dataclass default factory').\n"
            "- Tap ‘More results’ to paginate.\n"
            "- I keep it safe-search by default."
        ),
        disable_web_page_preview=True,
    )


def _ensure_data_dir() -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass


def _log_query(chat_id: int, query: str) -> None:
    _ensure_data_dir()
    record = {
        "ts": int(time.time()),
        "chat_id": chat_id,
        "query": query,
    }
    try:
        with open(os.path.join(DATA_DIR, "queries.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _read_recent_queries(chat_id: int, limit: int = 5) -> List[dict]:
    path = os.path.join(DATA_DIR, "queries.jsonl")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        records: List[dict] = []
        for line in reversed(lines):
            try:
                rec = json.loads(line)
                if rec.get("chat_id") == chat_id:
                    records.append(rec)
                    if len(records) >= limit:
                        break
            except Exception:
                continue
        return list(reversed(records))
    except Exception:
        return []


@bot.message_handler(commands=["recent"])  # type: ignore[arg-type]
def handle_recent(message: Message) -> None:
    args = (message.text or "").split()
    try:
        limit = int(args[1]) if len(args) > 1 else 5
        limit = max(1, min(20, limit))
    except Exception:
        limit = 5
    recents = _read_recent_queries(message.chat.id, limit=limit)
    if not recents:
        bot.reply_to(message, "No recent queries found.")
        return
    lines = ["🕘 Your recent queries:", ""]
    for r in recents:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.get("ts", 0)))
        q = html.escape(str(r.get("query", "")))
        lines.append(f"• {ts}: {q}")
    bot.reply_to(message, "\n".join(lines))


@bot.message_handler(func=lambda m: bool(m.text and m.text.strip()))  # type: ignore[arg-type]
def handle_query(message: Message) -> None:
    chat_id = message.chat.id
    query = message.text.strip()

    if _rate_limited(chat_id):
        bot.reply_to(message, "⏳ Easy there — please wait a moment before sending another query.")
        return

    bot.send_chat_action(chat_id, "typing")

    _log_query(chat_id, query)

    items, next_start, provider = search_web(query, start=1, num=3)
    if not items:
        bot.reply_to(message, "🔍 I couldn't find relevant results. Try rephrasing your question?")
        return

    text = _format_results_message(items)
    kb = _build_more_keyboard(has_more=bool(next_start) and provider == "google")

    with _state_lock:
        _chat_state[chat_id] = {"query": query, "next_start": next_start, "provider": provider}

    bot.send_message(chat_id, text, reply_markup=kb, disable_web_page_preview=True)


@bot.callback_query_handler(func=lambda c: c.data == "more")  # type: ignore[arg-type]
def handle_more(callback: CallbackQuery) -> None:
    chat_id = callback.message.chat.id if callback.message else callback.from_user.id

    with _state_lock:
        state = _chat_state.get(chat_id, {})
        query = state.get("query")
        next_start = state.get("next_start")
        provider = state.get("provider")

    if not query or not next_start or provider != "google":
        bot.answer_callback_query(callback.id, "No more results available.")
        return

    bot.answer_callback_query(callback.id)
    bot.send_chat_action(chat_id, "typing")

    items, new_next_start = google_search(str(query), start=int(next_start), num=3)
    if not items:
        bot.answer_callback_query(callback.id, "No more results.")
        return

    text = _format_results_message(items)
    kb = _build_more_keyboard(has_more=bool(new_next_start))

    with _state_lock:
        _chat_state[chat_id] = {"query": query, "next_start": new_next_start, "provider": "google"}

    # Edit the previous bot message if possible; otherwise send a new one
    try:
        if callback.message:
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=callback.message.message_id,
                text=text,
                reply_markup=kb,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        else:
            bot.send_message(chat_id, text, reply_markup=kb, disable_web_page_preview=True)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=kb, disable_web_page_preview=True)


def main() -> None:
    print("Bot is running...")
    bot.infinity_polling(timeout=20, long_polling_timeout=20, allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()

