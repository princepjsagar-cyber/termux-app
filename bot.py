import os
import html
import time
import json
import logging
import sqlite3
import shutil
from datetime import datetime
import threading
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from telebot import TeleBot
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, CallbackQuery
from io import BytesIO
try:
    from gtts import gTTS
except Exception:  # pragma: no cover
    gTTS = None  # type: ignore


# Load .env if present
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")  # Google CSE 'cx' value
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Optional fallback
PREFER_SERPAPI = os.getenv("PREFER_SERPAPI", "0").strip() in {"1", "true", "TRUE", "yes", "on"}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional for voice transcription
TTS_LANG = os.getenv("TTS_LANG", "en")

DATA_DIR = os.getenv("DATA_DIR", "/workspace/data")
BACKUP_DIR = os.getenv("BACKUP_DIR", "/workspace/backups")
BACKUP_INTERVAL_HOURS = int(os.getenv("BACKUP_INTERVAL_HOURS", "24"))
BACKUP_KEEP = int(os.getenv("BACKUP_KEEP", "7"))

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


def _ensure_dirs() -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(BACKUP_DIR, exist_ok=True)
    except Exception:
        pass


def _setup_logging() -> None:
    _ensure_dirs()
    log_path = os.path.join(DATA_DIR, "bot.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def _db_path() -> str:
    return os.path.join(DATA_DIR, "bot.db")


def _init_db() -> None:
    _ensure_dirs()
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                query TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                chat_id INTEGER PRIMARY KEY,
                voice_reply_enabled INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()


def _db_count_queries(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(1) FROM queries")
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _migrate_from_jsonl_if_needed() -> None:
    jsonl_path = os.path.join(DATA_DIR, "queries.jsonl")
    if not os.path.exists(jsonl_path):
        return
    try:
        with sqlite3.connect(_db_path()) as conn:
            if _db_count_queries(conn) > 0:
                return
            to_insert: List[Tuple[int, int, str]] = []
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        ts = int(rec.get("ts", int(time.time())))
                        chat_id = int(rec.get("chat_id", 0))
                        query = str(rec.get("query", "")).strip()
                        if chat_id and query:
                            to_insert.append((ts, chat_id, query))
                    except Exception:
                        continue
            if to_insert:
                conn.executemany("INSERT INTO queries (ts, chat_id, query) VALUES (?, ?, ?)", to_insert)
                conn.commit()
                logging.info("Migrated %s records from JSONL to SQLite", len(to_insert))
    except Exception as exc:
        logging.warning("Migration from JSONL failed: %s", exc)


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


def serpapi_search(
    query: str,
    start: int = 1,
    num: int = 3,
    tbm: Optional[str] = None,
    tbs: Optional[str] = None,
) -> Tuple[List[dict], Optional[int]]:
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
    if tbm:
        params["tbm"] = tbm
    if tbs:
        params["tbs"] = tbs
    try:
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        # When searching news (tbm=nws), results live under 'news_results' (sometimes 'top_stories')
        if tbm == "nws":
            raw = data.get("news_results", []) or data.get("top_stories", []) or []
        else:
            raw = data.get("organic_results", []) or []
        items: List[dict] = []
        for r in raw[:num]:
            title = r.get("title") or "No Title"
            link = r.get("link") or "#"
            snippet = (
                r.get("snippet")
                or r.get("date")
                or r.get("about_this_result", {}).get("source", {}).get("description")
                or "No description available"
            )
            items.append({"title": title, "link": link, "snippet": snippet})
        # SerpAPI next page calculation may require paid plan; keep simple.
        next_start = None
        return items, next_start
    except Exception:
        return [], None


def search_web(query: str, start: int = 1, num: int = 3) -> Tuple[List[dict], Optional[int], str]:
    """Try preferred provider first. Returns (items, next_start, provider)."""
    if PREFER_SERPAPI and SERPAPI_KEY:
        items2, next_start2 = serpapi_search(query, start=start, num=num)
        if items2:
            return items2, next_start2, "serpapi"
        items, next_start = google_search(query, start=start, num=num)
        if items:
            return items, next_start, "google"
        return [], None, "none"
    else:
        items, next_start = google_search(query, start=start, num=num)
        if items:
            return items, next_start, "google"
        items2, next_start2 = serpapi_search(query, start=start, num=num)
        if items2:
            return items2, next_start2, "serpapi"
        return [], None, "none"


def _format_results_message(items: List[dict], header: str = "🌐 Here are top results:") -> str:
    lines: List[str] = [header, ""]
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


def _toggle_voice_reply(chat_id: int, enable: bool) -> None:
    try:
        with sqlite3.connect(_db_path()) as conn:
            conn.execute(
                "INSERT INTO settings (chat_id, voice_reply_enabled) VALUES (?, ?) ON CONFLICT(chat_id) DO UPDATE SET voice_reply_enabled=excluded.voice_reply_enabled",
                (chat_id, 1 if enable else 0),
            )
            conn.commit()
    except Exception:
        pass


def _is_voice_reply_enabled(chat_id: int) -> bool:
    try:
        with sqlite3.connect(_db_path()) as conn:
            cur = conn.execute("SELECT voice_reply_enabled FROM settings WHERE chat_id=?", (chat_id,))
            row = cur.fetchone()
            return bool(row and int(row[0]) == 1)
    except Exception:
        return False


def _synthesize_voice(text: str) -> Optional[bytes]:
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang=TTS_LANG)
        buf = BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception as exc:
        logging.warning("TTS failed: %s", exc)
        return None


@bot.message_handler(commands=["rt"])  # type: ignore[arg-type]
def handle_realtime(message: Message) -> None:
    if not SERPAPI_KEY:
        bot.reply_to(message, "Real-time search requires SerpAPI. Configure SERPAPI_KEY.")
        return
    args = (message.text or "").split(maxsplit=1)
    query = (args[1] if len(args) > 1 else "").strip()
    if not query:
        bot.reply_to(message, "Usage: /rt <query>")
        return
    chat_id = message.chat.id
    if _rate_limited(chat_id, min_interval_seconds=1.0):
        bot.reply_to(message, "⏳ Slow down a bit—please wait a moment.")
        return
    bot.send_chat_action(chat_id, "typing")
    _log_query(chat_id, f"/rt {query}")
    items, _ = serpapi_search(query, start=1, num=3)
    if not items:
        bot.reply_to(message, "No live results found.")
        return
    text = _format_results_message(items, header="⚡ Live search results:")
    if _is_voice_reply_enabled(chat_id):
        audio_bytes = _synthesize_voice("; ".join([i.get("title", "") for i in items]))
        if audio_bytes:
            bot.send_voice(chat_id, audio=BytesIO(audio_bytes), caption=text)
            return
    bot.send_message(chat_id, text, disable_web_page_preview=True)


@bot.message_handler(commands=["news"])  # type: ignore[arg-type]
def handle_news(message: Message) -> None:
    if not SERPAPI_KEY:
        bot.reply_to(message, "News search requires SerpAPI. Configure SERPAPI_KEY.")
        return
    args = (message.text or "").split(maxsplit=1)
    topic = (args[1] if len(args) > 1 else "").strip()
    if not topic:
        bot.reply_to(message, "Usage: /news <topic>")
        return
    chat_id = message.chat.id
    if _rate_limited(chat_id, min_interval_seconds=1.0):
        bot.reply_to(message, "⏳ Slow down a bit—please wait a moment.")
        return
    bot.send_chat_action(chat_id, "typing")
    _log_query(chat_id, f"/news {topic}")
    # SerpAPI news tab: tbm=nws; optionally tbs for recency (e.g., qdr:h)
    items, _ = serpapi_search(topic, start=1, num=3, tbm="nws")
    if not items:
        bot.reply_to(message, "No news found.")
        return
    text = _format_results_message(items, header="🗞️ Top news:")
    if _is_voice_reply_enabled(chat_id):
        audio_bytes = _synthesize_voice("; ".join([i.get("title", "") for i in items]))
        if audio_bytes:
            bot.send_voice(chat_id, audio=BytesIO(audio_bytes), caption=text)
            return
    bot.send_message(chat_id, text, disable_web_page_preview=True)


@bot.message_handler(commands=["voice_on"])  # type: ignore[arg-type]
def handle_voice_on(message: Message) -> None:
    _toggle_voice_reply(message.chat.id, True)
    bot.reply_to(message, "🎙️ Voice replies enabled.")


@bot.message_handler(commands=["voice_off"])  # type: ignore[arg-type]
def handle_voice_off(message: Message) -> None:
    _toggle_voice_reply(message.chat.id, False)
    bot.reply_to(message, "🔇 Voice replies disabled.")

@bot.message_handler(commands=["start", "help"])  # type: ignore[arg-type]
def handle_start(message: Message) -> None:
    bot.send_message(
        message.chat.id,
        (
            "👋 Hi! Send me any question or keywords and I'll search the web for you.\n\n"
            "Tips:\n"
            "- Be specific (e.g., 'python dataclass default factory').\n"
            "- Tap ‘More results’ to paginate.\n"
            "- I keep it safe-search by default.\n"
            "- Use /news <topic> for real-time news, /rt <query> for live search (SerpAPI)."
        ),
        disable_web_page_preview=True,
    )


def _ensure_data_dir() -> None:
    _ensure_dirs()


def _log_query(chat_id: int, query: str) -> None:
    _ensure_dirs()
    ts = int(time.time())
    # Append to SQLite
    try:
        with sqlite3.connect(_db_path()) as conn:
            conn.execute("INSERT INTO queries (ts, chat_id, query) VALUES (?, ?, ?)", (ts, chat_id, query))
            conn.commit()
    except Exception as exc:
        logging.warning("Failed to write to DB: %s", exc)
    # Also append to JSONL for redundancy
    record = {"ts": ts, "chat_id": chat_id, "query": query}
    try:
        with open(os.path.join(DATA_DIR, "queries.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _read_recent_queries(chat_id: int, limit: int = 5) -> List[dict]:
    # Prefer DB for reads
    try:
        with sqlite3.connect(_db_path()) as conn:
            cur = conn.execute(
                "SELECT ts, query FROM queries WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            )
            rows = cur.fetchall()
            rows.reverse()
            return [{"ts": ts, "chat_id": chat_id, "query": query} for (ts, query) in rows]
    except Exception:
        pass
    # Fallback to JSONL
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


def _backup_db() -> None:
    src = _db_path()
    if not os.path.exists(src):
        return
    try:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(BACKUP_DIR, f"bot_{ts}.db")
        shutil.copy2(src, dst)
        # Rotate old backups
        backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.endswith('.db')])
        if len(backups) > BACKUP_KEEP:
            for old in backups[: len(backups) - BACKUP_KEEP]:
                try:
                    os.remove(os.path.join(BACKUP_DIR, old))
                except Exception:
                    pass
        logging.info("Backup created: %s", dst)
    except Exception as exc:
        logging.warning("Backup failed: %s", exc)


def _backup_worker() -> None:
    while True:
        try:
            _backup_db()
        except Exception:
            pass
        time.sleep(max(3600, BACKUP_INTERVAL_HOURS * 3600))


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

    if _is_voice_reply_enabled(chat_id):
        audio_bytes = _synthesize_voice("; ".join([i.get("title", "") for i in items]))
        if audio_bytes:
            bot.send_voice(chat_id, audio=BytesIO(audio_bytes), caption=text, reply_markup=kb)
            return
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
    _setup_logging()
    logging.info("Starting bot...")
    _init_db()
    _migrate_from_jsonl_if_needed()

    # Start backup thread
    t = threading.Thread(target=_backup_worker, name="backup-worker", daemon=True)
    t.start()

    # Resilient polling loop to run 24/7
    while True:
        try:
            bot.infinity_polling(
                timeout=20,
                long_polling_timeout=20,
                allowed_updates=["message", "callback_query"],
                skip_pending=True,
            )
        except Exception as exc:
            logging.error("Polling crashed: %s", exc)
            time.sleep(3)
            continue


if __name__ == "__main__":
    main()

