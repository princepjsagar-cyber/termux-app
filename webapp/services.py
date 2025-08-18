import os
import time
import json
import sqlite3
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
PREFER_SERPAPI = os.getenv("PREFER_SERPAPI", "0").strip() in {"1", "true", "TRUE", "yes", "on"}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TTS_LANG = os.getenv("TTS_LANG", "en")

DATA_DIR = os.getenv("DATA_DIR", "/workspace/data")
BACKUP_DIR = os.getenv("BACKUP_DIR", "/workspace/backups")


_cache: Dict[tuple, tuple] = {}
_session_singleton: Optional[requests.Session] = None

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "60"))
CACHE_MAX_KEYS = int(os.getenv("CACHE_MAX_KEYS", "200"))


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)


def db_path() -> str:
    ensure_dirs()
    return os.path.join(DATA_DIR, "bot.db")


def init_db() -> None:
    ensure_dirs()
    with sqlite3.connect(db_path()) as conn:
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
        conn.commit()


def log_query(chat_id: int, query: str) -> None:
    ensure_dirs()
    ts = int(time.time())
    try:
        with sqlite3.connect(db_path()) as conn:
            conn.execute("INSERT INTO queries (ts, chat_id, query) VALUES (?, ?, ?)", (ts, chat_id, query))
            conn.commit()
    except Exception:
        pass
    try:
        with open(os.path.join(DATA_DIR, "queries.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": ts, "chat_id": chat_id, "query": query}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def session() -> requests.Session:
    global _session_singleton
    if _session_singleton is not None:
        return _session_singleton
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    _session_singleton = s
    return s


def cache_get(key: tuple):
    hit = _cache.get(key)
    if not hit:
        return None
    ts, value = hit
    if time.time() - ts > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return value


def cache_set(key: tuple, value):
    if len(_cache) > CACHE_MAX_KEYS:
        try:
            _cache.pop(next(iter(_cache)))
        except Exception:
            _cache.clear()
    _cache[key] = (time.time(), value)


def google_search(query: str, start: int = 1, num: int = 3) -> Tuple[List[dict], Optional[int]]:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": SEARCH_ENGINE_ID, "q": query, "num": num, "start": start, "safe": "active"}
    key = ("google", query, start, num)
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        resp = session().get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", []) or []
        next_start = None
        queries = data.get("queries", {})
        next_page = queries.get("nextPage", [])
        if next_page and isinstance(next_page, list):
            next_start = next_page[0].get("startIndex")
        result = (items, next_start)
        cache_set(key, result)
        return result
    except Exception:
        return [], None


def serpapi_search(query: str, start: int = 1, num: int = 3, tbm: Optional[str] = None, tbs: Optional[str] = None) -> Tuple[List[dict], Optional[int]]:
    if not SERPAPI_KEY:
        return [], None
    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "num": num, "start": max(0, start - 1), "api_key": SERPAPI_KEY, "safe": "active", "hl": "en", "gl": "us"}
    if tbm:
        params["tbm"] = tbm
    if tbs:
        params["tbs"] = tbs
    key = ("serpapi", query, start, num, tbm or "", tbs or "")
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        resp = session().get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if tbm == "nws":
            raw = data.get("news_results", []) or data.get("top_stories", []) or []
        else:
            raw = data.get("organic_results", []) or []
        items: List[dict] = []
        for r in raw[:num]:
            title = r.get("title") or "No Title"
            link = r.get("link") or "#"
            snippet = r.get("snippet") or r.get("date") or r.get("about_this_result", {}).get("source", {}).get("description") or ""
            items.append({"title": title, "link": link, "snippet": snippet})
        result = (items, None)
        cache_set(key, result)
        return result
    except Exception:
        return [], None


def search_web(query: str, start: int = 1, num: int = 3) -> Tuple[List[dict], Optional[int], str]:
    if PREFER_SERPAPI and SERPAPI_KEY:
        items, next_start = serpapi_search(query, start=start, num=num)
        if items:
            return items, next_start, "serpapi"
        items2, next_start2 = google_search(query, start=start, num=num)
        if items2:
            return items2, next_start2, "google"
        return [], None, "none"
    else:
        items2, next_start2 = google_search(query, start=start, num=num)
        if items2:
            return items2, next_start2, "google"
        items, next_start = serpapi_search(query, start=start, num=num)
        if items:
            return items, next_start, "serpapi"
        return [], None, "none"


def fetch_ai_news(max_results: int = 3) -> List[dict]:
    if not NEWS_API_KEY:
        return []
    since = time.strftime("%Y-%m-%d", time.localtime(time.time() - 24*3600))
    url = f"https://newsapi.org/v2/everything?q=artificial+intelligence+OR+ai&from={since}&sortBy=popularity&pageSize={max_results}"
    headers = {"X-Api-Key": NEWS_API_KEY}
    key = ("newsapi", max_results, since)
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        resp = session().get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            return []
        items: List[dict] = []
        for a in data.get("articles", [])[:max_results]:
            title = a.get("title") or "No Title"
            link = a.get("url") or "#"
            snippet = a.get("description") or a.get("source", {}).get("name") or ""
            items.append({"title": title, "link": link, "snippet": snippet})
        cache_set(key, items)
        return items
    except Exception:
        return []


def openai_answer(question: str, context: str = "") -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": OPENAI_MODEL, "messages": [{"role": "system", "content": "You're a helpful assistant. Answer concisely."}, {"role": "user", "content": (context + question).strip()}], "temperature": 0.7}
        resp = session().post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return content or None
    except Exception:
        return None


def openai_transcribe(voice_bytes: bytes, filename: str = "audio.ogg") -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        import mimetypes
        from uuid import uuid4
        boundary = f"----WebKitFormBoundary{uuid4().hex}"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": f"multipart/form-data; boundary={boundary}"}
        body = BytesIO()
        def write_field(name: str, value: str):
            body.write(f"--{boundary}\r\n".encode()); body.write(f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode()); body.write(value.encode()); body.write(b"\r\n")
        def write_file(name: str, filename: str, content: bytes, content_type: str):
            body.write(f"--{boundary}\r\n".encode()); body.write((f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n" f"Content-Type: {content_type}\r\n\r\n").encode()); body.write(content); body.write(b"\r\n")
        write_field("model", "whisper-1")
        content_type = mimetypes.guess_type(filename)[0] or "audio/ogg"
        write_file("file", filename, voice_bytes, content_type)
        body.write(f"--{boundary}--\r\n".encode())
        resp = session().post("https://api.openai.com/v1/audio/transcriptions", headers=headers, data=body.getvalue(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("text") or "").strip() or None
    except Exception:
        return None


try:
    from gtts import gTTS
except Exception:
    gTTS = None  # type: ignore


def tts_bytes(text: str) -> Optional[bytes]:
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang=TTS_LANG)
        out = BytesIO(); tts.write_to_fp(out)
        return out.getvalue()
    except Exception:
        return None

