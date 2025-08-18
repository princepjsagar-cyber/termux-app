import os
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from . import services as svc


app = FastAPI(title="AI Search Bot Web")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def _startup() -> None:
    svc.init_db()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/search")
async def api_search(query: str = Form(...), start: int = Form(1), num: int = Form(3)):
    items, next_start, provider = svc.search_web(query, start=start, num=num)
    svc.log_query(0, f"[web] {query}")
    return {"items": items, "next_start": next_start, "provider": provider}


@app.post("/api/news")
async def api_news(limit: int = Form(3)):
    items = svc.fetch_ai_news(limit)
    return {"items": items}


@app.post("/api/ask")
async def api_ask(question: str = Form(...)):
    ans = svc.openai_answer(question)
    return {"answer": ans}


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    data = await file.read()
    text = svc.openai_transcribe(data, filename=file.filename)
    return {"text": text}


@app.get("/api/tts")
def api_tts(text: str):
    audio = svc.tts_bytes(text)
    if not audio:
        return JSONResponse({"error": "TTS unavailable"}, status_code=400)
    return StreamingResponse(iter([audio]), media_type="audio/mpeg")

