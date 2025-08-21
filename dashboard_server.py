import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Bot Dashboard")

public_dir = os.path.join(os.path.dirname(__file__), "public")
app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")

@app.get("/healthz")
def healthz():
    return {"ok": True}

