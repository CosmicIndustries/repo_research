"""
main.py — FastAPI backend for the GitHub Research Assistant.
Serves the frontend, exposes REST + SSE endpoints, owns ChromaDB + model lifecycle.
"""

import asyncio
import json
import os
import subprocess
import shlex
import time
from pathlib import Path

import chromadb
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from indexer import GitHubIndexer

# ── Init ───────────────────────────────────────────────────────────────────────

DB_PATH    = os.path.expanduser("~/.github_research_db")
MODEL_NAME = "all-MiniLM-L6-v2"       # ~22 MB, fast, good quality
PORT       = 8080

print(f"[boot] Loading embedding model '{MODEL_NAME}'…")
model = SentenceTransformer(MODEL_NAME)
print("[boot] Model ready.")

chroma  = chromadb.PersistentClient(path=DB_PATH)
collection = chroma.get_or_create_collection(
    name="github_repos",
    metadata={"hnsw:space": "cosine"},
)

indexer = GitHubIndexer(collection, model)

app = FastAPI(title="GitHub Research Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request models ─────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    username: str
    token: str = ""

class SearchRequest(BaseModel):
    query: str
    n_results: int = 12
    repo_filter: str = ""

class DeleteRequest(BaseModel):
    username: str

class ExecRequest(BaseModel):
    action: str          # "open_url" | "clone" | "open_editor" | "shell"
    url: str  = ""
    repo: str = ""       # "owner/repo"
    path: str = ""       # file path within repo
    text: str = ""       # chunk text (for shell passthrough)
    cmd:  str = ""       # custom shell template — use {text}, {url}, {path}

CLONE_BASE = os.path.expanduser("~/github_clones")

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "frontend" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/index")
async def start_index(req: IndexRequest, background_tasks: BackgroundTasks):
    status = indexer.get_status()
    if status["state"] == "running":
        raise HTTPException(status_code=409, detail="Indexing already in progress.")
    background_tasks.add_task(indexer.index_user, req.username.strip(), req.token.strip())
    return {"ok": True, "username": req.username}


@app.get("/api/status")
async def get_status():
    return indexer.get_status()


@app.get("/api/status/stream")
async def status_stream():
    """SSE endpoint — pushes status updates every second while indexing."""
    async def generate():
        while True:
            data = json.dumps(indexer.get_status())
            yield f"data: {data}\n\n"
            await asyncio.sleep(1.0)
            if indexer.get_status()["state"] in ("done", "error", "idle"):
                data = json.dumps(indexer.get_status())
                yield f"data: {data}\n\n"
                break

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/search")
async def search(req: SearchRequest):
    if collection.count() == 0:
        return {"results": [], "query": req.query, "total_indexed": 0}
    return indexer.search(req.query, req.n_results, req.repo_filter)


@app.get("/api/repos")
async def list_repos():
    return {"repos": indexer.list_repos()}


@app.post("/api/pause")
async def pause_index():
    indexer.pause()
    return {"ok": True, "paused": True}


@app.post("/api/resume")
async def resume_index():
    indexer.resume()
    return {"ok": True, "paused": False}


@app.post("/api/exec")
async def exec_action(req: ExecRequest):
    """Run OS-level actions triggered from search results."""
    try:
        if req.action == "open_url":
            if not req.url.startswith("https://github.com"):
                raise HTTPException(400, "Only github.com URLs allowed.")
            subprocess.Popen(["xdg-open", req.url])
            return {"ok": True, "msg": f"Opening {req.url}"}

        elif req.action == "clone":
            if not req.repo or "/" not in req.repo:
                raise HTTPException(400, "Invalid repo.")
            os.makedirs(CLONE_BASE, exist_ok=True)
            dest = os.path.join(CLONE_BASE, req.repo.replace("/", "__"))
            if os.path.exists(dest):
                return {"ok": True, "msg": f"Already cloned at {dest}", "path": dest}
            clone_url = f"https://github.com/{req.repo}.git"
            proc = subprocess.Popen(
                ["git", "clone", "--depth=1", clone_url, dest],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            out, _ = proc.communicate(timeout=120)
            if proc.returncode != 0:
                raise HTTPException(500, f"Clone failed: {out[-300:]}")
            return {"ok": True, "msg": f"Cloned to {dest}", "path": dest}

        elif req.action == "open_editor":
            # Try cloned path first, then fall back to xdg-open on URL
            dest = os.path.join(CLONE_BASE, req.repo.replace("/", "__"))
            local_file = os.path.join(dest, req.path) if req.path else dest
            if os.path.exists(local_file):
                subprocess.Popen(["xdg-open", local_file])
                return {"ok": True, "msg": f"Opening {local_file}"}
            elif req.url:
                subprocess.Popen(["xdg-open", req.url])
                return {"ok": True, "msg": f"Not cloned locally — opening GitHub URL."}
            raise HTTPException(404, "File not found locally and no URL provided.")

        elif req.action == "shell":
            if not req.cmd:
                raise HTTPException(400, "cmd is required for shell action.")
            rendered = req.cmd.format(
                text=shlex.quote(req.text),
                url=shlex.quote(req.url),
                path=shlex.quote(req.path),
            )
            proc = subprocess.Popen(
                rendered, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            out, _ = proc.communicate(timeout=30)
            return {"ok": proc.returncode == 0, "output": out[-2000:], "cmd": rendered}

        else:
            raise HTTPException(400, f"Unknown action: {req.action}")

    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Command timed out.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/index")
async def delete_index(req: DeleteRequest):
    n = indexer.delete_username(req.username.strip())
    return {"deleted_chunks": n, "username": req.username}


@app.get("/api/stats")
async def stats():
    repos = indexer.list_repos()
    return {
        "total_chunks": collection.count(),
        "total_repos":  len(repos),
        "repos":        repos,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[boot] Starting server on http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
