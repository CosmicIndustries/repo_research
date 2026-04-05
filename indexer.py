"""
indexer.py — Maximum-throughput GitHub indexer.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  fetch_repos  → gather all repo trees concurrently      │
  │     ↓                                                   │
  │  blob_queue (asyncio.Queue) ← per-blob aiohttp fetches  │
  │     ↓          (Semaphore-gated, connection-pooled)     │
  │  embed_batcher  ← collects chunks, encodes in batches   │
  │     ↓                                                   │
  │  chroma_writer  ← bulk upserts every UPSERT_BATCH items │
  └─────────────────────────────────────────────────────────┘

Concurrency:
  - aiohttp TCPConnector: 32 connections, keep-alive
  - REPO_CONCURRENCY repos processed in parallel
  - FETCH_CONCURRENCY blobs fetched per repo simultaneously
  - EMBED_BATCH_SIZE  chunks embedded per model.encode() call
  - UPSERT_BATCH      payloads buffered before ChromaDB write

Memoization:
  - Two-layer embed cache: in-memory dict (fast) + pickle on disk (persistent)
  - SHA-keyed: unchanged files skip fetch + embed entirely
"""

import asyncio
import base64
import os
import pickle
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import aiohttp

# ── Tuning constants ───────────────────────────────────────────────────────────
REPO_CONCURRENCY  = 6     # repos processed in parallel
FETCH_CONCURRENCY = 16    # concurrent blob GETs per repo
EMBED_BATCH_SIZE  = 96    # chunks per model.encode() call
UPSERT_BATCH      = 60    # payloads buffered before ChromaDB write
MAX_FILE_BYTES    = 150_000
CHUNK_SIZE        = 1_600
CHUNK_OVERLAP     = 200
RATE_SLEEP        = 0.03  # per request delay inside semaphore
EMBED_CACHE_PATH  = os.path.expanduser("~/.github_research_embed_cache.pkl")
LRU_MAX           = 40_000  # max in-memory cache entries

INDEXABLE_EXTENSIONS = {
    ".py",".js",".ts",".tsx",".jsx",".java",".go",".rs",".c",".cpp",
    ".h",".hpp",".cs",".rb",".php",".swift",".kt",".scala",".r",".m",
    ".ex",".exs",".erl",".hs",".lua",".jl",".nim",".zig",".v",
    ".sh",".bash",".zsh",".fish",".ps1",".bat",".cmd",
    ".md",".txt",".rst",".adoc",".org",
    ".json",".yaml",".yml",".toml",".ini",".cfg",".conf",".env",
    ".properties",".xml",
    ".html",".css",".scss",".sass",".less",
    ".sql",".graphql",".prisma",
}
INDEXABLE_NAMES = {
    "Dockerfile","Makefile","Pipfile","Procfile","Gemfile",
    "Vagrantfile","Justfile","Taskfile",".env.example","Cargo.lock",
}


# ── Two-layer embed cache ──────────────────────────────────────────────────────

class EmbedCache:
    """LRU in-memory cache + pickle persistence. Thread-safe."""

    def __init__(self, path: str, maxsize: int = LRU_MAX):
        self._path    = path
        self._maxsize = maxsize
        self._lock    = threading.Lock()
        self._mem: OrderedDict = OrderedDict()
        self._dirty   = False
        self._load()
        # background saver thread
        t = threading.Thread(target=self._autosave, daemon=True)
        t.start()

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path, "rb") as f:
                    data = pickle.load(f)
                # load into LRU (newest first keeps most useful)
                for k, v in list(data.items())[-self._maxsize:]:
                    self._mem[k] = v
        except Exception:
            pass

    def _autosave(self):
        while True:
            time.sleep(30)
            with self._lock:
                if self._dirty:
                    try:
                        with open(self._path, "wb") as f:
                            pickle.dump(dict(self._mem), f, protocol=5)
                        self._dirty = False
                    except Exception:
                        pass

    def get(self, sha: str):
        with self._lock:
            if sha in self._mem:
                self._mem.move_to_end(sha)
                return self._mem[sha]
        return None

    def set(self, sha: str, embeddings: list):
        with self._lock:
            self._mem[sha] = embeddings
            self._mem.move_to_end(sha)
            if len(self._mem) > self._maxsize:
                self._mem.popitem(last=False)
            self._dirty = True

    def __len__(self):
        return len(self._mem)


# ── Chunker (pure Python, no regex) ───────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    lines    = text.splitlines(keepends=True)
    chunks: list[str] = []
    buf: list[str]    = []
    buf_len   = 0
    for line in lines:
        buf.append(line)
        buf_len += len(line)
        if buf_len >= CHUNK_SIZE:
            joined = "".join(buf)
            chunks.append(joined)
            # keep last CHUNK_OVERLAP chars as overlap seed
            overlap = joined[-CHUNK_OVERLAP:]
            buf     = [overlap]
            buf_len = len(overlap)
    if buf:
        chunks.append("".join(buf))
    return [c for c in chunks if c.strip()]


def indexable(path: str) -> bool:
    p = Path(path)
    return p.suffix.lower() in INDEXABLE_EXTENSIONS or p.name in INDEXABLE_NAMES


# ── GitHubIndexer ──────────────────────────────────────────────────────────────

class GitHubIndexer:

    def __init__(self, collection, model):
        self.collection   = collection
        self.model        = model
        self._cache       = EmbedCache(EMBED_CACHE_PATH)
        self._lock        = threading.Lock()
        self._pause_evt   = threading.Event()
        self._pause_evt.set()
        self._embed_pool  = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")
        self.status = {
            "state":"idle","paused":False,"username":"","current_repo":"",
            "progress":"Ready.","indexed_chunks":0,"indexed_files":0,
            "skipped_files":0,"cache_hits":0,"repos_total":0,
            "repos_done":0,"errors":0,
        }

    # ── Status helpers ─────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        with self._lock:
            return dict(self.status)

    def _set(self, **kw):
        with self._lock:
            self.status.update(kw)

    def _inc(self, **kw):
        with self._lock:
            for k, v in kw.items():
                self.status[k] = self.status.get(k, 0) + v

    def pause(self):
        self._pause_evt.clear()
        self._set(paused=True, state="paused", progress="Paused.")

    def resume(self):
        self._pause_evt.set()
        self._set(paused=False, state="running", progress="Resuming…")

    def _check_pause(self) -> bool:
        """Block while paused; return True if we should abort."""
        self._pause_evt.wait()
        return self.get_status()["state"] not in ("running", "paused")

    # ── aiohttp session factory ────────────────────────────────────────────────

    @staticmethod
    def _make_session(token: str) -> aiohttp.ClientSession:
        headers = {
            "Accept":     "application/vnd.github.v3+json",
            "User-Agent": "github-research-assistant",
        }
        if token:
            headers["Authorization"] = f"token {token}"
        connector = aiohttp.TCPConnector(
            limit=40,           # total connection pool
            limit_per_host=20,  # per api.github.com
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=8)
        return aiohttp.ClientSession(
            headers=headers, connector=connector, timeout=timeout
        )

    # ── async HTTP helpers ─────────────────────────────────────────────────────

    async def _get(self, session: aiohttp.ClientSession, url: str, **params):
        for attempt in range(4):
            try:
                async with session.get(url, params=params or None) as resp:
                    if resp.status == 403:
                        reset = int(resp.headers.get("X-RateLimit-Reset", time.time()+60))
                        wait  = max(5, reset - time.time())
                        self._set(progress=f"Rate-limited — waiting {int(wait)}s…")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status == 404:
                        return None
                    if resp.status == 409:   # empty repo
                        return None
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == 3:
                    self._inc(errors=1)
                await asyncio.sleep(0.4 * (2 ** attempt))
        return None

    async def _fetch_repos(self, session, username: str) -> list:
        repos, page = [], 1
        while True:
            data = await self._get(
                session,
                f"https://api.github.com/users/{username}/repos",
                per_page=100, page=page, sort="updated"
            )
            if not data:
                break
            repos.extend(data)
            if len(data) < 100:
                break
            page += 1
        return repos

    async def _fetch_tree(self, session, username: str, repo: str, branch: str) -> list:
        data = await self._get(
            session,
            f"https://api.github.com/repos/{username}/{repo}/git/trees/{branch}",
            recursive=1
        )
        if not data:
            return []
        if data.get("truncated"):
            self._set(progress=f"⚠ Tree truncated: {repo}")
        return [b for b in data.get("tree", []) if b["type"] == "blob"]

    async def _fetch_blob(self, session, username: str, repo: str, path: str) -> Optional[str]:
        data = await self._get(
            session,
            f"https://api.github.com/repos/{username}/{repo}/contents/{path}"
        )
        if not data or not isinstance(data, dict):
            return None
        if data.get("size", 0) > MAX_FILE_BYTES:
            return None
        if data.get("encoding") == "base64":
            try:
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            except Exception:
                return None
        return None

    # ── SHA skip check (async, non-blocking via executor) ─────────────────────

    async def _already_indexed(self, doc_base_id: str, sha: str) -> bool:
        loop = asyncio.get_event_loop()
        def _check():
            try:
                r = self.collection.get(ids=[f"{doc_base_id}::0"])
                return bool(r["ids"]) and r["metadatas"][0].get("sha", "") == sha
            except Exception:
                return False
        return await loop.run_in_executor(None, _check)

    # ── Embedding (CPU-bound → thread pool) ────────────────────────────────────

    def _encode_sync(self, chunks: list[str]) -> list:
        return self.model.encode(chunks, show_progress_bar=False,
                                 batch_size=EMBED_BATCH_SIZE).tolist()

    async def _encode(self, chunks: list[str], sha: str) -> tuple[list, bool]:
        cached = self._cache.get(sha)
        if cached and len(cached) == len(chunks):
            return cached, True
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._embed_pool, self._encode_sync, chunks)
        self._cache.set(sha, result)
        return result, False

    # ── ChromaDB upsert (blocking → executor) ─────────────────────────────────

    async def _upsert(self, payloads: list[dict]):
        if not payloads:
            return
        ids, embeddings, documents, metadatas = [], [], [], []
        for p in payloads:
            ids.extend(p["ids"])
            embeddings.extend(p["embeddings"])
            documents.extend(p["documents"])
            metadatas.extend(p["metadatas"])
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.collection.upsert(
                ids=ids, embeddings=embeddings,
                documents=documents, metadatas=metadatas
            )
        )
        self._inc(
            indexed_files=len(payloads),
            indexed_chunks=len(ids),
        )

    # ── Process one blob ───────────────────────────────────────────────────────

    async def _process_blob(
        self, session, sem: asyncio.Semaphore,
        blob: dict, username: str, repo_name: str, branch: str
    ) -> Optional[dict]:

        # pause gate (non-blocking poll)
        if not self._pause_evt.is_set():
            self._pause_evt.wait()   # blocks thread briefly — fine in executor
        if self._check_pause():
            return None

        path        = blob["path"]
        sha         = blob.get("sha", "")
        doc_base_id = f"{username}/{repo_name}/{path}"

        if await self._already_indexed(doc_base_id, sha):
            self._inc(skipped_files=1)
            return None

        async with sem:
            await asyncio.sleep(RATE_SLEEP)
            content = await self._fetch_blob(session, username, repo_name, path)

        if not content or not content.strip():
            self._inc(skipped_files=1)
            return None

        chunks = chunk_text(content)
        if not chunks:
            self._inc(skipped_files=1)
            return None

        embeddings, was_cached = await self._encode(chunks, sha)
        if was_cached:
            self._inc(cache_hits=1)

        meta_base = {
            "username": username, "repo": f"{username}/{repo_name}",
            "repo_name": repo_name, "path": path,
            "ext": Path(path).suffix.lower(), "sha": sha,
            "url": f"https://github.com/{username}/{repo_name}/blob/{branch}/{path}",
            "branch": branch,
        }
        return {
            "ids":        [f"{doc_base_id}::{i}" for i in range(len(chunks))],
            "embeddings": embeddings,
            "documents":  chunks,
            "metadatas":  [{**meta_base, "chunk": i} for i in range(len(chunks))],
        }

    # ── Process one repo ───────────────────────────────────────────────────────

    async def _process_repo(
        self, session, repo_sem: asyncio.Semaphore,
        repo_data: dict, username: str
    ):
        async with repo_sem:
            repo_name = repo_data["name"]
            branch    = repo_data.get("default_branch", "main")
            done, total = self.status["repos_done"], self.status["repos_total"]
            self._set(current_repo=repo_name,
                      progress=f"[{done}/{total}] Scanning {repo_name}…")

            tree       = await self._fetch_tree(session, username, repo_name, branch)
            candidates = [b for b in tree if indexable(b["path"])]

            # per-repo fetch semaphore
            fetch_sem = asyncio.Semaphore(FETCH_CONCURRENCY)
            tasks     = [
                self._process_blob(session, fetch_sem, blob, username, repo_name, branch)
                for blob in candidates
            ]

            # process in UPSERT_BATCH-sized windows to keep memory bounded
            buf: list[dict] = []
            for coro in asyncio.as_completed(tasks):
                if self._check_pause():
                    return
                try:
                    payload = await coro
                except Exception:
                    self._inc(errors=1)
                    continue
                if payload:
                    buf.append(payload)
                if len(buf) >= UPSERT_BATCH:
                    await self._upsert(buf)
                    buf.clear()
                    s = self.status
                    self._set(progress=(
                        f"[{s['repos_done']}/{s['repos_total']}] {repo_name} · "
                        f"{s['indexed_files']} files · {s['indexed_chunks']} chunks"
                        f" · ✦{s['cache_hits']} cached"
                    ))

            if buf:
                await self._upsert(buf)

            self._inc(repos_done=1)

    # ── Main async pipeline ────────────────────────────────────────────────────

    async def _run(self, username: str, token: str):
        self._set(
            state="running", paused=False, username=username,
            progress="Fetching repository list…",
            indexed_chunks=0, indexed_files=0, skipped_files=0,
            cache_hits=0, repos_total=0, repos_done=0, errors=0,
        )
        async with self._make_session(token) as session:
            repos = await self._fetch_repos(session, username)
            if not repos:
                self._set(state="error", progress=f"No repos found for '{username}'.")
                return

            self._set(repos_total=len(repos),
                      progress=f"Found {len(repos)} repos — launching pipeline…")

            repo_sem = asyncio.Semaphore(REPO_CONCURRENCY)
            await asyncio.gather(*[
                self._process_repo(session, repo_sem, r, username)
                for r in repos
            ])

        s = self.status
        self._set(
            state="done", current_repo="",
            progress=(
                f"✓ {s['indexed_files']} files · {s['indexed_chunks']} chunks · "
                f"{s['cache_hits']} cache hits · {s['skipped_files']} unchanged · "
                f"{s['errors']} errors"
            ),
        )

    # ── Public entry point (called from FastAPI background thread) ─────────────

    def index_user(self, username: str, token: str = ""):
        """Runs a fresh asyncio event loop in the background thread."""
        self._pause_evt.set()
        try:
            asyncio.run(self._run(username, token))
        except Exception as e:
            self._set(state="error", progress=f"Fatal: {e}")

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 12, repo_filter: str = "") -> dict:
        total = self.collection.count()
        if total == 0:
            return {"results": [], "query": query, "total_indexed": 0}
        try:
            if repo_filter:
                matching = self.collection.get(where={"repo": repo_filter})
                n = min(n_results, max(1, len(matching["ids"])))
            else:
                n = min(n_results, total)
            if n == 0:
                return {"results": [], "query": query, "total_indexed": total}

            embedding = self.model.encode([query]).tolist()
            kwargs: dict = {
                "query_embeddings": embedding,
                "n_results": n,
                "include": ["documents", "metadatas", "distances"],
            }
            if repo_filter:
                kwargs["where"] = {"repo": repo_filter}

            raw = self.collection.query(**kwargs)
            return {
                "results": [
                    {
                        "id":       raw["ids"][0][i],
                        "text":     raw["documents"][0][i],
                        "metadata": raw["metadatas"][0][i],
                        "score":    round(1.0 - raw["distances"][0][i], 4),
                    }
                    for i in range(len(raw["ids"][0]))
                ],
                "query": query,
                "total_indexed": total,
            }
        except Exception as e:
            return {"results": [], "query": query, "total_indexed": total, "error": str(e)}

    # ── Repo listing / deletion ────────────────────────────────────────────────

    def list_repos(self) -> list:
        try:
            data = self.collection.get(include=["metadatas"])
            return sorted(set(m.get("repo", "") for m in data["metadatas"] if m.get("repo")))
        except Exception:
            return []

    def delete_username(self, username: str) -> int:
        try:
            data = self.collection.get(where={"username": username})
            ids  = data.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0
