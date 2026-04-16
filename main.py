import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import faiss
from groq import Groq
import numpy as np
import pydantic_settings
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("corebrum")


class Settings(pydantic_settings.BaseSettings):
    groq_api_key: str = ""
    gemini_api_key: str = ""
    port: int = 8001
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8001",
    ]
    audio_sample_rate: int = 16000
    silence_threshold: float = 0.25
    heartbeat_interval: int = 15

    class Config:
        extra = "ignore"


settings = Settings()
GROQ_API_KEY = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")

if GROQ_API_KEY:
    logger.info("[INFO] API Keys loaded successfully")
else:
    logger.warning("[WARN] No GROQ_API_KEY found in environment")

HALLUCINATION_FILTER = frozenset(
    {"thank you", "thanks for watching", "subscribe", "watching", "amara.org", "bye"}
)

HALLUCINATION_EXTENDED = frozenset(
    {
        "thank you.",
        "thank you",
        "thank you...",
        "thank you so much.",
        "thanks for watching.",
        "thanks for watching!",
        "thanks for watching",
        "subscribe.",
        "please subscribe.",
        "subscribe for more.",
        "subscribe to my channel.",
        "watching.",
        "watching...",
        "thanks.",
        "thanks...",
        "you.",
        "amara.org",
        "amara.org.",
        "thank you very much.",
        "bye.",
        "please like",
        "like and subscribe",
    }
)

QUESTION_WORDS = frozenset(
    {
        "who",
        "what",
        "where",
        "when",
        "why",
        "how",
        "can",
        "could",
        "would",
        "should",
        "will",
        "do",
        "does",
        "is",
        "are",
        "tell",
        "explain",
        "describe",
        "give",
    }
)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
INDEX_DIR = KNOWLEDGE_DIR / ".index"
INDEX_FILE = INDEX_DIR / "knowledge.index"
METADATA_FILE = INDEX_DIR / "knowledge_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {
            "ui": [],
            "audio": [],
        }
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        async with self._lock:
            if websocket not in self.active_connections[channel]:
                self.active_connections[channel].append(websocket)
        logger.info(
            f"[WS] {channel} connection added. Total: {len(self.active_connections[channel])}"
        )
        print(
            f"[HUD_SYNC] {channel} connection added. Total: {len(self.active_connections[channel])}"
        )

    async def disconnect(self, websocket: WebSocket, channel: str):
        async with self._lock:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)
        logger.info(f"[WS] {channel} connection removed")

    async def broadcast(self, channel: str, payload: dict):
        async with self._lock:
            connections = list(self.active_connections[channel])

        logger.info(
            f"[HUD_SYNC] Broadcasting to {len(connections)} HUD clients: {payload.get('type', 'unknown')}"
        )
        print(f"[HUD_SYNC] Broadcasting to {len(connections)} HUD clients: {payload}")

        for connection in connections:
            try:
                await connection.send_json(payload)
                logger.info(f"[HUD_SYNC] Successfully sent to {channel}")
            except Exception as e:
                logger.warning(f"[WS] Failed to send to {channel}: {e}")
                await self.disconnect(connection, channel)

    def broadcast_task(self, channel: str, payload: dict):
        """Non-blocking broadcast using asyncio.create_task"""
        asyncio.create_task(self.broadcast(channel, payload))
        logger.info(
            f"[HUD_SYNC] Task created for {channel}: {payload.get('type', 'unknown')}"
        )

    async def broadcast_to_ui(self, payload: dict):
        """Explicit UI broadcast - sends to all /ws/ui clients"""
        async with self._lock:
            ui_connections = list(self.active_connections["ui"])

        logger.info(
            f"[HUD_SYNC] Pushing to {len(ui_connections)} HUD clients: {payload.get('type', 'unknown')}"
        )
        print(f"[HUD_SYNC] Pushing to {len(ui_connections)} HUD clients")

        if len(ui_connections) == 0:
            logger.warning(f"[HUD_SYNC] No HUD clients connected!")
            print(f"[HUD_SYNC] ⚠️ No HUD clients to push to!")

        for ws in ui_connections:
            try:
                await ws.send_json(payload)
                print(f"[HUD_SYNC] ✅ Pushed to UI")
                logger.info(f"[HUD_SYNC] ✅ Successfully pushed to UI client")
            except Exception as e:
                print(f"[HUD_SYNC] ❌ Failed: {e}")
                logger.warning(f"[HUD_SYNC] Failed to push: {e}")


connection_manager = ConnectionManager()
transcript_queue: asyncio.Queue[str] = asyncio.Queue()


class DocumentChunk:
    def __init__(self, source_path: str, title: str, chunk_id: str, text: str):
        self.source_path = source_path
        self.title = title
        self.chunk_id = chunk_id
        self.text = text

    def to_dict(self):
        return {
            "source_path": self.source_path,
            "title": self.title,
            "chunk_id": self.chunk_id,
            "text": self.text,
        }


class KnowledgeBase:
    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: list[DocumentChunk] = []
        self.embedder: Optional[SentenceTransformer] = None

    def build_index(self, chunks: list[DocumentChunk]) -> int:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if not chunks:
            self.index = None
            self.metadata = []
            return 0

        if self.embedder is None:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        embeddings = self.embedder.encode(
            [c.text for c in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        faiss.write_index(self.index, str(INDEX_FILE))
        with open(METADATA_FILE, "w") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2)

        self.metadata = chunks
        return len(chunks)

    def load_index(self) -> bool:
        if not INDEX_FILE.exists() or not METADATA_FILE.exists():
            return False

        self.index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE, "r") as f:
            raw = json.load(f)
            self.metadata = [DocumentChunk(**item) for item in raw]
        return True

    def ensure_index(self) -> bool:
        return self.load_index() or (self.index is not None)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.ensure_index() or self.index is None or not query.strip():
            return []

        if self.embedder is None:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        query_emb = self.embedder.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                chunk = self.metadata[idx]
                results.append(
                    {
                        "score": float(score),
                        "text": chunk.text,
                        "source": chunk.source_path,
                    }
                )
        return results

    def search_context(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k)
        if not results:
            return ""
        return "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)


knowledge_base = KnowledgeBase()


import io
import wave


# Global conversation history for copilot
conversation_history = []
last_transcribe_time = 0
MIN_TRANSCRIBE_INTERVAL = (
    8  # Wait 8 seconds between transcriptions to avoid rate limiting
)


async def groq_transcription_worker():
    logger.info("[GROQ_WORKER] Starting Whisper transcription worker")
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    SAMPLE_RATE = 16000

    # Buffer for accumulating audio (aim for ~5 seconds = 80000 bytes)
    audio_buffer = bytearray()
    CHUNK_DURATION_MS = 5000  # Process every 5 seconds
    BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16-bit mono
    TARGET_BUFFER_SIZE = (CHUNK_DURATION_MS * BYTES_PER_SECOND) // 1000

    while True:
        try:
            audio_data = await transcript_queue.get()
            receive_time = time.time()
            data_len = len(audio_data)
            logger.info(f"[GROQ_WORKER] Got {data_len} bytes from queue")

            # Add to buffer
            audio_buffer.extend(audio_data)

            # Only process if we have enough audio (at least 3 seconds)
            if len(audio_buffer) < (3 * BYTES_PER_SECOND):
                logger.info(
                    f"[GROQ_WORKER] Buffering: {len(audio_buffer)}/{TARGET_BUFFER_SIZE} bytes"
                )
                continue

            logger.info(f"[GROQ_WORKER] Processing buffer: {len(audio_buffer)} bytes")

            if not client:
                logger.warning("[GROQ_WORKER] No Groq API key configured")
                await asyncio.sleep(1)
                continue

            # Convert PCM16 bytes to proper WAV format
            pcm_data = bytes(audio_buffer)
            audio_buffer.clear()  # Clear buffer after processing

            wav_buffer = io.BytesIO()

            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                wav_file.setframerate(SAMPLE_RATE)
                import struct

                samples = struct.unpack(f"{len(pcm_data) // 2}h", pcm_data)
                wav_file.writeframes(struct.pack(f"{len(samples)}h", *samples))

            wav_data = wav_buffer.getvalue()
            logger.info(f"[GROQ_WORKER] Created WAV: {len(wav_data)} bytes")

            response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=("audio.wav", wav_data, "audio/wav"),
                response_format="json",
            )
            text = response.text.strip().lower()
            transcribe_time = time.time()
            logger.info(f"[GROQ_WORKER] Whisper returned: '{text}'")

            if text and not any(f in text for f in HALLUCINATION_EXTENDED):
                timestamp = (
                    time.strftime("%Y-%m-%d %H:%M:%S")
                    + f".{int(time.time() * 1000) % 1000:03d}"
                )
                latency_ms = int((transcribe_time - receive_time) * 1000)
                logger.info(
                    f"[TRANSCRIPT_LATENCY] {latency_ms}ms | [{timestamp}] {text}"
                )

                logger.info(
                    f"[BROADCAST] UI connections: {len(connection_manager.active_connections['ui'])}"
                )

                # Explicit direct broadcast to UI
                transcript_payload = {
                    "type": "transcript",
                    "content": text,
                    "timestamp": timestamp,
                    "latency_ms": latency_ms,
                }

                # Use thread-safe task
                connection_manager.broadcast_task("ui", transcript_payload)

                # Also do direct broadcast to ensure it goes through
                await connection_manager.broadcast_to_ui(transcript_payload)

                logger.info(f"[HUD_RELAY] Transcript broadcast complete")

                # Use Ollama for copilot (local, no rate limits)
                words = text.split()
                conversation_history.append(text)

                recent_text = " ".join(conversation_history[-10:])

                logger.info(f"[GROQ_WORKER] Words: {len(words)}, calling Ollama...")
                try:
                    await process_brain_ollama(text)
                except Exception as e:
                    logger.error(f"[GROQ_WORKER] Ollama failed: {e}")

        except Exception as e:
            logger.error(f"[GROQ_WORKER] Error: {e}")
            await asyncio.sleep(1)


async def process_brain(transcript: str):
    if not GROQ_API_KEY:
        logger.warning("[BRAIN] No Groq API key")
        return

    try:
        client = Groq(api_key=GROQ_API_KEY)
        context = knowledge_base.search_context(transcript, top_k=3)

        system_prompt = """You are an Interview Copilot helping during a job interview. 
Provide helpful, concise suggestions for what the interviewee could say next.
Focus on: key points to mention, relevant examples, and professional phrasing.
Keep responses to 2-3 bullet points max."""

        if context:
            system_prompt += f"\n\nRelevant knowledge base context:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Based on this interview content: '{transcript}'. What are 2-3 concise suggestions for the interviewee?",
            },
        ]

        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True,
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                await connection_manager.broadcast(
                    "ui", {"type": "copilot_partial", "content": content}
                )

        await connection_manager.broadcast(
            "ui", {"type": "copilot_final", "content": full_response}
        )
        logger.info(f"[BRAIN] Response generated: {len(full_response)} chars")

    except Exception as e:
        logger.error(f"[BRAIN] Error: {e}")
        await connection_manager.broadcast(
            "ui", {"type": "copilot_final", "content": "Analyzing..."}
        )


import httpx


async def process_brain_ollama(transcript: str):
    """Use local Ollama for copilot responses - no API rate limits"""
    try:
        ollama_url = "http://localhost:11434/api/generate"

        system_prompt = """You are an Interview Copilot helping during a job interview. 
Provide helpful, concise suggestions for what the interviewee could say next.
Focus on: key points to mention, relevant examples, and professional phrasing.
Keep responses to 2-3 bullet points max. Start directly with the suggestions."""

        payload = {
            "model": "llama3.2:3b",
            "prompt": f"{system_prompt}\n\nBased on this interview content: '{transcript}'. What are 2-3 concise suggestions for the interviewee?",
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(ollama_url, json=payload)

        if response.status_code == 200:
            result = response.json()
            full_response = result.get("response", "No response")

            logger.info(f"[OLLAMA] Response generated: {len(full_response)} chars")

            await connection_manager.broadcast(
                "ui", {"type": "copilot_partial", "content": full_response}
            )
            await connection_manager.broadcast(
                "ui", {"type": "copilot_final", "content": full_response}
            )
        else:
            logger.error(f"[OLLAMA] Error: {response.status_code}")
            await connection_manager.broadcast(
                "ui", {"type": "copilot_final", "content": "Service unavailable"}
            )

    except Exception as e:
        logger.error(f"[OLLAMA] Error: {e}")
        await connection_manager.broadcast(
            "ui", {"type": "copilot_final", "content": "Ollama not available"}
        )


async def heartbeat_worker():
    while True:
        await asyncio.sleep(settings.heartbeat_interval)
        await connection_manager.broadcast(
            "ui", {"type": "heartbeat", "time": time.time()}
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[LIFESPAN] Starting...")
    if GROQ_API_KEY:
        asyncio.create_task(groq_transcription_worker())
    asyncio.create_task(heartbeat_worker())
    if knowledge_base.load_index():
        logger.info(
            f"[RAG] Indexed {len(knowledge_base.metadata)} chunks from knowledge base"
        )
    yield
    logger.info("[LIFESPAN] Shutting down...")


app = FastAPI(title="Corebrum Production API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexContextRequest(BaseModel):
    text: Optional[str] = None
    role: Optional[str] = None
    seniority: Optional[str] = None


class GenerateAnswerRequest(BaseModel):
    transcript: str
    role: Optional[str] = None
    seniority: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok", "groq_configured": bool(GROQ_API_KEY)}


@app.post("/api/index-context")
async def index_context(
    files: Optional[list[UploadFile]] = File(None), request: IndexContextRequest = None
):
    try:
        chunks = []
        knowledge_dir = Path(__file__).parent / "knowledge"
        knowledge_dir.mkdir(exist_ok=True)

        if files:
            for file in files:
                content = await file.read()
                text = (
                    content.decode("utf-8", errors="ignore")
                    if isinstance(content, bytes)
                    else await file.read()
                )
                if text:
                    for i, part in enumerate(
                        [text[i : i + 900] for i in range(0, len(text), 750)]
                    ):
                        chunks.append(
                            DocumentChunk(
                                file.filename,
                                file.filename,
                                f"{file.filename}-{i}",
                                part,
                            )
                        )

        if request and request.text:
            for i, part in enumerate(
                [request.text[j : j + 900] for j in range(0, len(request.text), 750)]
            ):
                chunks.append(DocumentChunk("manual", "manual", f"manual-{i}", part))

        count = knowledge_base.build_index(chunks)
        return {"status": "indexed", "chunks": count}
    except Exception as e:
        logger.error(f"[INDEX] Error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/generate-answer")
async def generate_answer(request: GenerateAnswerRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured"}

    try:
        client = Groq(api_key=GROQ_API_KEY)
        context = knowledge_base.search_context(request.transcript, top_k=3)

        system_prompt = f"You are an Interview Copilot. Role: {request.role or 'general'}. Seniority: {request.seniority or 'mid'}. Give 3 concise bullet points."
        if context:
            system_prompt += f"\n\nContext:\n{context}"

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.transcript},
            ],
            stream=True,
        )

        async def generate():
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return generate()
    except Exception as e:
        logger.error(f"[GENERATE] Error: {e}")
        return {"error": str(e)}


@app.websocket("/ws/ui")
async def websocket_ui(websocket: WebSocket):
    await connection_manager.connect(websocket, "ui")
    await websocket.send_json(
        {"type": "status", "message": "Connected to Corebrum HUD"}
    )
    print("[WS_UI] HUD CONNECTED - Ready to receive transcripts!")
    logger.info("[WS_UI] HUD connected - ready for transcripts")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket, "ui")


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    chunk_counter = 0
    logger.info(f"[WS_AUDIO] WebSocket opening - preparing to accept...")

    await connection_manager.connect(websocket, "audio")
    logger.info(f"[WS_AUDIO] Chrome Extension connected & accepted")

    try:
        while True:
            msg = await websocket.receive()
            logger.info(
                f"[WS_AUDIO] Raw message received"
            )  # Always log that we got something

            if msg.get("type") == "text":
                text_data = msg.get("text", "")
                logger.info(f"[WS_AUDIO] Text message: {text_data[:100]}")
                try:
                    data = json.loads(text_data)
                    if (
                        data.get("type") == "debug"
                        and data.get("message") == "BROWSER_CONNECTED"
                    ):
                        print("\n" + "=" * 60)
                        print(
                            "[PM_DASHBOARD] SUCCESS: Chrome Extension linked to Python Backend"
                        )
                        print("=" * 60 + "\n")
                        logger.info("[PM_DASHBOARD] Handshake confirmed!")
                except json.JSONDecodeError:
                    logger.warning(f"[WS_AUDIO] Non-JSON text: {text_data[:100]}")

            elif msg.get("type") == "websocket.receive" and "bytes" in msg:
                data = msg.get("bytes")
                if data is None:
                    logger.warning(
                        f"[WS_AUDIO] bytes message but no data in msg keys: {msg.keys()}"
                    )
                    continue
                chunk_counter += 1
                data_len = len(data)

                if data_len < 2 or data_len > 1_000_000:
                    logger.warning(f"[WS_AUDIO] Invalid chunk size: {data_len}")
                    continue

                if chunk_counter % 10 == 0:
                    logger.info(
                        f"[WS_AUDIO] Receiving audio: {data_len} bytes (chunk #{chunk_counter})"
                    )
                else:
                    logger.debug(f"[WS_AUDIO] Queuing {data_len} bytes (VAD bypassed)")

                await transcript_queue.put(data)

    except WebSocketDisconnect:
        logger.info("[WS_AUDIO] Chrome Extension disconnected")
        await connection_manager.disconnect(websocket, "audio")


if __name__ == "__main__":
    logger.info(f"[SERVER] Starting on port {settings.port}")
    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")
