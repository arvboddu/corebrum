import asyncio
import sys
import ctypes
import threading
import time
import socket
import subprocess
from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator, Optional

import ollama
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings

try:
    from engine.audio_handler import AudioHandler, TranscriptSegment
    from engine.document_parser import KnowledgeBase
except ImportError:
    from audio_handler import AudioHandler, TranscriptSegment
    from document_parser import KnowledgeBase


class Settings(BaseSettings):
    ollama_model: str = "llama3.2:3b"
    ollama_api_base: str = "http://localhost:11434"
    audio_energy_threshold: int = 300

    model_config = {"env_prefix": "COREBRUM_"}


settings = Settings()

if not settings.ollama_model:
    raise ValueError("COREBRUM_OLLAMA_MODEL is required")


HOST = "127.0.0.1"
PORT = 8001
OLLAMA_MODEL = settings.ollama_model
MAX_CONTEXT_RESULTS = 2
AUDIO_RESTART_DELAY_SECONDS = 3
ENERGY_THRESHOLD = settings.audio_energy_threshold
SILENCE_TIMEOUT_SECONDS = 1.5


app = FastAPI(title="Corebrum Copilot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast_json(self, payload: dict) -> None:
        if self._connections:
            asyncio.create_task(self._send_to_all(payload))

    async def _send_to_all(self, payload: dict) -> None:
        async with self._lock:
            connections = list(self._connections)

        stale_connections = []
        for connection in connections:
            try:
                await connection.send_json(payload)
            except Exception:
                stale_connections.append(connection)

        if stale_connections:
            async with self._lock:
                for connection in stale_connections:
                    self._connections.discard(connection)


class CopilotOrchestrator:
    def __init__(self) -> None:
        self.connection_manager = ConnectionManager()
        self.knowledge_base = KnowledgeBase()
        self.transcript_queue: asyncio.Queue[TranscriptSegment] = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.audio_handler: Optional[AudioHandler] = None
        self.audio_stream_task: Optional[asyncio.Task] = None
        self.processing_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.audio_status = "idle"
        self.latest_transcript = ""
        self.latest_response = ""
        self.latest_status = {"type": "status", "state": "idle", "message": "Starting"}
        self._cached_context = ""
        self._should_run = threading.Event()
        self._sentence_buffer = ""
        self._buffer_start_time: Optional[float] = None

    async def _safe_build_knowledge_index(self) -> None:
        try:
            chunk_count = await asyncio.to_thread(self.knowledge_base.build_index)
            await self.connection_manager.broadcast_json(
                {
                    "type": "status",
                    "state": "connected",
                    "message": f"Knowledge base ready ({chunk_count} chunks)",
                }
            )
            print(f"[KNOWLEDGE] Indexed with {chunk_count} chunk(s).")
        except Exception as exc:
            print(f"[KNOWLEDGE] Indexing failed: {exc}")

    async def _run_audio_stream(self) -> None:
        while self._should_run.is_set():
            try:
                self.audio_status = "starting"
                print("[AUDIO] Creating AudioHandler...")
                await self.connection_manager.broadcast_json(
                    {
                        "type": "status",
                        "state": "idle",
                        "message": "Starting audio engine",
                    }
                )

                self.audio_handler = AudioHandler()
                self.audio_status = "connected"
                await self.connection_manager.broadcast_json(
                    {"type": "status", "state": "connected", "message": "Listening"}
                )

                print("[AUDIO] Starting stream_transcription...")
                async for segment in self.audio_handler.stream_transcription():
                    if not segment or not segment.text.strip():
                        continue

                    await self.connection_manager.broadcast_json(
                        {"type": "heard_partial", "text": segment.text.strip()}
                    )

                    if not self._buffer_start_time:
                        self._buffer_start_time = time.monotonic()

                    self._sentence_buffer += segment.text.strip() + " "

                    is_sentence_end = any(p in segment.text for p in ".!?") or (
                        time.monotonic() - self._buffer_start_time
                        > SILENCE_TIMEOUT_SECONDS
                    )

                    if is_sentence_end and self._sentence_buffer.strip():
                        final_text = self._sentence_buffer.strip()
                        await self.connection_manager.broadcast_json(
                            {"type": "heard_final", "text": final_text}
                        )
                        self.enqueue_transcript(
                            TranscriptSegment(
                                text=final_text,
                                started_at=segment.started_at,
                                ended_at=segment.ended_at,
                            )
                        )
                        self._sentence_buffer = ""
                        self._buffer_start_time = None

            except Exception as exc:
                self.audio_status = "error"
                print(f"[AUDIO] Stream error: {exc}")
                await self.connection_manager.broadcast_json(
                    {
                        "type": "status",
                        "state": "disconnected",
                        "message": "Audio stream error, retrying",
                    }
                )
                if not self._should_run.is_set():
                    break
                await asyncio.sleep(AUDIO_RESTART_DELAY_SECONDS)
            finally:
                if self.audio_handler:
                    self.audio_handler.stop()
                    self.audio_handler = None

    def enqueue_transcript(self, segment: TranscriptSegment) -> None:
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.transcript_queue.put(segment), self.loop)

    async def _process_transcripts(self) -> None:
        while True:
            segment = await self.transcript_queue.get()
            try:
                await self._handle_transcript(segment)
            except Exception as exc:
                print(f"Transcript orchestration failed: {exc}")

    async def _handle_transcript(self, segment: TranscriptSegment) -> None:
        print(f"[TRANSCRIPT] {segment.text}")
        self.latest_transcript = segment.text
        self.latest_response = ""
        await self.connection_manager.broadcast_json(
            {"type": "transcript", "content": segment.text}
        )
        await self.connection_manager.broadcast_json({"type": "clear", "content": ""})

        word_count = len(segment.text.split())
        has_punctuation = any(c in segment.text for c in ".!?")

        QUESTION_WORDS = {
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
        transcript_lower = segment.text.lower()
        has_question_word = any(qw in transcript_lower.split() for qw in QUESTION_WORDS)
        has_question_mark = "?" in segment.text

        words = segment.text.split()
        repeated_count = sum(
            1 for i in range(len(words) - 1) if words[i].lower() == words[i + 1].lower()
        )

        if repeated_count > 3:
            return

        if not has_question_word and not has_question_mark:
            return

        if word_count < 5 and not has_punctuation:
            return

        asyncio.create_task(self._process_brain(segment.text))

    async def _process_brain(self, transcript: str) -> None:
        print("[BRAIN] Processing...")
        await self.connection_manager.broadcast_json(
            {"type": "status", "state": "connected", "message": "Generating..."}
        )

        try:
            stream = await asyncio.to_thread(
                ollama.chat,
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are an AI interview assistant."},
                    {"role": "user", "content": transcript},
                ],
                stream=True,
            )
        except Exception as exc:
            print(f"Ollama request failed: {exc}")
            await self.connection_manager.broadcast_json(
                {"type": "advice", "content": "Ollama unavailable"}
            )
            return

        generated_any_output = False
        try:
            async for content in self._stream_ollama_content(stream):
                generated_any_output = True
                self.latest_response += content
                await self.connection_manager.broadcast_json(
                    {"type": "copilot_partial", "content": content}
                )
        except Exception as exc:
            print(f"Ollama streaming failed: {exc}")

        if generated_any_output:
            print(f"[OLLAMA] Response: {len(self.latest_response)} chars")

        await self.connection_manager.broadcast_json(
            {"type": "status", "state": "connected", "message": "Listening"}
        )

    async def _stream_ollama_content(self, stream) -> AsyncIterator[str]:
        if self.loop is None:
            return

        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        sentinel: object = object()

        def worker() -> None:
            if self.loop is None:
                return
            try:
                for chunk in stream:
                    message = getattr(chunk, "message", None)
                    content = getattr(message, "content", "") if message else ""
                    if content:
                        self.loop.call_soon_threadsafe(queue.put_nowait, content)
            except Exception as exc:
                self.loop.call_soon_threadsafe(queue.put_nowait, str(exc))
            finally:
                self.loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            if item:
                yield item

    async def _send_heartbeat(self) -> None:
        while True:
            await asyncio.sleep(5)
            await self.connection_manager.broadcast_json(
                {"type": "status", "content": "Listening..."}
            )

    async def shutdown(self) -> None:
        self._should_run.clear()
        if self.audio_handler is not None:
            self.audio_handler.stop()

        for task in [self.audio_stream_task, self.processing_task, self.heartbeat_task]:
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


orchestrator = CopilotOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    print("[LIFESPAN] Starting...")

    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00008000
            )
        except Exception:
            pass

    orchestrator._should_run.set()
    orchestrator.loop = asyncio.get_running_loop()

    asyncio.create_task(start_background_tasks())

    yield

    print("[LIFESPAN] Shutting down...")
    await orchestrator.shutdown()


async def start_background_tasks() -> None:
    await asyncio.sleep(1)
    print("[BACKGND] Building knowledge index...")
    await orchestrator._safe_build_knowledge_index()
    print("[BACKGND] Starting audio stream...")
    orchestrator.processing_task = asyncio.create_task(
        orchestrator._process_transcripts()
    )
    orchestrator.heartbeat_task = asyncio.create_task(orchestrator._send_heartbeat())
    orchestrator.audio_stream_task = asyncio.create_task(
        orchestrator._run_audio_stream()
    )
    print("[BACKGND] Ready")


app.router.lifespan_context = lifespan


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "audio": orchestrator.audio_status, "model": OLLAMA_MODEL}


@app.websocket("/")
async def websocket_root(websocket: WebSocket) -> None:
    await websocket_endpoint(websocket)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    print("[WS] New WebSocket connection")
    await orchestrator.connection_manager.connect(websocket)
    await websocket.send_json(
        {"type": "transcript", "content": "SYSTEM CHECK: Audio Link Active"}
    )

    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                continue
    except WebSocketDisconnect:
        print("HUD disconnected")
        await orchestrator.connection_manager.disconnect(websocket)


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_port_process(port: int) -> None:
    result = subprocess.run(
        f"netstat -ano | findstr :{port} | findstr LISTENING",
        shell=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().split("\n"):
        if line:
            parts = line.split()
            if len(parts) >= 5:
                pid = parts[-1]
                try:
                    subprocess.run(
                        f"taskkill /F /PID {pid}", shell=True, capture_output=True
                    )
                    print(f"[PORT] Killed {pid} on port {port}")
                except Exception:
                    pass


def main() -> None:
    if is_port_in_use(PORT):
        print(f"[PORT] Port {PORT} in use, killing...")
        kill_port_process(PORT)
        time.sleep(1)

    print(f"[SERVER] http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
