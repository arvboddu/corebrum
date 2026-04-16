import asyncio
import sys
import ctypes
import threading
import time
import socket
import subprocess
import logging
import json
from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator, Optional

import ollama
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("hud_out.txt", mode="w")],
)
logger = logging.getLogger("corebrum")

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
    resume_max_chars: int = 5000
    cors_origins: list[str] = ["*"]
    max_context_results: int = 2
    audio_restart_delay_seconds: int = 3
    silence_timeout_seconds: float = 1.5

    model_config = {"env_prefix": "COREBRUM_"}


settings = Settings()

if not settings.ollama_model:
    raise ValueError("COREBRUM_OLLAMA_MODEL is required")


HOST = "0.0.0.0"
PORT = 8001
OLLAMA_MODEL = settings.ollama_model
MAX_CONTEXT_RESULTS = settings.max_context_results
AUDIO_RESTART_DELAY_SECONDS = settings.audio_restart_delay_seconds
ENERGY_THRESHOLD = settings.audio_energy_threshold
SILENCE_TIMEOUT_SECONDS = settings.silence_timeout_seconds

RESUME_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
RESUME_FILENAME_KEYWORDS = {
    "resume",
    "cv",
    "curriculum",
    "pmp",
    "csm",
    "profile",
    "bio",
}


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

app = FastAPI(title="Corebrum Copilot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            if websocket not in self.active_connections:
                self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, payload: dict) -> None:
        if not self.active_connections:
            return

        async with self._lock:
            connections = list(self.active_connections)

        stale_connections = []

        for connection in connections:
            try:
                await connection.send_json(payload)
            except Exception:
                stale_connections.append(connection)

        if stale_connections:
            async with self._lock:
                for connection in stale_connections:
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)


class CopilotOrchestrator:
    def __init__(self) -> None:
        self.connection_manager = ConnectionManager()
        self.knowledge_base = KnowledgeBase()
        self.transcript_queue: asyncio.Queue[TranscriptSegment] = asyncio.Queue()
        self.ui_message_queue: asyncio.Queue[dict] = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.audio_handler: Optional[AudioHandler] = None
        self.audio_stream_task: Optional[asyncio.Task] = None
        self.processing_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.ui_relay_task: Optional[asyncio.Task] = None
        self.audio_status = "idle"
        self.latest_transcript = ""
        self.latest_response = ""
        self.latest_status = {"type": "status", "state": "idle", "message": "Starting"}
        self._cached_context = ""
        self._should_run = threading.Event()
        self._sentence_buffer = ""
        self._buffer_start_time: Optional[float] = None
        self._partial_last_send_time: float = 0.0
        self._resume_content = ""

    async def _ui_relay_worker(self) -> None:
        while True:
            try:
                payload = await self.ui_message_queue.get()
                await self.connection_manager.broadcast(payload)
            except Exception as exc:
                logger.error(f"UI relay failed: {exc}")

    def queue_ui_message(self, payload: dict) -> None:
        if self.loop is not None and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.ui_message_queue.put(payload), self.loop
            )

    async def _load_resume(self) -> None:
        from pathlib import Path

        knowledge_dir = Path(__file__).resolve().parent.parent / "knowledge"
        if not knowledge_dir.exists():
            logger.warning("Knowledge directory not found")
            return

        for file_path in knowledge_dir.iterdir():
            if not file_path.is_file():
                continue
            name_lower = file_path.name.lower()
            if any(kw in name_lower for kw in RESUME_FILENAME_KEYWORDS):
                ext = file_path.suffix.lower()
                if ext in RESUME_EXTENSIONS:
                    try:
                        from engine.document_parser import KnowledgeBase

                        kb = KnowledgeBase()
                        text = kb._read_document(file_path)
                        truncated = text[: settings.resume_max_chars]
                        self._resume_content = truncated
                        logger.info(
                            f"[SUCCESS] Resume loaded: {file_path.name} ({len(truncated)} chars indexed)"
                        )
                        self.queue_ui_message(
                            {
                                "type": "status",
                                "state": "connected",
                                "message": f"Resume loaded: {file_path.name}",
                            }
                        )
                    except Exception as exc:
                        logger.error(f"[RESUME] Load failed: {exc}")
                    break

    async def _safe_build_knowledge_index(self) -> None:
        try:
            chunk_count = await asyncio.to_thread(self.knowledge_base.build_index)
            file_count = len(
                set(chunk.source_path for chunk in self.knowledge_base.metadata)
            )

            self.queue_ui_message(
                {
                    "type": "status",
                    "state": "connected",
                    "message": f"Knowledge base ready ({chunk_count} chunks)",
                }
            )
            logger.info(f"[KNOWLEDGE] Indexed with {chunk_count} chunk(s).")
            logger.info(f"[BRAIN] Successfully indexed {file_count} documents.")
        except Exception as exc:
            logger.error(f"[KNOWLEDGE] Indexing failed: {exc}")

    async def _run_audio_stream(self) -> None:
        self.audio_status = "starting"
        logger.info("[AUDIO] Creating AudioHandler in background thread...")

        def audio_thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:

                def audio_signal_cb(level: float):
                    loop = getattr(self, "loop", None)
                    if loop and not loop.is_closed():
                        self.queue_ui_message({"type": "signal", "level": level})

                audio_handler = AudioHandler(signal_callback=audio_signal_cb)
                self.audio_handler = audio_handler

                async def process_segments():
                    self.queue_ui_message(
                        {"type": "status", "state": "connected", "message": "Listening"}
                    )

                    async for segment in audio_handler.stream_transcription():
                        if not segment or not segment.text.strip():
                            continue

                        segment_text = segment.text.strip()
                        cleaned_text = segment_text.lower()

                        if any(
                            f in cleaned_text
                            for f in (
                                "thank you",
                                "thanks for watching",
                                "subscribe",
                                "watching",
                                "amara.org",
                            )
                        ):
                            logger.debug(
                                f"[CLEANUP] Filtered hallucination: '{segment_text}'"
                            )
                            continue

                        logger.debug(f"[HUD_RELAY] Sending text to HUD: {segment_text}")
                        self.queue_ui_message(
                            {"type": "heard_partial", "text": segment_text}
                        )

                        if not self._buffer_start_time:
                            self._buffer_start_time = time.monotonic()

                        self._sentence_buffer += segment_text + " "

                        is_sentence_end = any(p in segment.text for p in ".!?") or (
                            time.monotonic() - self._buffer_start_time
                            > SILENCE_TIMEOUT_SECONDS
                        )

                        if is_sentence_end and self._sentence_buffer.strip():
                            final_text = self._sentence_buffer.strip()
                            self.queue_ui_message(
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

                loop.run_until_complete(process_segments())
            except Exception as exc:
                logger.error(f"[AUDIO] Thread error: {exc}")
            finally:
                loop.close()

        thread = threading.Thread(target=audio_thread_target, daemon=True)
        thread.start()

        self.audio_status = "connected"

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
                logger.error(f"Transcript orchestration failed: {exc}")

    async def _handle_transcript(self, segment: TranscriptSegment) -> None:
        logger.info(f"[TRANSCRIPT] {segment.text}")
        self.latest_transcript = segment.text
        self.latest_response = ""
        self.queue_ui_message({"type": "transcript", "content": segment.text})
        self.queue_ui_message({"type": "clear", "content": ""})

        word_count = len(segment.text.split())
        has_punctuation = any(c in segment.text for c in ".!?")

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
        logger.info("[BRAIN] Processing...")
        self.queue_ui_message(
            {"type": "status", "state": "connected", "message": "Generating..."}
        )

        INTERVIEW_SYSTEM_PROMPT = "You are an Interview Copilot. Give 3 concise bullet points for the current question."

        messages = [
            {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
        ]

        context_parts = []
        if self._resume_content:
            context_parts.append(f"RESUME:\n{self._resume_content}")

        knowledge_context = self.knowledge_base.search_context(
            transcript, top_k=MAX_CONTEXT_RESULTS
        )
        if knowledge_context:
            context_parts.append(f"KNOWLEDGE BASE:\n{knowledge_context}")

        if context_parts:
            messages.append({"role": "system", "content": "\n\n".join(context_parts)})

        messages.append({"role": "user", "content": transcript})

        try:
            stream = await asyncio.to_thread(
                ollama.chat,
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
            )
        except Exception as exc:
            logger.error(f"Ollama request failed: {exc}")
            self.queue_ui_message({"type": "advice", "content": "Ollama unavailable"})
            return

        generated_any_output = False
        try:
            async for content in self._stream_ollama_content(stream):
                generated_any_output = True
                self.latest_response += content
                self.queue_ui_message({"type": "copilot_partial", "content": content})
        except Exception as exc:
            logger.error(f"Ollama streaming failed: {exc}")

        if generated_any_output:
            logger.info(f"[OLLAMA] Response: {len(self.latest_response)} chars")

        self.queue_ui_message(
            {"type": "status", "state": "connected", "message": "Listening"}
        )

    async def _stream_ollama_content(self, stream) -> AsyncIterator[str]:
        if self.loop is None:
            return

        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        sentinel: str = "__END__"

        def worker() -> None:
            if self.loop is None:
                return
            try:
                for chunk in stream:
                    message = getattr(chunk, "message", None)
                    content: str = getattr(message, "content", "") if message else ""
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
            self.queue_ui_message({"type": "status", "content": "Listening..."})

    async def shutdown(self) -> None:
        self._should_run.clear()
        if self.audio_handler is not None:
            self.audio_handler.stop()

        for task in [
            self.audio_stream_task,
            self.processing_task,
            self.heartbeat_task,
            self.ui_relay_task,
        ]:
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


orchestrator = CopilotOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("[LIFESPAN] Starting...")

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

    logger.info("[LIFESPAN] Shutting down...")
    await orchestrator.shutdown()


async def start_background_tasks() -> None:
    await asyncio.sleep(2)
    logger.info("[BACKGND] Building knowledge index...")
    await orchestrator._safe_build_knowledge_index()
    logger.info("[BACKGND] Loading resume...")
    await orchestrator._load_resume()
    logger.info("[BACKGND] Starting audio stream...")
    orchestrator.processing_task = asyncio.create_task(
        orchestrator._process_transcripts()
    )
    orchestrator.heartbeat_task = asyncio.create_task(orchestrator._send_heartbeat())
    orchestrator.ui_relay_task = asyncio.create_task(orchestrator._ui_relay_worker())
    await asyncio.sleep(1)
    orchestrator.audio_stream_task = asyncio.create_task(
        orchestrator._run_audio_stream()
    )
    logger.info("[BACKGND] Ready")


app.router.lifespan_context = lifespan


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "audio": orchestrator.audio_status, "model": OLLAMA_MODEL}


@app.websocket("/")
async def websocket_root(websocket: WebSocket) -> None:
    await websocket_endpoint(websocket)


@app.websocket("/ws/ui")
async def websocket_endpoint(websocket: WebSocket) -> None:
    logger.info("[WS] New WebSocket connection")
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
        logger.info("HUD disconnected")
        await orchestrator.connection_manager.disconnect(websocket)


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket) -> None:
    await orchestrator.connection_manager.connect(websocket)
    logger.info("[WS] Chrome Extension connected to Audio ingestion stream")
    chunk_count = 0
    MAX_CHUNK_SIZE = 1_000_000  # 1MB max per chunk

    try:
        while True:
            try:
                message = await websocket.receive()
            except Exception as exc:
                logger.error(f"[WS_AUDIO] Handshake Error: {exc}")
                continue

            if message["type"] == "text":
                text_data = message["text"]
                try:
                    json_msg = json.loads(text_data)
                    if (
                        json_msg.get("type") == "debug"
                        and json_msg.get("message") == "BROWSER_CONNECTED"
                    ):
                        print("\n" + "=" * 60)
                        print(
                            "[PM_DASHBOARD] SUCCESS: Chrome Extension linked to Python Backend"
                        )
                        print("=" * 60 + "\n")
                        logger.info(
                            "[WS_AUDIO] Debug signal received - Browser connected"
                        )
                    else:
                        logger.debug(f"[WS_AUDIO] Unknown JSON message: {json_msg}")
                except json.JSONDecodeError:
                    logger.debug(
                        f"[WS_AUDIO] Non-JSON text received: {text_data[:100]}"
                    )

            elif message["type"] == "bytes":
                data = message["bytes"]
                logger.debug(f"[WS_AUDIO] Received {len(data)} bytes")

                if not data or len(data) < 2:
                    logger.warning(f"[WS_AUDIO] Skipping empty/too-small chunk")
                    continue

                if len(data) > MAX_CHUNK_SIZE:
                    logger.warning(
                        f"[WS_AUDIO] Chunk too large ({len(data)} bytes), skipping"
                    )
                    continue

                chunk_count += 1
                logger.debug(
                    f"[WS_AUDIO_DEBUG] Received chunk {chunk_count}: {len(data)} bytes"
                )
                if orchestrator.audio_handler is not None:
                    orchestrator.audio_handler.push_chunk(data)
                else:
                    logger.error(
                        "[WS_AUDIO_DEBUG] Audio Handler is None - Whisper not ready"
                    )

    except WebSocketDisconnect:
        logger.info("[WS] Chrome Extension disconnected - Audio processing paused")
        if orchestrator.audio_handler is not None:
            orchestrator.audio_handler.reset_buffer()
    except Exception as exc:
        logger.error(f"[WS_AUDIO] Handshake Error: {exc}")


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("0.0.0.0", port)) == 0


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
                    logger.info(f"[PORT] Killed {pid} on port {port}")
                except Exception:
                    pass


def main() -> None:
    if is_port_in_use(PORT):
        logger.info(f"[PORT] Port {PORT} in use, killing...")
        kill_port_process(PORT)
        time.sleep(1)

    logger.info(f"[SERVER] http://{HOST}:{PORT}")
    logger.info("[SERVER] WebSocket endpoint /ws/audio is ready.")
    print("\n" + "=" * 60)
    print("[DASHBOARD] Listening for Audio on ws://localhost:8001/ws/audio")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
