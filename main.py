import asyncio
import json
import logging
import math
import os
import struct
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

load_dotenv()

import faiss
from faster_whisper import WhisperModel
from groq import Groq
import numpy as np
import pydantic_settings
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    cors_origins: list[str] = ["*"]
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


def clean_transcript(text: str) -> str:
    """Remove filler words and clean up transcript."""
    words = text.split()
    cleaned = [w for w in words if w.lower() not in FILLER_WORDS]
    text = " ".join(cleaned)
    text = " ".join(text.split())
    return text


def levenshtein_distance(a: str, b: str) -> float:
    """Calculate Levenshtein similarity (0.0 to 1.0). Returns 1.0 for identical strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    a = a.lower().strip()
    b = b.lower().strip()

    if a == b:
        return 1.0

    len_a, len_b = len(a), len(b)
    if abs(len_a - len_b) > max(len_a, len_b) * 0.5:
        return 0.0

    prev_row = list(range(len_b + 1))
    curr_row = [0] * (len_b + 1)

    for i in range(1, len_a + 1):
        curr_row[0] = i
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1, prev_row[j] + 1, prev_row[j - 1] + cost
            )
        prev_row, curr_row = curr_row, prev_row

    distance = prev_row[len_b]
    max_len = max(len_a, len_b)
    return 1.0 - (distance / max_len) if max_len > 0 else 1.0


# Transcript debouncer state
transcript_history: List[str] = []
MAX_HISTORY = 3


def is_duplicate_transcript(text: str) -> bool:
    """Check if text is >85% similar to any of last 3 messages."""
    if not text or not transcript_history:
        return False

    for prev in transcript_history[-MAX_HISTORY:]:
        if levenshtein_distance(text, prev) > 0.85:
            return True
    return False


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

FILLER_WORDS = frozenset(
    {
        "um",
        "uh",
        "mm",
        "hm",
        "hmm",
        "er",
        "ah",
        "like",
        "you know",
        "basically",
        "actually",
        "literally",
        "so yeah",
        "i mean",
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
        self.resume_summary: str = ""
        self.has_resume_summary: bool = False

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

        k = top_k
        distances, indices = self.index.search(query_emb, k)
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


def _run_whisper(model, audio_data):
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_data)
        temp_path = f.name
    try:
        segments, info = model.transcribe(temp_path, language="en")
        result = ([s.text.strip() for s in segments], info)
    finally:
        os.unlink(temp_path)
    return result


async def _stream_groq_chunks(stream):
    queue: asyncio.Queue[object] = asyncio.Queue()
    sentinel = object()

    def worker():
        try:
            for chunk in stream:
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

    loop = asyncio.get_running_loop()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def groq_transcription_worker():
    logger.info("[WHISPER_WORKER] Loading faster-whisper model...")
    SAMPLE_RATE = 16000

    # Try different model sizes in order of preference
    WHISPER_MODELS = ["base", "small", "medium", "tiny"]
    whisper_model = None

    for model_size in WHISPER_MODELS:
        try:
            whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info(f"[WHISPER_WORKER] Loaded faster-whisper-{model_size}")
            break
        except Exception as e:
            logger.warning(f"[WHISPER_WORKER] Failed to load {model_size}: {e}")
            continue

    if not whisper_model:
        logger.error("[WHISPER_WORKER] No faster-whisper model available")
        return

    if GROQ_API_KEY:
        logger.info("[WHISPER_WORKER] Groq client available for LLM")

    # Buffer for accumulating audio
    audio_buffer = bytearray()
    current_utterance_id = str(uuid.uuid4())
    CHUNK_DURATION_MS = 5000
    BYTES_PER_SECOND = SAMPLE_RATE * 2
    SILENCE_FLUSH_SECONDS = 1
    MIN_AUDIO_SECONDS = SILENCE_FLUSH_SECONDS
    MIN_AUDIO_SIZE = MIN_AUDIO_SECONDS * BYTES_PER_SECOND
    TARGET_BUFFER_SIZE = (CHUNK_DURATION_MS * BYTES_PER_SECOND) // 1000
    MAX_QUEUE_SIZE = 10
    last_sent_text = ""
    current_transcript_buffer = ""

    # Deduplication
    last_transcript = ""
    last_transcript_time = 0.0
    REPEAT_THRESHOLD = 0.9  # 90% similarity
    HALLUCINATION_SILENCE = frozenset(
        {"thank you very much.", "of anybody.", "thank you.", "thanks."}
    )
    silence_repeat_count = 0

    def is_similar(a: str, b: str) -> bool:
        """Check if two strings are 90% similar."""
        if not a or not b:
            return False
        a = a.lower().strip()
        b = b.lower().strip()
        if a == b:
            return True
        # Simple word overlap check
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
        return overlap >= REPEAT_THRESHOLD

    def detect_voice_activity(audio_data: bytes) -> tuple[bool, float]:
        """Simple VAD: check if audio has enough energy above silence threshold."""
        if len(audio_data) < 2:
            return False, 0.0
        try:
            samples = struct.unpack(f"{len(audio_data) // 2}h", audio_data)
            # Calculate RMS
            rms = sum(s * s for s in samples) / len(samples)
            rms = math.sqrt(rms)
            # Threshold for voice activity (adjust as needed)
            return rms > 500, rms  # ~3% of max (32767)
        except:
            return True, 0.0  # Default to processing if VAD fails

    while True:
        try:
            # Drop oldest chunk if queue exceeds limit
            queue_size = transcript_queue.qsize()
            if queue_size > MAX_QUEUE_SIZE:
                try:
                    await transcript_queue.get()
                    logger.warning(
                        f"[WHISPER_WORKER] Dropped oldest, queue was {queue_size}"
                    )
                except asyncio.QueueEmpty:
                    pass

            audio_data = await transcript_queue.get()
            receive_time = time.time()
            data_len = len(audio_data)
            logger.info(f"[WHISPER_WORKER] Got {data_len} bytes (queue: {queue_size})")

            # VAD: skip chunks with no voice activity
            has_speech, speech_level = detect_voice_activity(audio_data)
            if not has_speech:
                logger.info("[WHISPER_WORKER] No voice activity, skipping chunk")
                continue
            logger.info(f"[AUDIO] Speech detected (Level: {speech_level:.1f})")

            audio_buffer.extend(audio_data)

            # Only process if we have enough audio (at least 1 second)
            if len(audio_buffer) < MIN_AUDIO_SIZE:
                logger.info(
                    f"[WHISPER_WORKER] Buffering: {len(audio_buffer)}/{MIN_AUDIO_SIZE}"
                )
                continue

            logger.info(f"[WHISPER_WORKER] Processing: {len(audio_buffer)} bytes")

            pcm_data = bytes(audio_buffer)
            audio_buffer.clear()

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                samples = struct.unpack(f"{len(pcm_data) // 2}h", pcm_data)
                wav_file.writeframes(struct.pack(f"{len(samples)}h", *samples))

            wav_data = wav_buffer.getvalue()
            logger.info(f"[WHISPER_WORKER] WAV: {len(wav_data)} bytes")

            segments, info = await asyncio.to_thread(
                _run_whisper, whisper_model, wav_data
            )
            text = " ".join(segments).strip().lower()
            text = clean_transcript(text)

            if not text:
                logger.info("[WHISPER_WORKER] No speech detected")
                continue

            interim_payload = {
                "type": "transcript",
                "utterance_id": current_utterance_id,
                "who": "YOU",
                "text": text,
                "is_final": False,
            }
            connection_manager.broadcast_task("ui", interim_payload)

            # 1. Last Message Check: skip if identical within 1 second
            current_time = time.time()
            if text == last_transcript and (current_time - last_transcript_time) < 1.0:
                logger.info(f"[WHISPER_WORKER] Skipping duplicate < 1s: '{text}'")
                continue

            # 2. Silence Suppression: ignore repeating hallucinations
            silence_phrase = text.strip().lower()
            if silence_phrase in HALLUCINATION_SILENCE:
                silence_repeat_count += 1
                if silence_repeat_count > 2:
                    logger.info(
                        f"[WHISPER_WORKER] Skipping silence hallucination: '{text}'"
                    )
                    continue
            else:
                silence_repeat_count = 0

            # 3. Regular deduplication
            if is_similar(text, last_transcript):
                logger.info(f"[WHISPER_WORKER] Skipping similar: '{text}'")
                continue

            last_transcript = text
            last_transcript_time = current_time

            transcribe_time = time.time()
            logger.info(f"[WHISPER_WORKER] Transcript: '{text}'")

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

                inferred_speaker = (
                    "INTERVIEWER"
                    if any(qw in text.lower().split() for qw in QUESTION_WORDS)
                    or "?" in text
                    else "YOU"
                )

                transcript_payload = {
                    "type": "transcript",
                    "utterance_id": current_utterance_id,
                    "who": inferred_speaker,
                    "text": text,
                    "is_final": True,
                }

                # 4. Last Sent Text Check: skip exactly same as previous broadcast
                if text == last_sent_text:
                    logger.info(
                        f"[DEBOUNCER] Skipping identical to last sent: '{text}'"
                    )
                elif is_duplicate_transcript(text):
                    logger.info(f"[DEBOUNCER] Skipping >85% similar: '{text[:50]}...'")
                else:
                    # Add to history (keep last 3)
                    transcript_history.append(text)
                    if len(transcript_history) > MAX_HISTORY:
                        transcript_history.pop(0)

                    # 5. Buffer Reset: clear after final
                    current_transcript_buffer = ""
                    last_sent_text = text

                    connection_manager.broadcast_task("ui", transcript_payload)
                    await connection_manager.broadcast_to_ui(transcript_payload)
                    logger.info(f"[HUD_RELAY] Transcript broadcast complete")

                    # Generate new utterance_id for next speech
                    current_utterance_id = str(uuid.uuid4())

                # 6. Non-Blocking LLM Call with try/except
                words = text.split()
                conversation_history.append(text)

                logger.info(f"[GROQ_WORKER] Words: {len(words)}, calling Ollama...")
                try:
                    await process_brain_ollama(text)
                except Exception as e:
                    logger.error(f"[GROQ_WORKER] Ollama failed: {e}")
                    await connection_manager.broadcast(
                        "ui", {"type": "copilot_final", "text": "LLM Busy..."}
                    )

        except Exception as e:
            logger.error(f"[GROQ_WORKER] Error: {e}")
            await asyncio.sleep(1)


async def process_brain(transcript: str):
    if not GROQ_API_KEY:
        logger.warning("[BRAIN] No Groq API key")
        return

    try:
        client = Groq(api_key=GROQ_API_KEY)
        context = knowledge_base.search_context(transcript, top_k=5)

        persona = "You are an Interview Copilot helping during a job interview. Be practical and actionable."
        star_method = "For behavioral questions (like 'tell me about a time...'), use STAR: Situation → Task → Action → Result with metrics."

        system_prompt = f"""{persona}
IMPORTANT: The transcript is from a live interview; ignore minor phonetic errors and prioritize professional context. Aggressively infer the intended meaning. Ignore filler words like "um", "uh", "like", "basically".
{star_method}
Use specific examples from the provided context that match what the interviewer is asking about.
Provide 2-3 concise bullet points starting with action verbs."""

        if context:
            system_prompt += f"\n\nResume/JD Context:\n{context}"

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
        async for chunk in _stream_groq_chunks(stream):
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                await connection_manager.broadcast(
                    "ui", {"type": "copilot_partial", "text": content}
                )

        await connection_manager.broadcast(
            "ui", {"type": "copilot_final", "text": full_response}
        )
        logger.info(f"[BRAIN] Response generated: {len(full_response)} chars")

    except Exception as e:
        logger.error(f"[BRAIN] Error: {e}")
        await connection_manager.broadcast(
            "ui", {"type": "copilot_final", "text": "Analyzing..."}
        )


import httpx


async def process_brain_ollama(transcript: str):
    """Fault 8: Parallel RAG + LLM with 400ms race window"""
    try:
        ollama_url = "http://localhost:11434/api/generate"
        RAG_TIMEOUT = 0.4  # 400ms max wait for RAG

        def sync_rag_search():
            """Synchronous RAG search - runs in thread pool"""
            raw_results = knowledge_base.search(transcript, top_k=5)
            has_high_score = any(r.get("score", 0) > 0.5 for r in raw_results)
            if has_high_score:
                return knowledge_base.search_context(transcript, top_k=5)
            elif (
                hasattr(knowledge_base, "resume_summary")
                and knowledge_base.resume_summary
            ):
                return knowledge_base.resume_summary
            return ""

        # Fire RAG in background thread (Fault 8: Parallel RAG)
        rag_task = asyncio.create_task(asyncio.to_thread(sync_rag_search))

        persona = "You are an Interview Copilot helping during a job interview. Be practical and actionable."
        star_method = "For behavioral questions (like 'tell me about a time...'), use STAR: Situation → Task → Action → Result with metrics."

        system_prompt = f"""{persona}
IMPORTANT: The transcript is from a live interview; ignore minor phonetic errors and prioritize professional context. Aggressively infer the intended meaning. Ignore filler words like "um", "uh", "like", "basically".
{star_method}
Use specific examples from the provided context that match what the interviewer is asking about.
Provide 2-3 concise bullet points starting with action verbs."""

        # Wait up to 400ms for RAG
        try:
            context = await asyncio.wait_for(rag_task, timeout=RAG_TIMEOUT)
        except asyncio.TimeoutError:
            context = ""  # RAG not ready
            logger.info("[RAG] Timeout - proceeding without context")
        except Exception as e:
            logger.warning(f"[RAG] Error: {e}")
            context = ""

        if context:
            system_prompt += f"\n\nResume/JD Context:\n{context}"

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
                "ui", {"type": "copilot_partial", "text": full_response}
            )
            await connection_manager.broadcast(
                "ui", {"type": "copilot_final", "text": full_response}
            )
        else:
            logger.error(f"[OLLAMA] Error: {response.status_code}")
            await connection_manager.broadcast(
                "ui", {"type": "copilot_final", "text": "Service unavailable"}
            )

    except Exception as e:
        logger.error(f"[OLLAMA] Error: {e}")
        await connection_manager.broadcast(
            "ui", {"type": "copilot_final", "text": f"Ollama error: {str(e)}"}
        )


async def heartbeat_worker():
    while True:
        await asyncio.sleep(settings.heartbeat_interval)
        await connection_manager.broadcast(
            "ui", {"type": "heartbeat", "text": str(time.time())}
        )


async def buffer_flush_worker():
    """Flush audio queue every 30 seconds to prevent memory leaks."""
    while True:
        await asyncio.sleep(30)
        try:
            queue_size = transcript_queue.qsize()
            if queue_size > 0:
                logger.info(f"[BUFFER_FLUSH] Clearing {queue_size} chunks from queue")
                while not transcript_queue.empty():
                    try:
                        transcript_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
        except Exception as e:
            logger.warning(f"[BUFFER_FLUSH] Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[LIFESPAN] Starting...")
    asyncio.create_task(groq_transcription_worker())
    asyncio.create_task(heartbeat_worker())
    asyncio.create_task(buffer_flush_worker())
    if knowledge_base.load_index():
        logger.info(
            f"[RAG] Indexed {len(knowledge_base.metadata)} chunks from knowledge base"
        )
        # Build resume summary for cold start fallback
        if knowledge_base.metadata:
            summary_parts = [c.text[:200] for c in knowledge_base.metadata[:3]]
            knowledge_base.resume_summary = "\n\n".join(summary_parts)
            knowledge_base.has_resume_summary = bool(knowledge_base.resume_summary)
            logger.info(
                f"[RAG] Cold start summary prepared ({len(knowledge_base.resume_summary)} chars)"
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


def extract_keywords(text: str, min_count: int = 2) -> list[str]:
    """Extract key technical terms/keywords from context for grounding."""
    import re

    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    common = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "has",
        "are",
        "was",
        "were",
        "been",
        "being",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "work",
        "job",
        "year",
        "years",
        "experience",
        "team",
        "lead",
        "manage",
        "project",
        "skill",
        "skillset",
        "product",
        "service",
        "company",
        "client",
        "customer",
        "role",
        "position",
        "responsibility",
        "requirement",
    }
    filtered = [w for w in words if w.lower() not in common]
    return list(dict.fromkeys(filtered))[:min_count]


@app.post("/api/generate-answer")
async def generate_answer(request: GenerateAnswerRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured"}

    try:
        client = Groq(api_key=GROQ_API_KEY)
        context = knowledge_base.search_context(request.transcript, top_k=5)

        keywords = extract_keywords(context, 2) if context else []
        keyword_instruction = (
            f"IMPORTANT: Reference these specific keywords in your answer: {', '.join(keywords)}."
            if keywords
            else ""
        )

        role = request.role or "general"
        seniority = request.seniority or "mid"

        persona = f"You are a {seniority} {role} with 10 years of experience. Give practical, real-world answers."

        star_method = "For behavioral questions, use STAR method: Situation (brief context), Task (your responsibility), Action (what you did), Result (outcome + metrics)."

        system_prompt = f"""{persona}
{star_method}
{keyword_instruction}
The candidate's resume and job description are provided in the context below. Use specific examples from the resume that match the job requirements.
Give 2-3 concise bullet points starting with action verbs."""

        if context:
            system_prompt += f"\n\nResume/JD Context:\n{context}"

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.transcript},
            ],
            stream=True,
        )

        async def generate():
            async for chunk in _stream_groq_chunks(response):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
    except Exception as e:
        logger.error(f"[GENERATE] Error: {e}")
        return {"error": str(e)}


@app.websocket("/ws/ui")
async def websocket_ui(websocket: WebSocket):
    await connection_manager.connect(websocket, "ui")
    await websocket.send_json({"type": "status", "text": "Connected to Corebrum HUD"})
    print("[WS_UI] HUD CONNECTED - Ready to receive transcripts!")
    logger.info("[WS_UI] HUD connected - ready for transcripts")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "question":
                    transcript = msg.get("content", "")
                    if transcript:
                        logger.info(f"[WS_UI] Received question: {transcript[:50]}...")
                        await process_brain_ollama(transcript)
            except json.JSONDecodeError:
                logger.warning(f"[WS_UI] Invalid JSON: {data}")
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket, "ui")


async def send_periodic_ping(websocket: WebSocket, interval: float = 10.0):
    """Send periodic pings to keep connection alive and prevent timeout."""
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await websocket.send_text(json.dumps({"type": "ping"}))
                logger.debug("[WS_AUDIO] Sent ping to keepalive")
            except Exception:
                break
    except asyncio.CancelledError:
        pass


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    chunk_counter = 0
    last_byte_time = time.time()
    audio_timeout_seconds = 5.0
    logger.info(f"[WS_AUDIO] WebSocket opening - preparing to accept...")

    await connection_manager.connect(websocket, "audio")
    logger.info(f"[WS_AUDIO] Chrome Extension connected & accepted")

    ping_task = asyncio.create_task(send_periodic_ping(websocket, 10.0))

    # Audio heartbeat monitor
    async def audio_heartbeat_monitor():
        nonlocal last_byte_time
        while True:
            await asyncio.sleep(1.0)
            if time.time() - last_byte_time > audio_timeout_seconds:
                logger.warning("[WS_AUDIO] No audio data for 5s - sending alert")
                await connection_manager.broadcast(
                    "ui",
                    {
                        "type": "status",
                        "text": "WARNING: No audio data detected. Re-share tab with audio.",
                    },
                )

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
                    elif data.get("type") == "pong":
                        logger.debug("[WS_AUDIO] Received pong from client")
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
                last_byte_time = time.time()  # Reset timer on new data

    except WebSocketDisconnect:
        logger.info("[WS_AUDIO] Chrome Extension disconnected - scheduling reconnect")
    finally:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass
        await connection_manager.disconnect(websocket, "audio")
        logger.info("[WS_AUDIO] Notifying UI of reconnection needed")
        await connection_manager.broadcast_to_ui(
            {
                "type": "reconnect_needed",
                "text": "STT disconnected, please restart capture",
            }
        )


if __name__ == "__main__":
    logger.info(f"[SERVER] Starting on port {settings.port}")
    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")
