import asyncio
import queue
import numpy as np
import sounddevice as sd
import threading
import time
from typing import AsyncGenerator, Optional
from dataclasses import dataclass


try:
    from faster_whisper import WhisperModel
except ImportError:
    raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")


@dataclass
class TranscriptSegment:
    text: str
    started_at: float
    ended_at: float


class AudioHandler:
    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16000,
        buffer_duration: float = 0.5,
        overlap_duration: float = 0.1,
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.overlap_duration = overlap_duration
        self._stop_event = threading.Event()
        self._stream: Optional[sd.InputStream] = None
        self.last_non_zero_time = time.monotonic()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio Status: {status}")
        if not self._stop_event.is_set():
            audio_data = indata[:, 0].copy()
            self.audio_queue.put(audio_data)
            if np.max(np.abs(audio_data)) > 0.005:
                self.last_non_zero_time = time.monotonic()

    def start(self):
        self._stream = sd.InputStream(
            channels=1,
            samplerate=48000,
            blocksize=1024,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        self._stop_event.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()

    async def stream_transcription(self) -> AsyncGenerator[TranscriptSegment, None]:
        self.start()
        print("Live Ear Active - Listening...")

        buffer_samples = int(self.sample_rate * self.buffer_duration)
        overlap_samples = int(self.sample_rate * self.overlap_duration)
        audio_buffer = np.array([], dtype=np.float32)
        buffer_start_time = time.monotonic()

        while not self._stop_event.is_set():
            while len(audio_buffer) < buffer_samples:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    audio_buffer = np.concatenate([audio_buffer, chunk])
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    await asyncio.sleep(0.01)
                    continue

            if len(audio_buffer) < buffer_samples:
                continue

            segments, info = self.model.transcribe(
                audio_buffer,
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            for segment in segments:
                if segment.text.strip():
                    yield TranscriptSegment(
                        text=segment.text.strip(),
                        started_at=buffer_start_time,
                        ended_at=time.monotonic(),
                    )

            audio_buffer = audio_buffer[-overlap_samples:]
            buffer_start_time = (
                time.monotonic()
                - self.overlap_duration / self.sample_rate * len(audio_buffer)
            )


async def main():
    handler = AudioHandler()
    try:
        async for segment in handler.stream_transcription():
            print(f"[TRANSCRIPT] {segment.text}")
    except KeyboardInterrupt:
        handler.stop()


if __name__ == "__main__":
    asyncio.run(main())
