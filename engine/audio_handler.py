import asyncio
import logging
import queue
import numpy as np
import threading
import time
from collections import deque
from typing import AsyncGenerator, Optional
from dataclasses import dataclass

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

logger = logging.getLogger("corebrum.audio")


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
        signal_callback=None,
    ):
        logger.info("[AUDIO] Initializing WebSocket buffer receiver")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.overlap_duration = overlap_duration
        self._stop_event = threading.Event()
        self.last_non_zero_time = time.monotonic()
        self.signal_callback = signal_callback

        self.noise_floor_samples = deque(maxlen=100)
        self.dynamic_threshold = 0.01
        self.prebuffer_samples = int(0.5 * sample_rate)
        self.prebuffer = np.array([], dtype=np.float32)
        self.silence_start_time: Optional[float] = None

    def push_chunk(self, data: bytes):
        if not self._stop_event.is_set():
            pcm16 = np.frombuffer(data, dtype=np.int16)
            audio_data = pcm16.astype(np.float32) / 32768.0

            rms = float(np.sqrt(np.mean(np.square(audio_data))))
            if rms > 0:
                self.noise_floor_samples.append(rms)

            if len(self.noise_floor_samples) >= 10:
                avg_noise = sum(self.noise_floor_samples) / len(
                    self.noise_floor_samples
                )
                self.dynamic_threshold = avg_noise * 1.1

            self.prebuffer = np.concatenate([self.prebuffer, audio_data])
            if len(self.prebuffer) > self.prebuffer_samples:
                self.prebuffer = self.prebuffer[-self.prebuffer_samples :]

            self.audio_queue.put(audio_data)

            max_amp = np.max(np.abs(audio_data))
            logger.debug(
                f"[AUDIO_CHUNK] {len(audio_data)} samples, max_amp={max_amp:.6f}"
            )
            if max_amp > 0.005:
                self.last_non_zero_time = time.monotonic()

    def reset_buffer(self):
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.prebuffer = np.array([], dtype=np.float32)
        logger.info("[AUDIO] Internal audio buffer reset.")

    def start(self):
        pass

    def stop(self):
        self._stop_event.set()

    async def stream_transcription(self) -> AsyncGenerator[TranscriptSegment, None]:
        self.start()
        logger.info("Live Ear Active - Listening...")

        buffer_samples = int(self.sample_rate * self.buffer_duration)
        overlap_samples = int(self.sample_rate * self.overlap_duration)
        audio_buffer = np.array([], dtype=np.float32)
        buffer_start_time = time.monotonic()
        last_signal_print = 0.0
        is_speaking = False

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

            audio_boosted = audio_buffer * 2.5
            audio_boosted = np.clip(audio_boosted, -1.0, 1.0)

            raw_amplitude = float(np.max(np.abs(audio_buffer)))
            rms_amplitude = float(np.sqrt(np.mean(np.square(audio_buffer))))

            logger.debug(
                f"[VAD_CHECK] raw={raw_amplitude:.4f}, rms={rms_amplitude:.4f}, threshold={self.dynamic_threshold:.4f}"
            )

            if self.signal_callback:
                self.signal_callback(raw_amplitude)

            if time.monotonic() - last_signal_print >= 1.0:
                logger.info(
                    f"[SIGNAL_CHECK] RMS: {rms_amplitude:.4f} | Peak: {raw_amplitude:.4f} | Threshold: {self.dynamic_threshold:.4f}"
                )
                last_signal_print = time.monotonic()

            below_threshold = raw_amplitude < 0.01

            if below_threshold:
                if self.silence_start_time is None:
                    self.silence_start_time = time.monotonic()
                elif time.monotonic() - self.silence_start_time > 2.0:
                    if is_speaking:
                        logger.info("[AUDIO] Silence > 2s - ending speech segment")
                        is_speaking = False
                    logger.debug(
                        "[AUDIO] Silence > 2s - skipping Whisper to save power"
                    )
                    audio_buffer = audio_buffer[-overlap_samples:]
                    buffer_start_time = (
                        time.monotonic()
                        - self.overlap_duration / self.sample_rate * len(audio_buffer)
                    )
                    await asyncio.sleep(0.01)
                    continue
            else:
                self.silence_start_time = None

            if raw_amplitude > self.dynamic_threshold:
                if not is_speaking:
                    is_speaking = True
                    if len(self.prebuffer) > 0:
                        buffered_audio = self.prebuffer[-self.prebuffer_samples :]
                        audio_buffer = np.concatenate([buffered_audio, audio_buffer])
                        buffer_start_time = time.monotonic() - 0.5
                        logger.info(
                            "[VAD] Voice detected - using prebuffer (500ms lead-in)"
                        )

                segments, info = self.model.transcribe(
                    audio_boosted,
                    beam_size=1,
                    vad_filter=False,
                )

                segment_count = 0

                for segment in segments:
                    segment_count += 1
                    if segment.text.strip():
                        current_time = time.strftime("%H:%M:%S.%f")[:-3]
                        logger.info(
                            f"[TRANSCRIPT_LIVE] [{current_time}] {segment.text.strip()}"
                        )
                        yield TranscriptSegment(
                            text=segment.text.strip(),
                            started_at=buffer_start_time,
                            ended_at=time.monotonic(),
                        )

                if segment_count > 0:
                    logger.debug(f"[VAD] Processed {segment_count} segment(s)")
            else:
                logger.debug(
                    f"[VAD] Below dynamic threshold ({raw_amplitude:.4f} < {self.dynamic_threshold:.4f})"
                )

            audio_buffer = audio_buffer[-overlap_samples:]
            buffer_start_time = (
                time.monotonic()
                - self.overlap_duration / self.sample_rate * len(audio_buffer)
            )

            await asyncio.sleep(0.01)


async def main():
    handler = AudioHandler()
    try:
        async for segment in handler.stream_transcription():
            logger.info(f"[TRANSCRIPT] {segment.text}")
    except KeyboardInterrupt:
        handler.stop()


if __name__ == "__main__":
    asyncio.run(main())
