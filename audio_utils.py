import asyncio
import logging
import math
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechDetectionResult:
    available: bool
    total_audio_seconds: float
    detected_speech_seconds: float
    speech_ratio: float
    reason: str | None = None


def ensure_ffmpeg_available() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"Missing required audio tool(s): {', '.join(missing)}. Install ffmpeg and make sure it is in PATH."
        )


def is_speech_vad_available() -> bool:
    try:
        import webrtcvad  # noqa: F401
    except ImportError:
        return False
    return True


async def _run_command(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(message or f"Command failed: {' '.join(args)}")
    return stdout.decode("utf-8", errors="ignore").strip()


async def get_audio_duration_seconds(input_path: Path) -> int:
    output = await _run_command(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    )
    durations: list[float] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.upper() == "N/A":
            continue
        try:
            durations.append(float(line))
        except ValueError:
            continue

    if not durations:
        raise RuntimeError("Could not read audio duration with ffprobe.")

    duration = max(durations)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Invalid audio duration reported by ffprobe.")

    return max(1, math.ceil(duration))


async def has_audio_stream(input_path: Path) -> bool:
    output = await _run_command(
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(input_path),
    )
    return bool(output.strip())


async def convert_to_wav_16k_mono(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    await _run_command(
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    )
    return output_path


def detect_speech_in_wav(
    wav_path: Path,
    *,
    aggressiveness: int = 3,
    frame_ms: int = 30,
) -> SpeechDetectionResult:
    try:
        import webrtcvad
    except ImportError:
        logger.warning("SPEECH_VAD_ENABLED=true but webrtcvad is not installed. Continuing without VAD.")
        return SpeechDetectionResult(
            available=False,
            total_audio_seconds=0.0,
            detected_speech_seconds=0.0,
            speech_ratio=0.0,
            reason="webrtcvad_unavailable",
        )

    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        total_audio_seconds = frame_count / sample_rate if sample_rate else 0.0
        pcm = wav_file.readframes(frame_count)

    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        logger.warning(
            "Speech VAD expected 16kHz mono 16-bit PCM WAV, got channels=%s sample_width=%s sample_rate=%s",
            channels,
            sample_width,
            sample_rate,
        )
        return SpeechDetectionResult(
            available=False,
            total_audio_seconds=total_audio_seconds,
            detected_speech_seconds=0.0,
            speech_ratio=0.0,
            reason="unsupported_wav_format",
        )

    vad = webrtcvad.Vad(max(0, min(3, aggressiveness)))
    bytes_per_sample = sample_width
    frame_size = int(sample_rate * frame_ms / 1000) * bytes_per_sample
    if frame_size <= 0:
        return SpeechDetectionResult(
            available=False,
            total_audio_seconds=total_audio_seconds,
            detected_speech_seconds=0.0,
            speech_ratio=0.0,
            reason="invalid_frame_size",
        )

    speech_frames = 0
    total_frames = 0
    for offset in range(0, len(pcm) - frame_size + 1, frame_size):
        frame = pcm[offset : offset + frame_size]
        total_frames += 1
        if vad.is_speech(frame, sample_rate):
            speech_frames += 1

    detected_speech_seconds = speech_frames * frame_ms / 1000
    analyzed_seconds = total_frames * frame_ms / 1000
    speech_ratio = detected_speech_seconds / analyzed_seconds if analyzed_seconds else 0.0
    return SpeechDetectionResult(
        available=True,
        total_audio_seconds=total_audio_seconds,
        detected_speech_seconds=detected_speech_seconds,
        speech_ratio=speech_ratio,
    )
