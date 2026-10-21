import cuda_dll_paths  # noqa: F401
import logging
import re
import tempfile
import wave
from threading import Lock
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    formatted_text: str
    segments: list[TranscriptionSegment]
    detected_language: str | None
    language_probability: float | None


@dataclass(frozen=True)
class AudioChunk:
    start_seconds: float
    end_seconds: float
    start_sample: int
    end_sample: int


class GpuSetupError(RuntimeError):
    """Raised when CUDA is requested but required NVIDIA runtime libraries are missing."""


class LocalWhisperTranscriber:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        initial_prompt: str | None = None,
        vad_filter: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        no_speech_threshold: float = 0.8,
        condition_on_previous_text: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.best_of = best_of
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self._model: WhisperModel | None = None
        self._inference_lock = Lock()

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as exc:
                if self.device.lower() == "cuda" and is_cuda_setup_error(exc):
                    raise GpuSetupError(
                        "CUDA was requested, but CUDA/cuBLAS/cuDNN runtime libraries are not available."
                    ) from exc
                raise
        return self._model

    def transcribe_normal(
        self,
        wav_path: Path,
        *,
        use_initial_prompt: bool = True,
        condition_on_previous_text: bool | None = None,
    ) -> TranscriptionResult:
        # One model instance serves all Telegram users in this process.
        with self._inference_lock:
            return self._transcribe_normal(
                wav_path,
                use_initial_prompt=use_initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
            )

    def _transcribe_normal(
        self,
        wav_path: Path,
        *,
        use_initial_prompt: bool = True,
        condition_on_previous_text: bool | None = None,
    ) -> TranscriptionResult:
        try:
            transcribe_options = {
                "language": None,
                "task": "transcribe",
                "condition_on_previous_text": (
                    self.condition_on_previous_text
                    if condition_on_previous_text is None
                    else condition_on_previous_text
                ),
                "vad_filter": self.vad_filter,
                "beam_size": self.beam_size,
                "best_of": self.best_of,
                "no_speech_threshold": self.no_speech_threshold,
            }
            if use_initial_prompt and self.initial_prompt:
                transcribe_options["initial_prompt"] = self.initial_prompt

            segments, _info = self.model.transcribe(str(wav_path), **transcribe_options)
            logger.info(
                "Detected transcription language=%s probability=%.4f",
                getattr(_info, "language", "unknown"),
                float(getattr(_info, "language_probability", 0.0) or 0.0),
            )
            segment_texts: list[str] = []
            transcription_segments: list[TranscriptionSegment] = []
            for segment in segments:
                segment_text = segment.text.strip() if segment.text else ""
                start = float(segment.start)
                end = float(segment.end)
                logger.debug(
                    "Transcription segment start=%.2f end=%.2f",
                    start,
                    end,
                )
                if segment_text:
                    segment_texts.append(segment_text)
                    transcription_segments.append(
                        TranscriptionSegment(
                            start=start,
                            end=end,
                            text=segment_text,
                            no_speech_prob=as_optional_float(getattr(segment, "no_speech_prob", None)),
                            avg_logprob=as_optional_float(getattr(segment, "avg_logprob", None)),
                            compression_ratio=as_optional_float(getattr(segment, "compression_ratio", None)),
                        )
                    )
            text = " ".join(segment_texts)
        except Exception as exc:
            if self.device.lower() == "cuda" and is_cuda_setup_error(exc):
                raise GpuSetupError(
                    "CUDA was requested, but CUDA/cuBLAS/cuDNN runtime libraries are not available."
                ) from exc
            raise
        return TranscriptionResult(
            text=clean_transcript(text),
            formatted_text=format_timestamped_transcript(transcription_segments),
            segments=transcription_segments,
            detected_language=getattr(_info, "language", None),
            language_probability=getattr(_info, "language_probability", None),
        )

    def transcribe_with_speakers(self, wav_path: Path) -> str:
        # TODO: Add optional diarization later using WhisperX + pyannote.
        # Planned flow:
        # 1. Run WhisperX alignment on the faster-whisper transcript.
        # 2. Run pyannote speaker diarization with a user-provided Hugging Face token.
        # 3. Merge words/segments with speaker labels.
        # 4. Return a readable transcript such as "Speaker 1: ...".
        raise NotImplementedError("Speaker diarization is planned for a future optional mode.")

    def transcribe_chunks(
        self,
        wav_path: Path,
        *,
        chunk_seconds: int,
        overlap_seconds: int,
        min_chunk_seconds: int,
    ) -> tuple[TranscriptionResult, int]:
        chunks = split_wav_into_chunks(
            wav_path,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            min_chunk_seconds=min_chunk_seconds,
        )
        all_segments: list[TranscriptionSegment] = []
        detected_languages: list[str] = []
        language_probabilities: list[float] = []

        with tempfile.TemporaryDirectory(prefix="voicebot_chunks_") as temp_dir:
            temp_path = Path(temp_dir)
            for index, chunk in enumerate(chunks):
                chunk_path = temp_path / f"chunk_{index:04d}.wav"
                write_wav_chunk(wav_path, chunk_path, chunk.start_sample, chunk.end_sample)
                result = self.transcribe_normal(
                    chunk_path,
                    use_initial_prompt=True,
                    condition_on_previous_text=False,
                )
                if result.detected_language:
                    detected_languages.append(result.detected_language)
                if result.language_probability is not None:
                    language_probabilities.append(float(result.language_probability))
                for segment in result.segments:
                    shifted = TranscriptionSegment(
                        start=segment.start + chunk.start_seconds,
                        end=segment.end + chunk.start_seconds,
                        text=segment.text,
                        no_speech_prob=segment.no_speech_prob,
                        avg_logprob=segment.avg_logprob,
                        compression_ratio=segment.compression_ratio,
                    )
                    if shifted.end <= chunk.end_seconds + 2.0:
                        all_segments.append(shifted)

        text = clean_transcript(" ".join(segment.text for segment in all_segments))
        language = most_common(detected_languages)
        probability = (
            sum(language_probabilities) / len(language_probabilities)
            if language_probabilities
            else None
        )
        return (
            TranscriptionResult(
                text=text,
                formatted_text=format_timestamped_transcript(all_segments),
                segments=all_segments,
                detected_language=language,
                language_probability=probability,
            ),
            len(chunks),
        )


def clean_transcript(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.!?؟،:;])", r"\1", text)
    text = re.sub(r"([.!?؟])\s+", r"\1\n", text)
    return text.strip()


def as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_timestamped_transcript(segments: list[TranscriptionSegment]) -> str:
    lines = [
        f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}] {segment.text}"
        for segment in segments
        if segment.text
    ]
    return "\n".join(lines).strip()


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds_part = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def split_wav_into_chunks(
    wav_path: Path,
    *,
    chunk_seconds: int,
    overlap_seconds: int,
    min_chunk_seconds: int,
) -> list[AudioChunk]:
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        total_samples = wav_file.getnframes()

    total_seconds = total_samples / sample_rate if sample_rate else 0
    if total_seconds <= 0:
        return []
    if total_seconds <= chunk_seconds:
        return [AudioChunk(0.0, total_seconds, 0, total_samples)]

    chunk_seconds = max(min_chunk_seconds, chunk_seconds)
    overlap_seconds = max(0, min(overlap_seconds, chunk_seconds - 1))
    step_seconds = chunk_seconds - overlap_seconds
    chunks: list[AudioChunk] = []
    start = 0.0
    while start < total_seconds:
        end = min(total_seconds, start + chunk_seconds)
        if chunks and end - start < min_chunk_seconds:
            previous = chunks[-1]
            chunks[-1] = AudioChunk(
                start_seconds=previous.start_seconds,
                end_seconds=total_seconds,
                start_sample=previous.start_sample,
                end_sample=total_samples,
            )
            break
        chunks.append(
            AudioChunk(
                start_seconds=start,
                end_seconds=end,
                start_sample=int(start * sample_rate),
                end_sample=min(total_samples, int(end * sample_rate)),
            )
        )
        if end >= total_seconds:
            break
        start += step_seconds
    return chunks


def write_wav_chunk(source_path: Path, chunk_path: Path, start_sample: int, end_sample: int) -> None:
    with wave.open(str(source_path), "rb") as source:
        params = source.getparams()
        source.setpos(start_sample)
        frames = source.readframes(max(0, end_sample - start_sample))
    with wave.open(str(chunk_path), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)


def most_common(values: list[str]) -> str | None:
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def is_cuda_setup_error(exc: Exception) -> bool:
    message = str(exc).lower()
    cuda_markers = (
        "cuda",
        "cublas",
        "cudnn",
        "cudart",
        "ctranslate2",
        "nvidia",
        "dll",
        "dynamic library",
        "library",
    )
    missing_markers = (
        "not found",
        "could not load",
        "cannot load",
        "failed to load",
        "load library",
        "no such file",
        "missing",
        "not available",
    )
    return any(marker in message for marker in cuda_markers) and any(
        marker in message for marker in missing_markers
    )
