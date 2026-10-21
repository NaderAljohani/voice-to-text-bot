from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os
import math


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
STORED_AUDIO_DIR = DATA_DIR / "stored_audio"
DATABASE_PATH = DATA_DIR / "bot.sqlite3"

FREE_TRIAL_SECONDS = 5 * 60
MAX_TELEGRAM_MESSAGE_LENGTH = 3900


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    required_channel_username: str | None
    admin_telegram_ids: frozenset[int]
    admin_bypass_balance: bool
    max_audio_duration_seconds: int
    daily_free_minutes: int
    daily_quota_timezone: str
    store_audio_files: bool
    audio_retention_days: int
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_initial_prompt: str | None
    whisper_vad_filter: bool
    whisper_beam_size: int
    whisper_best_of: int
    whisper_no_speech_threshold: float
    whisper_condition_on_previous_text: bool
    transcript_qc_enabled: bool
    transcript_retry_on_qc_fail: bool
    min_meaningful_words: int
    prompt_leak_rejection: bool
    repetition_rejection: bool
    speech_vad_enabled: bool
    speech_vad_aggressiveness: int
    min_speech_seconds: float
    min_speech_ratio: float
    allowed_transcript_languages: frozenset[str]
    mixed_language_mode: bool
    chunk_transcription_enabled: bool
    chunk_fallback_always: bool
    chunk_seconds: int
    chunk_overlap_seconds: int
    chunk_min_seconds: int
    mixed_language_protection: bool
    language_rejection_soft: bool


def parse_admin_ids(raw_value: str) -> frozenset[int]:
    admin_ids: set[int] = set()
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            admin_ids.add(int(item))
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid ADMIN_TELEGRAM_IDS value: {item!r}. Use comma-separated numeric Telegram IDs."
            ) from exc
    return frozenset(admin_ids)


def load_settings() -> Settings:
    load_dotenv(BASE_DIR / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Add it to your .env file.")

    required_channel = os.getenv("REQUIRED_CHANNEL_USERNAME", "").strip() or None

    return Settings(
        telegram_bot_token=token,
        required_channel_username=required_channel,
        admin_telegram_ids=parse_admin_ids(os.getenv("ADMIN_TELEGRAM_IDS", "")),
        admin_bypass_balance=parse_bool(os.getenv("ADMIN_BYPASS_BALANCE", "false")),
        max_audio_duration_seconds=parse_positive_int(
            os.getenv("MAX_AUDIO_DURATION_SECONDS", "1800"),
            "MAX_AUDIO_DURATION_SECONDS",
        ),
        daily_free_minutes=parse_positive_int(os.getenv("DAILY_FREE_MINUTES", "10"), "DAILY_FREE_MINUTES"),
        daily_quota_timezone=os.getenv("DAILY_QUOTA_TIMEZONE", "Asia/Riyadh").strip() or "Asia/Riyadh",
        store_audio_files=parse_bool(os.getenv("STORE_AUDIO_FILES", "false")),
        audio_retention_days=parse_positive_int(os.getenv("AUDIO_RETENTION_DAYS", "7"), "AUDIO_RETENTION_DAYS"),
        whisper_model=os.getenv("WHISPER_MODEL", "large-v3").strip() or "large-v3",
        whisper_device=os.getenv("WHISPER_DEVICE", "cuda").strip() or "cuda",
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16").strip()
        or "int8_float16",
        whisper_initial_prompt=os.getenv("WHISPER_INITIAL_PROMPT", "").strip() or None,
        whisper_vad_filter=parse_bool(os.getenv("WHISPER_VAD_FILTER", "false")),
        whisper_beam_size=parse_positive_int(os.getenv("WHISPER_BEAM_SIZE", "5"), "WHISPER_BEAM_SIZE"),
        whisper_best_of=parse_positive_int(os.getenv("WHISPER_BEST_OF", "5"), "WHISPER_BEST_OF"),
        whisper_no_speech_threshold=parse_float(
            os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.8"),
            "WHISPER_NO_SPEECH_THRESHOLD",
        ),
        whisper_condition_on_previous_text=parse_bool(
            os.getenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false")
        ),
        transcript_qc_enabled=parse_bool(os.getenv("TRANSCRIPT_QC_ENABLED", "true")),
        transcript_retry_on_qc_fail=parse_bool(os.getenv("TRANSCRIPT_RETRY_ON_QC_FAIL", "true")),
        min_meaningful_words=parse_positive_int(os.getenv("MIN_MEANINGFUL_WORDS", "3"), "MIN_MEANINGFUL_WORDS"),
        prompt_leak_rejection=parse_bool(os.getenv("PROMPT_LEAK_REJECTION", "true")),
        repetition_rejection=parse_bool(os.getenv("REPETITION_REJECTION", "true")),
        speech_vad_enabled=parse_bool(os.getenv("SPEECH_VAD_ENABLED", "true")),
        speech_vad_aggressiveness=parse_int_in_range(
            os.getenv("SPEECH_VAD_AGGRESSIVENESS", "3"),
            "SPEECH_VAD_AGGRESSIVENESS",
            minimum=0,
            maximum=3,
        ),
        min_speech_seconds=parse_positive_float(os.getenv("MIN_SPEECH_SECONDS", "0.7"), "MIN_SPEECH_SECONDS"),
        min_speech_ratio=parse_positive_float(os.getenv("MIN_SPEECH_RATIO", "0.02"), "MIN_SPEECH_RATIO"),
        allowed_transcript_languages=parse_language_codes(os.getenv("ALLOWED_TRANSCRIPT_LANGUAGES", "ar,en")),
        mixed_language_mode=parse_bool(os.getenv("MIXED_LANGUAGE_MODE", "true")),
        chunk_transcription_enabled=parse_bool(os.getenv("CHUNK_TRANSCRIPTION_ENABLED", "true")),
        chunk_fallback_always=parse_bool(os.getenv("CHUNK_FALLBACK_ALWAYS", "false")),
        chunk_seconds=parse_positive_int(os.getenv("CHUNK_SECONDS", "20"), "CHUNK_SECONDS"),
        chunk_overlap_seconds=parse_non_negative_int(
            os.getenv("CHUNK_OVERLAP_SECONDS", "3"),
            "CHUNK_OVERLAP_SECONDS",
        ),
        chunk_min_seconds=parse_positive_int(os.getenv("CHUNK_MIN_SECONDS", "8"), "CHUNK_MIN_SECONDS"),
        mixed_language_protection=parse_bool(os.getenv("MIXED_LANGUAGE_PROTECTION", "true")),
        language_rejection_soft=parse_bool(os.getenv("LANGUAGE_REJECTION_SOFT", "true")),
    )


def parse_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_positive_int(raw_value: str, name: str) -> int:
    try:
        value = int(raw_value.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


def parse_non_negative_int(raw_value: str, name: str) -> int:
    try:
        value = int(raw_value.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a non-negative integer.") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer.")
    return value


def parse_float(raw_value: str, name: str) -> float:
    try:
        value = float(raw_value.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number.") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be a finite number.")
    return value


def parse_positive_float(raw_value: str, name: str) -> float:
    value = parse_float(raw_value, name)
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive number.")
    return value


def parse_int_in_range(raw_value: str, name: str, *, minimum: int, maximum: int) -> int:
    try:
        value = int(raw_value.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer from {minimum} to {maximum}.") from exc
    if value < minimum or value > maximum:
        raise RuntimeError(f"{name} must be an integer from {minimum} to {maximum}.")
    return value


def parse_language_codes(raw_value: str) -> frozenset[str]:
    codes = {item.strip().lower() for item in raw_value.split(",") if item.strip()}
    return frozenset(codes or {"ar", "en"})


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    STORED_AUDIO_DIR.mkdir(exist_ok=True)
