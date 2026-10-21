import asyncio
import logging
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import FSInputFile, Message

from audio_utils import (
    convert_to_wav_16k_mono,
    detect_speech_in_wav,
    ensure_ffmpeg_available,
    get_audio_duration_seconds,
    has_audio_stream,
    is_speech_vad_available,
)
from config import (
    DATABASE_PATH,
    DOWNLOADS_DIR,
    MAX_TELEGRAM_MESSAGE_LENGTH,
    STORED_AUDIO_DIR,
    TRANSCRIPTS_DIR,
    ensure_directories,
    load_settings,
)
from database import Database
from messages import (
    ADMIN_BYPASS_BALANCE_MESSAGE,
    ADMIN_NO_LOGS,
    ADMIN_ONLY,
    ADMIN_PANEL,
    ADMIN_USAGE_ADMIN,
    ADMIN_USAGE_BAN,
    ADMIN_USAGE_ADDMINUTES,
    ADMIN_USAGE_LOGS,
    ADMIN_USAGE_SETBALANCE,
    ADMIN_USAGE_USER,
    ADMIN_USAGE_UNBAN,
    ADMIN_USAGE_USERLOGS,
    BANNED_USER,
    EMPTY_TRANSCRIPT,
    ERROR_OCCURRED,
    FILE_TOO_LONG_LIMIT,
    FREE_QUOTA_ENDED,
    GPU_SETUP_ERROR_ADMIN,
    GPU_SETUP_ERROR_PUBLIC,
    INSUFFICIENT_BALANCE_FOR_FILE,
    NO_CLEAR_SPEECH,
    PROCESSING_STARTED,
    RAW_TRANSCRIPT_CAPTION,
    RECEIVING_FILE,
    START,
    TIMESTAMPED_TRANSCRIPT_CAPTION,
    TRANSCRIPTION_COMPLETED,
    UNSUPPORTED_FILE_FORMAT,
    UNSUPPORTED_MESSAGE,
    USAGE_BALANCE,
    USAGE_MYID,
    USER_ALREADY_PROCESSING,
    VIDEO_HAS_NO_AUDIO,
    daily_balance_message,
)
from transcript_qc import TranscriptQcSettings, validate_transcription
from transcriber import (
    GpuSetupError,
    LocalWhisperTranscriber,
    TranscriptionResult,
    TranscriptionSegment,
    clean_transcript,
    format_timestamped_transcript,
    split_wav_into_chunks,
    write_wav_chunk,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

router = Router()
db: Database
transcriber: LocalWhisperTranscriber
admin_telegram_ids: frozenset[int] = frozenset()
admin_bypass_balance = False
store_audio_files = False
audio_retention_days = 7
max_audio_duration_seconds = 1800
daily_free_minutes = 10
daily_quota_timezone = "Asia/Riyadh"
transcript_qc_enabled = True
transcript_retry_on_qc_fail = True
min_meaningful_words = 3
prompt_leak_rejection = True
repetition_rejection = True
speech_vad_enabled = True
speech_vad_aggressiveness = 3
min_speech_seconds = 0.7
min_speech_ratio = 0.02
allowed_transcript_languages: frozenset[str] = frozenset({"ar", "en"})
mixed_language_mode = True
chunk_transcription_enabled = True
chunk_fallback_always = False
chunk_seconds = 20
chunk_overlap_seconds = 3
chunk_min_seconds = 8
mixed_language_protection = True
language_rejection_soft = True
user_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
TRANSCRIPT_PREVIEW_LENGTH = 700
ACCEPTED_EXTENSIONS = {
    ".ogg",
    ".oga",
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".webm",
    ".aac",
    ".flac",
}
ACCEPTED_MIME_TYPES = {
    "audio/ogg",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/aac",
    "audio/flac",
    "audio/webm",
    "video/mp4",
    "video/webm",
}
COMMAND_USAGE_MESSAGES = {
    "/addminutes": ADMIN_USAGE_ADDMINUTES,
    "/setbalance": ADMIN_USAGE_SETBALANCE,
    "/userlogs": ADMIN_USAGE_USERLOGS,
    "/unban": ADMIN_USAGE_UNBAN,
    "/admin": ADMIN_USAGE_ADMIN,
    "/balance": USAGE_BALANCE,
    "/ban": ADMIN_USAGE_BAN,
    "/logs": ADMIN_USAGE_LOGS,
    "/myid": USAGE_MYID,
    "/user": ADMIN_USAGE_USER,
}
ADMIN_ONLY_COMMANDS = {
    "/admin",
    "/ban",
    "/unban",
    "/logs",
    "/userlogs",
    "/addminutes",
    "/setbalance",
    "/user",
}


def split_for_telegram(text: str, max_length: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> list[str]:
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        chunk = remaining[:max_length]
        split_at = max(
            chunk.rfind("\n"),
            chunk.rfind(". "),
            chunk.rfind("! "),
            chunk.rfind("? "),
            chunk.rfind("؟ "),
        )
        if split_at > max_length // 2:
            chunk = remaining[: split_at + 1]
        chunks.append(chunk.strip())
        remaining = remaining[len(chunk) :].strip()
    return chunks


def transcript_path_for(message: Message) -> Path:
    safe_name = f"transcript_{message.from_user.id}_{message.message_id}.txt"
    return TRANSCRIPTS_DIR / safe_name


def timestamped_transcript_path_for(message: Message) -> Path:
    safe_name = f"transcript_timestamped_{message.from_user.id}_{message.message_id}.txt"
    return TRANSCRIPTS_DIR / safe_name


def transcript_preview(text: str) -> str:
    preview = text[:TRANSCRIPT_PREVIEW_LENGTH].strip()
    if len(text) > TRANSCRIPT_PREVIEW_LENGTH:
        preview = f"{preview}..."
    return preview


def current_quota_date() -> str:
    return datetime.now(get_quota_zone()).date().isoformat()


def get_quota_zone() -> tzinfo:
    try:
        return ZoneInfo(daily_quota_timezone)
    except ZoneInfoNotFoundError:
        logger.exception("Invalid DAILY_QUOTA_TIMEZONE=%s. Falling back to Asia/Riyadh.", daily_quota_timezone)
        return timezone(timedelta(hours=3), name="Asia/Riyadh")


def daily_free_seconds() -> int:
    return daily_free_minutes * 60


def balance_text_for_user(telegram_id: int) -> str:
    if can_bypass_balance(telegram_id):
        return ADMIN_BYPASS_BALANCE_MESSAGE
    info = db.balance_info(telegram_id, daily_free_seconds(), current_quota_date())
    return daily_balance_message(
        daily_remaining_seconds=info["daily_remaining_seconds"],
        daily_free_minutes=daily_free_minutes,
        extra_balance_seconds=info["extra_balance_seconds"],
    )


def is_admin(telegram_id: int) -> bool:
    return telegram_id in admin_telegram_ids


def can_bypass_balance(telegram_id: int) -> bool:
    return admin_bypass_balance and is_admin(telegram_id)


def gpu_error_message_for_user(telegram_id: int) -> str:
    if is_admin(telegram_id):
        return GPU_SETUP_ERROR_ADMIN
    return GPU_SETUP_ERROR_PUBLIC


def current_qc_settings() -> TranscriptQcSettings:
    return TranscriptQcSettings(
        enabled=transcript_qc_enabled,
        min_meaningful_words=min_meaningful_words,
        prompt_leak_rejection=prompt_leak_rejection,
        repetition_rejection=repetition_rejection,
        initial_prompt=transcriber.initial_prompt,
        mixed_language_protection=mixed_language_protection,
        language_rejection_soft=language_rejection_soft,
        allowed_languages=allowed_transcript_languages,
    )


def current_qc_settings_with_vad(vad_ratio: float | None) -> TranscriptQcSettings:
    return TranscriptQcSettings(
        enabled=transcript_qc_enabled,
        min_meaningful_words=min_meaningful_words,
        prompt_leak_rejection=prompt_leak_rejection,
        repetition_rejection=repetition_rejection,
        initial_prompt=transcriber.initial_prompt,
        mixed_language_protection=mixed_language_protection,
        language_rejection_soft=language_rejection_soft,
        allowed_languages=allowed_transcript_languages,
        vad_speech_ratio=vad_ratio,
    )


def speech_detection_rejects(
    *,
    total_audio_seconds: float,
    detected_speech_seconds: float,
    speech_ratio: float,
) -> bool:
    if total_audio_seconds < 3:
        return detected_speech_seconds <= 0
    if detected_speech_seconds < min_speech_seconds:
        return True
    return total_audio_seconds > 3 and speech_ratio < min_speech_ratio


def has_arabic_text(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)


def has_latin_text(text: str) -> bool:
    return any(("a" <= char.lower() <= "z") for char in text)


def transcript_reasonable_for_duration(text: str, duration_seconds: float | None) -> bool:
    if duration_seconds is None or duration_seconds <= 5:
        return True
    word_count = len(text.split())
    if duration_seconds > 20 and word_count < 3:
        return False
    if duration_seconds > 60 and word_count < 8:
        return False
    return True


def should_run_chunk_fallback(
    result: TranscriptionResult,
    qc_passed: bool,
    qc_reason: str | None,
    qc_severity: str | None,
    duration_seconds: float | None,
) -> bool:
    if not (mixed_language_mode and chunk_transcription_enabled):
        return False
    if chunk_fallback_always:
        return True
    detected_language = (result.detected_language or "").lower()
    if not qc_passed:
        return True
    text = result.text or ""
    if qc_severity in {"pass", "soft_warning"} and transcript_reasonable_for_duration(text, duration_seconds):
        if has_arabic_text(text) or has_latin_text(text):
            return False
    if detected_language and detected_language not in allowed_transcript_languages:
        return not (has_arabic_text(text) or has_latin_text(text))
    if qc_reason in {
        "generic_hallucination_phrase",
        "short_generic_hallucination",
        "transcript_audio_length_mismatch",
        "segment_duration_exceeds_audio",
    }:
        return True
    return not transcript_reasonable_for_duration(result.text or "", duration_seconds)


def choose_transcription_result(
    full_result: TranscriptionResult,
    full_qc_passed: bool,
    full_qc_reason: str | None,
    chunk_result: TranscriptionResult | None,
    chunk_qc_passed: bool,
    *,
    duration_seconds: float | None,
    chunk_valid_ratio: float | None,
) -> tuple[str, TranscriptionResult | None, str | None]:
    if chunk_result is None or not chunk_qc_passed:
        return ("full", full_result if full_qc_passed else None, full_qc_reason)
    if chunk_valid_ratio is not None and chunk_valid_ratio < 0.70:
        return ("full", full_result if full_qc_passed else None, full_qc_reason or "chunk_qc_failed")
    if not full_qc_passed:
        return ("chunk", chunk_result, None)

    full_text = full_result.text or ""
    chunk_text = chunk_result.text or ""
    full_words = len(full_text.split())
    chunk_words = len(chunk_text.split())
    full_has_mixed = has_arabic_text(full_text) and has_latin_text(full_text)
    chunk_has_mixed = has_arabic_text(chunk_text) and has_latin_text(chunk_text)

    if chunk_has_mixed and not full_has_mixed and chunk_words >= max(2, full_words):
        return ("chunk", chunk_result, None)
    if not transcript_reasonable_for_duration(full_text, duration_seconds) and chunk_words >= full_words:
        return ("chunk", chunk_result, None)
    if chunk_words >= full_words + max(8, full_words // 2):
        return ("chunk", chunk_result, None)
    return ("full", full_result, full_qc_reason)


async def transcribe_chunks_with_qc(
    wav_path: Path,
    *,
    duration_seconds: float | None,
) -> tuple[TranscriptionResult | None, str, int, int, int]:
    chunks = split_wav_into_chunks(
        wav_path,
        chunk_seconds=chunk_seconds,
        overlap_seconds=chunk_overlap_seconds,
        min_chunk_seconds=chunk_min_seconds,
    )
    valid_segments: list[TranscriptionSegment] = []
    detected_languages: list[str] = []
    language_probabilities: list[float] = []
    rejected_count = 0

    with tempfile.TemporaryDirectory(prefix="voicebot_chunks_") as temp_dir:
        temp_path = Path(temp_dir)
        for index, chunk in enumerate(chunks):
            chunk_path = temp_path / f"chunk_{index:04d}.wav"
            write_wav_chunk(wav_path, chunk_path, chunk.start_sample, chunk.end_sample)
            result = await asyncio.to_thread(
                transcriber.transcribe_normal,
                chunk_path,
                use_initial_prompt=True,
                condition_on_previous_text=False,
            )
            chunk_duration = chunk.end_seconds - chunk.start_seconds
            if chunk_timestamp_overflow(result, chunk_duration):
                rejected_count += 1
                continue
            qc_result = validate_transcription(
                result,
                duration_seconds=chunk_duration,
                settings=current_qc_settings(),
            )
            if qc_result.reason in {"prompt_leak_detected", "generic_hallucination_phrase", "segment_duration_exceeds_audio"}:
                rejected_count += 1
                continue
            if not qc_result.passed and transcript_retry_on_qc_fail:
                retry_result = await asyncio.to_thread(
                    transcriber.transcribe_normal,
                    chunk_path,
                    use_initial_prompt=False,
                    condition_on_previous_text=False,
                )
                if chunk_timestamp_overflow(retry_result, chunk_duration):
                    rejected_count += 1
                    continue
                qc_result = validate_transcription(
                    retry_result,
                    duration_seconds=chunk_duration,
                    settings=current_qc_settings(),
                )
                if qc_result.reason in {"prompt_leak_detected", "generic_hallucination_phrase", "segment_duration_exceeds_audio"}:
                    rejected_count += 1
                    continue
            if not qc_result.passed or qc_result.result is None:
                rejected_count += 1
                continue

            accepted = qc_result.result
            if accepted.detected_language:
                detected_languages.append(accepted.detected_language)
            if accepted.language_probability is not None:
                language_probabilities.append(float(accepted.language_probability))
            for segment in accepted.segments:
                shifted = TranscriptionSegment(
                    start=segment.start + chunk.start_seconds,
                    end=segment.end + chunk.start_seconds,
                    text=segment.text,
                    no_speech_prob=segment.no_speech_prob,
                    avg_logprob=segment.avg_logprob,
                    compression_ratio=segment.compression_ratio,
                )
                if duration_seconds is None or shifted.end <= duration_seconds + 2.0:
                    valid_segments.append(shifted)

    chunk_count = len(chunks)
    valid_chunk_count = chunk_count - rejected_count
    if chunk_count == 0 or valid_chunk_count == 0 or (valid_chunk_count / chunk_count) < 0.70:
        return (None, "chunk_qc_failed", chunk_count, valid_chunk_count, rejected_count)

    text = clean_transcript(" ".join(segment.text for segment in valid_segments))
    result = TranscriptionResult(
        text=text,
        formatted_text=format_timestamped_transcript(valid_segments),
        segments=valid_segments,
        detected_language=most_common(detected_languages),
        language_probability=(
            sum(language_probabilities) / len(language_probabilities)
            if language_probabilities
            else None
        ),
    )
    aggregate_qc = validate_transcription(
        result,
        duration_seconds=duration_seconds,
        settings=current_qc_settings(),
    )
    if not aggregate_qc.passed or aggregate_qc.result is None:
        return (None, aggregate_qc.reason or "chunk_qc_failed", chunk_count, valid_chunk_count, rejected_count)
    return (aggregate_qc.result, "passed", chunk_count, valid_chunk_count, rejected_count)


def most_common(values: list[str]) -> str | None:
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def chunk_timestamp_overflow(result: TranscriptionResult, chunk_duration: float) -> bool:
    if not result.segments:
        return False
    return max(segment.end for segment in result.segments) > chunk_duration + 1.5


async def reject_if_banned(message: Message) -> bool:
    if not message.from_user:
        return False
    if db.is_banned(message.from_user.id):
        await message.answer(BANNED_USER)
        return True
    return False


async def require_admin(message: Message) -> bool:
    if await reject_if_banned(message):
        return False
    if not message.from_user or not is_admin(message.from_user.id):
        await message.answer(ADMIN_ONLY)
        return False
    return True


async def register_user(message: Message) -> None:
    user = message.from_user
    db.get_or_create_user(
        telegram_id=user.id,
        username=user.username,
        first_name=user.first_name,
    )


def get_media(message: Message):
    return message.voice or message.audio or message.document or message.video or message.video_note


def media_file_name(message: Message) -> str | None:
    media = get_media(message)
    return getattr(media, "file_name", None)


def media_suffix(message: Message) -> str:
    if message.voice:
        return ".ogg"
    if message.video:
        suffix = Path(media_file_name(message) or "").suffix.lower()
        return suffix if suffix else ".mp4"
    if message.video_note:
        return ".mp4"
    suffix = Path(media_file_name(message) or "").suffix.lower()
    return suffix if suffix else ".audio"


def media_type_for(message: Message) -> str:
    if message.voice:
        return "voice"
    if message.audio:
        return "audio"
    if message.video:
        return "video"
    if message.video_note:
        return "video_note"
    if message.document:
        return "document"
    return "unknown"


def is_supported_media(message: Message) -> bool:
    if message.voice:
        return True
    if message.video_note:
        return True
    media = get_media(message)
    if media is None:
        return False
    file_name = getattr(media, "file_name", None)
    suffix = Path(file_name or "").suffix.lower()
    mime_type = (getattr(media, "mime_type", None) or "").lower()
    if suffix in ACCEPTED_EXTENSIONS:
        return True
    if mime_type in ACCEPTED_MIME_TYPES:
        return True
    return mime_type.startswith("audio/")


def media_metadata(message: Message) -> dict[str, object | None]:
    user = message.from_user
    media = get_media(message)
    return {
        "telegram_id": user.id,
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "chat_id": message.chat.id,
        "message_id": message.message_id,
        "telegram_file_id": getattr(media, "file_id", None),
        "telegram_file_unique_id": getattr(media, "file_unique_id", None),
        "original_file_name": getattr(media, "file_name", None),
        "file_size_bytes": getattr(media, "file_size", None),
    }


def malformed_known_command_usage(message: Message) -> tuple[str, str] | None:
    text = (message.text or "").strip()
    if not text.startswith("/"):
        return None

    for command_name in sorted(COMMAND_USAGE_MESSAGES, key=len, reverse=True):
        if text.startswith(command_name):
            return command_name, COMMAND_USAGE_MESSAGES[command_name]
    return None


async def answer_malformed_known_command(message: Message) -> bool:
    malformed_command = malformed_known_command_usage(message)
    if malformed_command is None:
        return False

    command_name, usage_message = malformed_command
    if command_name in ADMIN_ONLY_COMMANDS and not await require_admin(message):
        return True

    await message.answer(usage_message)
    return True


def save_transcription_log(
    message: Message,
    *,
    status: str,
    duration_seconds: float | None = None,
    detected_language: str | None = None,
    transcript: str | None = None,
    formatted_transcript: str | None = None,
    audio_path: str | None = None,
    error_message_internal: str | None = None,
    qc_status: str | None = None,
    qc_reason: str | None = None,
    retry_used: bool = False,
    transcript_preview_internal: str | None = None,
    selected_transcript_source: str | None = None,
    chunk_count: int | None = None,
    valid_chunk_count: int | None = None,
    rejected_chunk_count: int | None = None,
    vad_available: bool | None = None,
    detected_speech_seconds: float | None = None,
    speech_ratio: float | None = None,
    vad_rejected: bool | None = None,
    original_audio_path: str | None = None,
    converted_wav_path: str | None = None,
    suspicious_signals: list[str] | None = None,
    protective_signals: list[str] | None = None,
    qc_severity: str | None = None,
    full_qc_reason: str | None = None,
    retry_qc_reason: str | None = None,
    chunk_valid_ratio: float | None = None,
) -> None:
    try:
        db.add_transcription_log(
            **media_metadata(message),
            duration_seconds=duration_seconds,
            detected_language=detected_language,
            transcript=transcript,
            formatted_transcript=formatted_transcript,
            audio_path=audio_path,
            processing_status=status,
            error_message_internal=error_message_internal,
            qc_status=qc_status,
            qc_reason=qc_reason,
            retry_used=retry_used,
            transcript_preview_internal=transcript_preview_internal,
            selected_transcript_source=selected_transcript_source,
            chunk_count=chunk_count,
            valid_chunk_count=valid_chunk_count,
            rejected_chunk_count=rejected_chunk_count,
            vad_available=vad_available,
            detected_speech_seconds=detected_speech_seconds,
            speech_ratio=speech_ratio,
            vad_rejected=vad_rejected,
            media_type=media_type_for(message),
            file_extension=media_suffix(message),
            original_audio_path=original_audio_path,
            converted_wav_path=converted_wav_path,
            suspicious_signals=",".join(suspicious_signals or []),
            protective_signals=",".join(protective_signals or []),
            qc_severity=qc_severity,
            full_qc_reason=full_qc_reason,
            retry_qc_reason=retry_qc_reason,
            chunk_valid_ratio=chunk_valid_ratio,
        )
    except Exception:
        logger.exception("Failed to save transcription audit log")


def preserve_audio_files(
    *,
    telegram_id: int,
    job_id: str,
    input_path: Path,
    wav_path: Path,
) -> str | None:
    if not store_audio_files:
        return None

    destination_dir = STORED_AUDIO_DIR / str(telegram_id) / job_id
    destination_dir.mkdir(parents=True, exist_ok=True)

    for source_path in (input_path, wav_path):
        if source_path.exists():
            destination_path = destination_dir / source_path.name
            try:
                shutil.move(str(source_path), str(destination_path))
            except Exception:
                logger.exception("Failed to preserve audio file: %s", source_path)

    return str(destination_dir)


def stored_audio_paths(audio_path: str | None, input_path: Path, wav_path: Path) -> tuple[str | None, str | None]:
    if not audio_path:
        return None, None
    directory = Path(audio_path)
    return str(directory / input_path.name), str(directory / wav_path.name)


def chunk_ratio(valid_chunk_count: int, chunk_count: int) -> float | None:
    if chunk_count <= 0:
        return None
    return valid_chunk_count / chunk_count


def cleanup_stored_audio(retention_days: int) -> None:
    if retention_days < 1 or not STORED_AUDIO_DIR.exists():
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for path in STORED_AUDIO_DIR.rglob("*"):
        if not path.is_file():
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if modified < cutoff:
            try:
                path.unlink()
                logger.info("Deleted expired stored audio file: %s", path)
            except OSError:
                logger.exception("Failed to delete expired stored audio file: %s", path)

    for directory in sorted(
        [p for p in STORED_AUDIO_DIR.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


def format_log_rows(rows: list[object]) -> str:
    if not rows:
        return ADMIN_NO_LOGS

    lines: list[str] = []
    for row in rows:
        duration = row["duration_seconds"]
        duration_text = "-" if duration is None else f"{float(duration):.1f}s"
        lines.append(
            " | ".join(
                part
                for part in [
                    f"id={row['id']}",
                    f"user={row['telegram_id']}",
                    f"@{row['username'] or '-'}",
                    f"duration={duration_text}",
                    f"lang={row['detected_language'] or '-'}",
                    f"status={row['processing_status'] or '-'}",
                    f"reason={row['qc_reason']}" if "qc_reason" in row.keys() and row["qc_reason"] else None,
                    (
                        f"source={row['selected_transcript_source']}"
                        if "selected_transcript_source" in row.keys() and row["selected_transcript_source"]
                        else None
                    ),
                    format_admin_datetime(row["created_at"]),
                ]
                if part
            )
        )
    return "\n".join(lines)


def format_admin_datetime(raw_value: str | None) -> str:
    if not raw_value:
        return "-"
    try:
        dt = datetime.fromisoformat(raw_value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(get_quota_zone()).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw_value


def parse_admin_int_args(args: str | None, expected_count: int) -> list[int] | None:
    parts = (args or "").split()
    if len(parts) != expected_count:
        return None
    try:
        return [int(part) for part in parts]
    except ValueError:
        return None


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    if await reject_if_banned(message):
        return
    await register_user(message)
    await message.answer(START)


@router.message(Command("balance"))
async def handle_balance(message: Message, command: CommandObject) -> None:
    if await reject_if_banned(message):
        return
    if (command.args or "").strip():
        await message.answer(USAGE_BALANCE)
        return
    await register_user(message)
    await message.answer(balance_text_for_user(message.from_user.id))


@router.message(Command("myid"))
async def handle_myid(message: Message, command: CommandObject) -> None:
    if await reject_if_banned(message):
        return
    if (command.args or "").strip():
        await message.answer(USAGE_MYID)
        return
    await message.answer(f"Telegram ID الخاص بك: {message.from_user.id}")


@router.message(Command("admin"))
async def handle_admin(message: Message, command: CommandObject) -> None:
    if not await require_admin(message):
        return
    if (command.args or "").strip():
        await message.answer(ADMIN_USAGE_ADMIN)
        return
    await message.answer(ADMIN_PANEL)


@router.message(Command("ban"))
async def handle_ban(message: Message, command: CommandObject) -> None:
    if not await require_admin(message):
        return

    args = (command.args or "").strip()
    if not args or " " not in args:
        await message.answer(ADMIN_USAGE_BAN)
        return

    telegram_id_text, reason = args.split(maxsplit=1)
    try:
        telegram_id = int(telegram_id_text)
    except ValueError:
        await message.answer(ADMIN_USAGE_BAN)
        return

    user = db.get_user(telegram_id)
    username = user["username"] if user else None
    db.ban_user(
        telegram_id=telegram_id,
        username=username,
        reason=reason.strip(),
        banned_by=message.from_user.id,
    )
    await message.answer(f"تم حظر المستخدم {telegram_id}.")


@router.message(Command("unban"))
async def handle_unban(message: Message, command: CommandObject) -> None:
    if not await require_admin(message):
        return

    args = (command.args or "").strip()
    try:
        telegram_id = int(args)
    except ValueError:
        await message.answer(ADMIN_USAGE_UNBAN)
        return

    removed = db.unban_user(telegram_id)
    if removed:
        await message.answer(f"تم رفع الحظر عن المستخدم {telegram_id}.")
    else:
        await message.answer("المستخدم غير موجود في قائمة الحظر.")


@router.message(Command("logs"))
async def handle_logs(message: Message, command: CommandObject) -> None:
    if not await require_admin(message):
        return
    if (command.args or "").strip():
        await message.answer(ADMIN_USAGE_LOGS)
        return

    await message.answer(format_log_rows(db.latest_transcription_logs(10)))


@router.message(Command("userlogs"))
async def handle_userlogs(message: Message, command: CommandObject) -> None:
    if not await require_admin(message):
        return

    args = (command.args or "").strip()
    try:
        telegram_id = int(args)
    except ValueError:
        await message.answer(ADMIN_USAGE_USERLOGS)
        return

    await message.answer(format_log_rows(db.latest_user_transcription_logs(telegram_id, 10)))


@router.message(Command("addminutes"))
async def handle_addminutes(message: Message, command: CommandObject) -> None:
    if not await require_admin(message):
        return

    parsed = parse_admin_int_args(command.args, 2)
    if parsed is None:
        await message.answer(ADMIN_USAGE_ADDMINUTES)
        return

    telegram_id, minutes = parsed
    if minutes < 1 or db.get_user(telegram_id) is None:
        await message.answer(ADMIN_USAGE_ADDMINUTES)
        return

    db.add_extra_minutes(telegram_id, minutes)
    await message.answer(f"تمت إضافة {minutes} دقيقة إلى الرصيد الإضافي للمستخدم {telegram_id}.")


@router.message(Command("setbalance"))
async def handle_setbalance(message: Message, command: CommandObject) -> None:
    if not await require_admin(message):
        return

    parsed = parse_admin_int_args(command.args, 2)
    if parsed is None:
        await message.answer(ADMIN_USAGE_SETBALANCE)
        return

    telegram_id, minutes = parsed
    if minutes < 1 or db.get_user(telegram_id) is None:
        await message.answer(ADMIN_USAGE_SETBALANCE)
        return

    db.set_extra_balance_minutes(telegram_id, minutes)
    await message.answer(f"تم تعيين الرصيد الإضافي للمستخدم {telegram_id} إلى {minutes} دقيقة.")


@router.message(Command("user"))
async def handle_user(message: Message, command: CommandObject) -> None:
    if not await require_admin(message):
        return

    parsed = parse_admin_int_args(command.args, 1)
    if parsed is None:
        await message.answer(ADMIN_USAGE_USER)
        return

    telegram_id = parsed[0]
    user = db.get_user(telegram_id)
    if user is None:
        await message.answer("المستخدم غير موجود.")
        return

    info = db.balance_info(telegram_id, daily_free_seconds(), current_quota_date())
    stats = db.user_log_stats(telegram_id)
    ban = db.get_ban(telegram_id)
    await message.answer(
        "\n".join(
            [
                f"username: @{user['username'] or '-'}",
                f"telegram_id: {telegram_id}",
                f"استخدام اليوم: {info['daily_used_seconds'] // 60} دقيقة",
                f"المتبقي اليوم: {info['daily_remaining_seconds'] // 60} دقيقة",
                f"الرصيد الإضافي: {info['extra_balance_seconds'] // 60} دقيقة",
                f"الحظر: {'محظور' if ban else 'غير محظور'}",
                f"إجمالي السجلات: {stats['total_logs'] or 0}",
                f"الناجحة: {stats['successful_logs'] or 0}",
                f"الفاشلة: {stats['failed_logs'] or 0}",
                f"المرفوضة: {stats['rejected_logs'] or 0}",
                f"آخر استخدام: {format_admin_datetime(stats['latest_usage_time'])}",
            ]
        )
    )


@router.message(F.voice | F.audio | F.document | F.video | F.video_note)
async def handle_audio(message: Message, bot: Bot) -> None:
    if await reject_if_banned(message):
        return
    await register_user(message)

    telegram_user_id = message.from_user.id
    bypass_balance = can_bypass_balance(telegram_user_id)
    media = get_media(message)
    if media is None or not is_supported_media(message):
        save_transcription_log(
            message,
            status="rejected",
            error_message_internal="Unsupported file type or extension.",
            qc_status="skipped",
            qc_reason="unsupported_format",
        )
        await message.answer(UNSUPPORTED_FILE_FORMAT)
        return

    lock = user_locks[telegram_user_id]
    if lock.locked():
        await message.answer(USER_ALREADY_PROCESSING)
        return

    suffix = media_suffix(message)
    job_id = uuid4().hex
    input_path = DOWNLOADS_DIR / f"{telegram_user_id}_{job_id}{suffix}"
    wav_path = DOWNLOADS_DIR / f"{telegram_user_id}_{job_id}_16k.wav"
    duration_seconds: float | None = None
    vad_available: bool | None = None
    detected_speech_seconds: float | None = None
    speech_ratio: float | None = None
    vad_rejected = False

    try:
        async with lock:
            await message.answer(RECEIVING_FILE)
            file = await bot.get_file(media.file_id)
            if not file.file_path:
                raise RuntimeError("Telegram did not return a downloadable file path.")
            await bot.download_file(file.file_path, destination=input_path)

            if (message.video or message.video_note) and not await has_audio_stream(input_path):
                save_transcription_log(
                    message,
                    status="rejected",
                    error_message_internal="Video has no audio stream.",
                    qc_status="skipped",
                    qc_reason="no_audio_stream",
                )
                await message.answer(VIDEO_HAS_NO_AUDIO)
                return

            duration_seconds = float(await get_audio_duration_seconds(input_path))
            if duration_seconds > max_audio_duration_seconds:
                audio_path = preserve_audio_files(
                    telegram_id=telegram_user_id,
                    job_id=job_id,
                    input_path=input_path,
                    wav_path=wav_path,
                )
                save_transcription_log(
                    message,
                    status="rejected",
                    duration_seconds=duration_seconds,
                    audio_path=audio_path,
                    error_message_internal="Audio duration exceeds MAX_AUDIO_DURATION_SECONDS.",
                )
                await message.answer(FILE_TOO_LONG_LIMIT)
                return

            balance_info = db.balance_info(telegram_user_id, daily_free_seconds(), current_quota_date())
            remaining_seconds = balance_info["total_available_seconds"]
            if not bypass_balance and duration_seconds > remaining_seconds:
                audio_path = preserve_audio_files(
                    telegram_id=telegram_user_id,
                    job_id=job_id,
                    input_path=input_path,
                    wav_path=wav_path,
                )
                save_transcription_log(
                    message,
                    status="rejected",
                    duration_seconds=duration_seconds,
                    audio_path=audio_path,
                    error_message_internal="Audio duration exceeds user balance.",
                )
                if remaining_seconds <= 0:
                    await message.answer(FREE_QUOTA_ENDED)
                else:
                    await message.answer(INSUFFICIENT_BALANCE_FOR_FILE)
                return

            await message.answer(PROCESSING_STARTED)
            await convert_to_wav_16k_mono(input_path, wav_path)

            if speech_vad_enabled:
                speech_detection = await asyncio.to_thread(
                    detect_speech_in_wav,
                    wav_path,
                    aggressiveness=speech_vad_aggressiveness,
                )
                vad_available = speech_detection.available
                detected_speech_seconds = speech_detection.detected_speech_seconds
                speech_ratio = speech_detection.speech_ratio
                if speech_detection.available and speech_detection_rejects(
                    total_audio_seconds=speech_detection.total_audio_seconds,
                    detected_speech_seconds=speech_detection.detected_speech_seconds,
                    speech_ratio=speech_detection.speech_ratio,
                ):
                    vad_rejected = True
                    audio_path = preserve_audio_files(
                        telegram_id=telegram_user_id,
                        job_id=job_id,
                        input_path=input_path,
                        wav_path=wav_path,
                    )
                    original_audio_path, converted_wav_path = stored_audio_paths(audio_path, input_path, wav_path)
                    save_transcription_log(
                        message,
                        status="rejected",
                        duration_seconds=duration_seconds,
                        audio_path=audio_path,
                        error_message_internal=(
                            "speech_vad_rejected=true; "
                            f"total_audio_seconds={speech_detection.total_audio_seconds:.3f}; "
                            f"detected_speech_seconds={speech_detection.detected_speech_seconds:.3f}; "
                            f"speech_ratio={speech_detection.speech_ratio:.5f}"
                        ),
                        qc_status="failed",
                        qc_reason="no_speech_detected",
                        retry_used=False,
                        vad_available=vad_available,
                        detected_speech_seconds=detected_speech_seconds,
                        speech_ratio=speech_ratio,
                        vad_rejected=vad_rejected,
                        original_audio_path=original_audio_path,
                        converted_wav_path=converted_wav_path,
                        suspicious_signals=["no_speech_detected", "weak_vad_speech_ratio"],
                        protective_signals=[],
                        qc_severity="hard_reject",
                    )
                    await message.answer(NO_CLEAR_SPEECH)
                    return
                if not speech_detection.available:
                    logger.info("Speech VAD skipped reason=%s", speech_detection.reason)

            result = await asyncio.to_thread(transcriber.transcribe_normal, wav_path)
            qc_result = validate_transcription(
                result,
                duration_seconds=duration_seconds,
                settings=current_qc_settings_with_vad(speech_ratio),
            )
            retry_used = False
            first_qc_reason = qc_result.reason
            retry_qc_reason: str | None = None
            qc_status = "passed" if qc_result.passed else "failed"

            if not qc_result.passed and transcript_retry_on_qc_fail:
                logger.info("Transcription QC failed reason=%s. Retrying once without initial_prompt.", qc_result.reason)
                retry_used = True
                retry_result = await asyncio.to_thread(
                    transcriber.transcribe_normal,
                    wav_path,
                    use_initial_prompt=False,
                    condition_on_previous_text=False,
                )
                qc_result = validate_transcription(
                    retry_result,
                    duration_seconds=duration_seconds,
                    settings=current_qc_settings_with_vad(speech_ratio),
                )
                retry_qc_reason = qc_result.reason
                qc_status = "retry_passed" if qc_result.passed else "retry_failed"

            full_qc_status = qc_status
            full_result = qc_result.result or result
            full_qc_passed = qc_result.passed and qc_result.result is not None
            full_qc_reason = first_qc_reason if first_qc_reason else qc_result.reason
            chunk_result: TranscriptionResult | None = None
            chunk_qc_status = "skipped"
            chunk_count = 0
            valid_chunk_count = 0
            rejected_chunk_count = 0

            if should_run_chunk_fallback(
                full_result,
                full_qc_passed,
                full_qc_reason,
                qc_result.qc_severity,
                duration_seconds,
            ):
                logger.info(
                    "Running chunk fallback full_qc_passed=%s full_qc_reason=%s detected_language=%s",
                    full_qc_passed,
                    full_qc_reason,
                    full_result.detected_language,
                )
                (
                    chunk_result,
                    chunk_qc_status,
                    chunk_count,
                    valid_chunk_count,
                    rejected_chunk_count,
                ) = await transcribe_chunks_with_qc(wav_path, duration_seconds=duration_seconds)

            selected_source, selected_result, selected_reason = choose_transcription_result(
                full_result,
                full_qc_passed,
                full_qc_reason,
                chunk_result,
                chunk_result is not None and chunk_qc_status == "passed",
                duration_seconds=duration_seconds,
                chunk_valid_ratio=chunk_ratio(valid_chunk_count, chunk_count),
            )

            if selected_result is None:
                audio_path = preserve_audio_files(
                    telegram_id=telegram_user_id,
                    job_id=job_id,
                    input_path=input_path,
                    wav_path=wav_path,
                )
                original_audio_path, converted_wav_path = stored_audio_paths(audio_path, input_path, wav_path)
                qc_reason = selected_reason or chunk_qc_status or "no_clear_speech_or_prompt_leak"
                save_transcription_log(
                    message,
                    status="rejected",
                    duration_seconds=duration_seconds,
                    detected_language=result.detected_language,
                    transcript=result.text,
                    formatted_transcript=result.formatted_text,
                    audio_path=audio_path,
                    error_message_internal=(
                        f"first_attempt_failed_validation={first_qc_reason}; "
                        f"retry_used={str(retry_used).lower()}; final_status=rejected; "
                        f"qc_reason={qc_reason}; full_pass_qc_status={qc_status}; "
                        f"chunk_pass_qc_status={chunk_qc_status}; selected_transcript_source={selected_source}; "
                        f"chunk_count={chunk_count}; valid_chunk_count={valid_chunk_count}; "
                        f"rejected_chunk_count={rejected_chunk_count}; "
                        f"chunk_valid_ratio={chunk_ratio(valid_chunk_count, chunk_count)}"
                    ),
                    qc_status=chunk_qc_status if selected_source == "chunk" else qc_status,
                    qc_reason=qc_reason,
                    retry_used=retry_used,
                    transcript_preview_internal=transcript_preview(result.text or ""),
                    selected_transcript_source=selected_source,
                    chunk_count=chunk_count,
                    valid_chunk_count=valid_chunk_count,
                    rejected_chunk_count=rejected_chunk_count,
                    vad_available=vad_available,
                    detected_speech_seconds=detected_speech_seconds,
                    speech_ratio=speech_ratio,
                    vad_rejected=vad_rejected,
                    original_audio_path=original_audio_path,
                    converted_wav_path=converted_wav_path,
                    suspicious_signals=qc_result.suspicious_signals,
                    protective_signals=qc_result.protective_signals,
                    qc_severity=qc_result.qc_severity,
                    full_qc_reason=full_qc_reason,
                    retry_qc_reason=retry_qc_reason,
                    chunk_valid_ratio=chunk_ratio(valid_chunk_count, chunk_count),
                )
                await message.answer(NO_CLEAR_SPEECH)
                return

            result = selected_result
            qc_status = "passed" if selected_source == "full" else "chunk_passed"
            transcript = result.text or EMPTY_TRANSCRIPT
            formatted_transcript = result.formatted_text or transcript

            transcript_file = transcript_path_for(message)
            transcript_file.write_text(transcript, encoding="utf-8")
            timestamped_transcript_file = timestamped_transcript_path_for(message)
            timestamped_transcript_file.write_text(formatted_transcript, encoding="utf-8")

            await message.answer(f"{TRANSCRIPTION_COMPLETED}\n\n{transcript_preview(transcript)}")

            await message.answer_document(
                FSInputFile(transcript_file),
                caption=RAW_TRANSCRIPT_CAPTION,
            )
            await message.answer_document(
                FSInputFile(timestamped_transcript_file),
                caption=TIMESTAMPED_TRANSCRIPT_CAPTION,
            )

            audio_path = preserve_audio_files(
                telegram_id=telegram_user_id,
                job_id=job_id,
                input_path=input_path,
                wav_path=wav_path,
            )
            original_audio_path, converted_wav_path = stored_audio_paths(audio_path, input_path, wav_path)
            save_transcription_log(
                message,
                status="success",
                duration_seconds=duration_seconds,
                detected_language=result.detected_language,
                transcript=transcript,
                formatted_transcript=formatted_transcript,
                audio_path=audio_path,
                error_message_internal=(
                    "; ".join(
                        part
                        for part in [
                            "admin_balance_bypass=true" if bypass_balance else None,
                            "retry_used=true" if retry_used else None,
                            f"first_attempt_failed_validation={first_qc_reason}" if first_qc_reason else None,
                            f"full_pass_qc_status={full_qc_status}",
                            f"chunk_pass_qc_status={chunk_qc_status}",
                            f"selected_transcript_source={selected_source}",
                            f"chunk_count={chunk_count}",
                            f"valid_chunk_count={valid_chunk_count}",
                            f"rejected_chunk_count={rejected_chunk_count}",
                            f"chunk_valid_ratio={chunk_ratio(valid_chunk_count, chunk_count)}",
                            "final_status=success",
                        ]
                        if part
                    )
                ),
                qc_status=qc_status,
                qc_reason=first_qc_reason if retry_used else None,
                retry_used=retry_used,
                transcript_preview_internal=transcript_preview(transcript),
                selected_transcript_source=selected_source,
                chunk_count=chunk_count,
                valid_chunk_count=valid_chunk_count,
                rejected_chunk_count=rejected_chunk_count,
                vad_available=vad_available,
                detected_speech_seconds=detected_speech_seconds,
                speech_ratio=speech_ratio,
                vad_rejected=vad_rejected,
                original_audio_path=original_audio_path,
                converted_wav_path=converted_wav_path,
                suspicious_signals=qc_result.suspicious_signals,
                protective_signals=qc_result.protective_signals,
                qc_severity=qc_result.qc_severity,
                full_qc_reason=full_qc_reason,
                retry_qc_reason=retry_qc_reason,
                chunk_valid_ratio=chunk_ratio(valid_chunk_count, chunk_count),
            )

            if not bypass_balance:
                db.deduct_seconds(
                    telegram_user_id,
                    int(duration_seconds),
                    daily_free_seconds(),
                    current_quota_date(),
                )
                await message.answer(balance_text_for_user(telegram_user_id))
            else:
                await message.answer(ADMIN_BYPASS_BALANCE_MESSAGE)

    except GpuSetupError as exc:
        logger.exception(
            "CUDA setup is incomplete. WHISPER_DEVICE=cuda was requested, but required NVIDIA libraries are missing."
        )
        audio_path = preserve_audio_files(
            telegram_id=telegram_user_id,
            job_id=job_id,
            input_path=input_path,
            wav_path=wav_path,
        )
        save_transcription_log(
            message,
            status="failed",
            duration_seconds=duration_seconds,
            audio_path=audio_path,
            error_message_internal=str(exc),
        )
        await message.answer(gpu_error_message_for_user(telegram_user_id))
    except Exception as exc:
        logger.exception("Audio processing failed")
        audio_path = preserve_audio_files(
            telegram_id=telegram_user_id,
            job_id=job_id,
            input_path=input_path,
            wav_path=wav_path,
        )
        save_transcription_log(
            message,
            status="failed",
            duration_seconds=duration_seconds,
            audio_path=audio_path,
            error_message_internal=str(exc),
        )
        await message.answer(ERROR_OCCURRED)
    finally:
        if not store_audio_files:
            for path in (input_path, wav_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove temporary file: %s", path)


@router.message()
async def handle_unsupported(message: Message) -> None:
    if await reject_if_banned(message):
        return
    await register_user(message)
    if await answer_malformed_known_command(message):
        return
    save_transcription_log(
        message,
        status="rejected",
        error_message_internal="Unsupported message type.",
        qc_status="skipped",
        qc_reason="unsupported_format",
    )
    await message.answer(UNSUPPORTED_FILE_FORMAT)


async def main() -> None:
    ensure_directories()
    ensure_ffmpeg_available()

    settings = load_settings()
    database = Database(DATABASE_PATH)
    database.init()

    global db, transcriber, admin_telegram_ids, admin_bypass_balance
    global store_audio_files, audio_retention_days, max_audio_duration_seconds
    global daily_free_minutes, daily_quota_timezone, transcript_qc_enabled
    global transcript_retry_on_qc_fail, min_meaningful_words
    global prompt_leak_rejection, repetition_rejection
    global speech_vad_enabled, speech_vad_aggressiveness
    global min_speech_seconds, min_speech_ratio
    global allowed_transcript_languages, mixed_language_mode
    global chunk_transcription_enabled, chunk_fallback_always
    global chunk_seconds, chunk_overlap_seconds, chunk_min_seconds
    global mixed_language_protection, language_rejection_soft
    db = database
    admin_telegram_ids = settings.admin_telegram_ids
    admin_bypass_balance = settings.admin_bypass_balance
    store_audio_files = settings.store_audio_files
    audio_retention_days = settings.audio_retention_days
    max_audio_duration_seconds = settings.max_audio_duration_seconds
    daily_free_minutes = settings.daily_free_minutes
    daily_quota_timezone = settings.daily_quota_timezone
    transcript_qc_enabled = settings.transcript_qc_enabled
    transcript_retry_on_qc_fail = settings.transcript_retry_on_qc_fail
    min_meaningful_words = settings.min_meaningful_words
    prompt_leak_rejection = settings.prompt_leak_rejection
    repetition_rejection = settings.repetition_rejection
    speech_vad_enabled = settings.speech_vad_enabled
    speech_vad_aggressiveness = settings.speech_vad_aggressiveness
    min_speech_seconds = settings.min_speech_seconds
    min_speech_ratio = settings.min_speech_ratio
    allowed_transcript_languages = settings.allowed_transcript_languages
    mixed_language_mode = settings.mixed_language_mode
    chunk_transcription_enabled = settings.chunk_transcription_enabled
    chunk_fallback_always = settings.chunk_fallback_always
    chunk_seconds = settings.chunk_seconds
    chunk_overlap_seconds = settings.chunk_overlap_seconds
    chunk_min_seconds = settings.chunk_min_seconds
    mixed_language_protection = settings.mixed_language_protection
    language_rejection_soft = settings.language_rejection_soft
    cleanup_stored_audio(audio_retention_days)

    transcriber = LocalWhisperTranscriber(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        initial_prompt=settings.whisper_initial_prompt,
        vad_filter=settings.whisper_vad_filter,
        beam_size=settings.whisper_beam_size,
        best_of=settings.whisper_best_of,
        no_speech_threshold=settings.whisper_no_speech_threshold,
        condition_on_previous_text=settings.whisper_condition_on_previous_text,
    )

    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    logger.info(
        "Bot is starting with model=%s device=%s compute_type=%s store_audio_files=%s",
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
        settings.store_audio_files,
    )
    logger.info("Speech VAD available: %s", is_speech_vad_available())
    logger.info("Mixed language mode: %s", mixed_language_mode)
    logger.info("Chunk transcription enabled: %s", chunk_transcription_enabled)
    logger.info("Chunk fallback always: %s", chunk_fallback_always)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
