import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    telegram_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    joined_at TEXT NOT NULL,
                    free_seconds_used INTEGER NOT NULL DEFAULT 0,
                    paid_seconds_balance INTEGER NOT NULL DEFAULT 0,
                    daily_quota_date TEXT,
                    daily_seconds_used INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._ensure_column(conn, "users", "daily_quota_date", "TEXT")
            self._ensure_column(conn, "users", "daily_seconds_used", "INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcription_logs (
                    id INTEGER PRIMARY KEY,
                    telegram_id INTEGER NOT NULL,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    chat_id INTEGER,
                    message_id INTEGER,
                    telegram_file_id TEXT,
                    telegram_file_unique_id TEXT,
                    original_file_name TEXT,
                    file_size_bytes INTEGER,
                    duration_seconds REAL,
                    detected_language TEXT,
                    transcript TEXT,
                    formatted_transcript TEXT,
                    audio_path TEXT,
                    processing_status TEXT,
                    error_message_internal TEXT,
                    created_at TEXT
                )
                """
            )
            self._ensure_column(conn, "transcription_logs", "formatted_transcript", "TEXT")
            self._ensure_column(conn, "transcription_logs", "qc_status", "TEXT")
            self._ensure_column(conn, "transcription_logs", "qc_reason", "TEXT")
            self._ensure_column(conn, "transcription_logs", "retry_used", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "transcription_logs", "transcript_preview_internal", "TEXT")
            self._ensure_column(conn, "transcription_logs", "selected_transcript_source", "TEXT")
            self._ensure_column(conn, "transcription_logs", "chunk_count", "INTEGER")
            self._ensure_column(conn, "transcription_logs", "valid_chunk_count", "INTEGER")
            self._ensure_column(conn, "transcription_logs", "rejected_chunk_count", "INTEGER")
            self._ensure_column(conn, "transcription_logs", "vad_available", "INTEGER")
            self._ensure_column(conn, "transcription_logs", "detected_speech_seconds", "REAL")
            self._ensure_column(conn, "transcription_logs", "speech_ratio", "REAL")
            self._ensure_column(conn, "transcription_logs", "vad_rejected", "INTEGER")
            self._ensure_column(conn, "transcription_logs", "media_type", "TEXT")
            self._ensure_column(conn, "transcription_logs", "file_extension", "TEXT")
            self._ensure_column(conn, "transcription_logs", "original_audio_path", "TEXT")
            self._ensure_column(conn, "transcription_logs", "converted_wav_path", "TEXT")
            self._ensure_column(conn, "transcription_logs", "suspicious_signals", "TEXT")
            self._ensure_column(conn, "transcription_logs", "protective_signals", "TEXT")
            self._ensure_column(conn, "transcription_logs", "qc_severity", "TEXT")
            self._ensure_column(conn, "transcription_logs", "full_qc_reason", "TEXT")
            self._ensure_column(conn, "transcription_logs", "retry_qc_reason", "TEXT")
            self._ensure_column(conn, "transcription_logs", "chunk_valid_ratio", "REAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS banned_users (
                    telegram_id INTEGER PRIMARY KEY,
                    username TEXT,
                    reason TEXT,
                    banned_at TEXT,
                    banned_by INTEGER
                )
                """
            )

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def upsert_user(
        self,
        telegram_id: int,
        username: str | None,
        first_name: str | None,
    ) -> sqlite3.Row:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    telegram_id, username, first_name, joined_at,
                    free_seconds_used, paid_seconds_balance
                )
                VALUES (?, ?, ?, ?, 0, 0)
                ON CONFLICT(telegram_id) DO UPDATE SET
                    username = excluded.username,
                    first_name = excluded.first_name
                """,
                (telegram_id, username, first_name, utc_now_iso()),
            )
            return conn.execute(
                "SELECT * FROM users WHERE telegram_id = ?",
                (telegram_id,),
            ).fetchone()

    def get_user(self, telegram_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM users WHERE telegram_id = ?",
                (telegram_id,),
            ).fetchone()

    def get_or_create_user(
        self,
        telegram_id: int,
        username: str | None,
        first_name: str | None,
    ) -> sqlite3.Row:
        user = self.get_user(telegram_id)
        if user is not None:
            return self.upsert_user(telegram_id, username, first_name)
        return self.upsert_user(telegram_id, username, first_name)

    def ensure_daily_quota(self, telegram_id: int, quota_date: str) -> None:
        with self.connect() as conn:
            user = conn.execute(
                "SELECT daily_quota_date FROM users WHERE telegram_id = ?",
                (telegram_id,),
            ).fetchone()
            if user is None:
                return
            if user["daily_quota_date"] != quota_date:
                conn.execute(
                    """
                    UPDATE users
                    SET daily_quota_date = ?, daily_seconds_used = 0
                    WHERE telegram_id = ?
                    """,
                    (quota_date, telegram_id),
                )

    def balance_info(self, telegram_id: int, daily_free_seconds: int, quota_date: str) -> dict[str, int]:
        self.ensure_daily_quota(telegram_id, quota_date)
        user = self.get_user(telegram_id)
        if user is None:
            return {
                "daily_used_seconds": 0,
                "daily_remaining_seconds": daily_free_seconds,
                "extra_balance_seconds": 0,
                "total_available_seconds": daily_free_seconds,
            }

        daily_used = int(user["daily_seconds_used"])
        daily_remaining = max(0, daily_free_seconds - daily_used)
        extra_balance = int(user["paid_seconds_balance"])
        return {
            "daily_used_seconds": daily_used,
            "daily_remaining_seconds": daily_remaining,
            "extra_balance_seconds": extra_balance,
            "total_available_seconds": daily_remaining + extra_balance,
        }

    def remaining_seconds(self, telegram_id: int, daily_free_seconds: int, quota_date: str) -> int:
        return self.balance_info(telegram_id, daily_free_seconds, quota_date)["total_available_seconds"]

    def deduct_seconds(self, telegram_id: int, seconds: int, daily_free_seconds: int, quota_date: str) -> None:
        seconds = max(0, int(seconds))
        if seconds == 0:
            return

        with self.connect() as conn:
            self._ensure_daily_quota_in_connection(conn, telegram_id, quota_date)
            user = conn.execute(
                "SELECT * FROM users WHERE telegram_id = ?",
                (telegram_id,),
            ).fetchone()
            if user is None:
                raise RuntimeError("Cannot deduct balance for unknown user.")

            daily_used = int(user["daily_seconds_used"])
            paid_balance = int(user["paid_seconds_balance"])
            daily_remaining = max(0, daily_free_seconds - daily_used)

            daily_to_use = min(daily_remaining, seconds)
            paid_to_use = seconds - daily_to_use

            if paid_to_use > paid_balance:
                raise RuntimeError("Insufficient balance while deducting seconds.")

            conn.execute(
                """
                UPDATE users
                SET daily_seconds_used = ?, paid_seconds_balance = ?
                WHERE telegram_id = ?
                """,
                (daily_used + daily_to_use, paid_balance - paid_to_use, telegram_id),
            )

    def _ensure_daily_quota_in_connection(
        self,
        conn: sqlite3.Connection,
        telegram_id: int,
        quota_date: str,
    ) -> None:
        user = conn.execute(
            "SELECT daily_quota_date FROM users WHERE telegram_id = ?",
            (telegram_id,),
        ).fetchone()
        if user is not None and user["daily_quota_date"] != quota_date:
            conn.execute(
                """
                UPDATE users
                SET daily_quota_date = ?, daily_seconds_used = 0
                WHERE telegram_id = ?
                """,
                (quota_date, telegram_id),
            )

    def add_extra_minutes(self, telegram_id: int, minutes: int) -> None:
        seconds = max(0, int(minutes)) * 60
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET paid_seconds_balance = paid_seconds_balance + ?
                WHERE telegram_id = ?
                """,
                (seconds, telegram_id),
            )

    def set_extra_balance_minutes(self, telegram_id: int, minutes: int) -> None:
        seconds = max(0, int(minutes)) * 60
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET paid_seconds_balance = ?
                WHERE telegram_id = ?
                """,
                (seconds, telegram_id),
            )

    def add_transcription_log(
        self,
        *,
        telegram_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        chat_id: int | None = None,
        message_id: int | None = None,
        telegram_file_id: str | None = None,
        telegram_file_unique_id: str | None = None,
        original_file_name: str | None = None,
        file_size_bytes: int | None = None,
        duration_seconds: float | None = None,
        detected_language: str | None = None,
        transcript: str | None = None,
        formatted_transcript: str | None = None,
        audio_path: str | None = None,
        processing_status: str | None = None,
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
        media_type: str | None = None,
        file_extension: str | None = None,
        original_audio_path: str | None = None,
        converted_wav_path: str | None = None,
        suspicious_signals: str | None = None,
        protective_signals: str | None = None,
        qc_severity: str | None = None,
        full_qc_reason: str | None = None,
        retry_qc_reason: str | None = None,
        chunk_valid_ratio: float | None = None,
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO transcription_logs (
                    telegram_id, username, first_name, last_name, chat_id, message_id,
                    telegram_file_id, telegram_file_unique_id, original_file_name,
                    file_size_bytes, duration_seconds, detected_language, transcript,
                    formatted_transcript, audio_path, processing_status,
                    error_message_internal, qc_status, qc_reason, retry_used,
                    transcript_preview_internal, selected_transcript_source,
                    chunk_count, valid_chunk_count, rejected_chunk_count,
                    vad_available, detected_speech_seconds, speech_ratio,
                    vad_rejected, media_type, file_extension, original_audio_path,
                    converted_wav_path, suspicious_signals, protective_signals,
                    qc_severity, full_qc_reason, retry_qc_reason, chunk_valid_ratio,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    telegram_id,
                    username,
                    first_name,
                    last_name,
                    chat_id,
                    message_id,
                    telegram_file_id,
                    telegram_file_unique_id,
                    original_file_name,
                    file_size_bytes,
                    duration_seconds,
                    detected_language,
                    transcript,
                    formatted_transcript,
                    audio_path,
                    processing_status,
                    error_message_internal,
                    qc_status,
                    qc_reason,
                    1 if retry_used else 0,
                    transcript_preview_internal,
                    selected_transcript_source,
                    chunk_count,
                    valid_chunk_count,
                    rejected_chunk_count,
                    None if vad_available is None else int(vad_available),
                    detected_speech_seconds,
                    speech_ratio,
                    None if vad_rejected is None else int(vad_rejected),
                    media_type,
                    file_extension,
                    original_audio_path,
                    converted_wav_path,
                    suspicious_signals,
                    protective_signals,
                    qc_severity,
                    full_qc_reason,
                    retry_qc_reason,
                    chunk_valid_ratio,
                    utc_now_iso(),
                ),
            )
            return int(cursor.lastrowid)

    def ban_user(
        self,
        telegram_id: int,
        username: str | None,
        reason: str,
        banned_by: int,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO banned_users (telegram_id, username, reason, banned_at, banned_by)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(telegram_id) DO UPDATE SET
                    username = excluded.username,
                    reason = excluded.reason,
                    banned_at = excluded.banned_at,
                    banned_by = excluded.banned_by
                """,
                (telegram_id, username, reason, utc_now_iso(), banned_by),
            )

    def unban_user(self, telegram_id: int) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM banned_users WHERE telegram_id = ?",
                (telegram_id,),
            )
            return cursor.rowcount > 0

    def is_banned(self, telegram_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT telegram_id FROM banned_users WHERE telegram_id = ?",
                (telegram_id,),
            ).fetchone()
            return row is not None

    def latest_transcription_logs(self, limit: int = 10) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT id, telegram_id, username, duration_seconds,
                           detected_language, processing_status, qc_reason,
                           selected_transcript_source, created_at
                    FROM transcription_logs
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            )

    def latest_user_transcription_logs(self, telegram_id: int, limit: int = 10) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT id, telegram_id, username, duration_seconds,
                           detected_language, processing_status, qc_reason,
                           selected_transcript_source, created_at
                    FROM transcription_logs
                    WHERE telegram_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (telegram_id, limit),
                ).fetchall()
            )

    def user_log_stats(self, telegram_id: int) -> sqlite3.Row:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT
                    COUNT(*) AS total_logs,
                    SUM(CASE WHEN processing_status = 'success' THEN 1 ELSE 0 END) AS successful_logs,
                    SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END) AS failed_logs,
                    SUM(CASE WHEN processing_status = 'rejected' THEN 1 ELSE 0 END) AS rejected_logs,
                    MAX(created_at) AS latest_usage_time
                FROM transcription_logs
                WHERE telegram_id = ?
                """,
                (telegram_id,),
            ).fetchone()

    def get_ban(self, telegram_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM banned_users WHERE telegram_id = ?",
                (telegram_id,),
            ).fetchone()
