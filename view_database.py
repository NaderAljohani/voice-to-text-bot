import html
import os
import sqlite3
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "database_report.html"


def main() -> None:
    db_path = find_database()
    html_report = build_report(db_path)
    REPORT_PATH.write_text(html_report, encoding="utf-8")
    webbrowser.open(REPORT_PATH.resolve().as_uri())
    print(f"Report written to: {REPORT_PATH}")
    print(f"Database: {db_path if db_path else 'not found'}")


def find_database() -> Path | None:
    try:
        from config import DATABASE_PATH

        configured = Path(DATABASE_PATH)
        if configured.exists():
            return configured
    except Exception:
        pass

    candidates: list[Path] = []
    for pattern in ("*.sqlite3", "*.sqlite", "*.db"):
        candidates.extend(path for path in BASE_DIR.rglob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_report(db_path: Path | None) -> str:
    generated_at = format_datetime(datetime.now(timezone.utc).isoformat())
    if db_path is None:
        body = section("Summary", paragraph("غير متوفر: لم يتم العثور على قاعدة بيانات."))
    else:
        try:
            from database import Database

            Database(db_path).init()
        except Exception:
            pass
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            body = "\n".join(
                [
                    section(
                        "Summary",
                        paragraph(f"المسار: {escape(str(db_path))}")
                        + paragraph(f"تاريخ التقرير: {escape(generated_at)}")
                        + summary_table(conn),
                    ),
                    section("Users", render_query(conn, "SELECT * FROM users ORDER BY joined_at DESC LIMIT 200")),
                    section(
                        "Latest transcription logs",
                        render_query(
                            conn,
                            """
                            SELECT id, telegram_id, username, duration_seconds, detected_language,
                                   processing_status, qc_status, qc_reason, retry_used,
                                   selected_transcript_source, chunk_count, valid_chunk_count,
                                   rejected_chunk_count, vad_available, detected_speech_seconds,
                                   speech_ratio, vad_rejected, qc_severity, full_qc_reason,
                                   retry_qc_reason, chunk_valid_ratio, transcript_preview_internal,
                                   audio_path, created_at
                            FROM transcription_logs
                            ORDER BY id DESC
                            LIMIT 200
                            """,
                        ),
                    ),
                    section(
                        "Rejected logs",
                        render_query(
                            conn,
                            """
                            SELECT id, telegram_id, username, duration_seconds, processing_status,
                                   qc_status, qc_reason, retry_used, audio_path,
                                   selected_transcript_source, chunk_count, valid_chunk_count,
                                   rejected_chunk_count, vad_available, detected_speech_seconds,
                                   speech_ratio, vad_rejected, qc_severity, full_qc_reason,
                                   retry_qc_reason, chunk_valid_ratio, error_message_internal, created_at
                            FROM transcription_logs
                            WHERE processing_status = 'rejected'
                            ORDER BY id DESC
                            LIMIT 100
                            """,
                        ),
                    ),
                    section("Banned users", render_query(conn, "SELECT * FROM banned_users ORDER BY banned_at DESC")),
                    section(
                        "Daily quota usage",
                        render_query(
                            conn,
                            """
                            SELECT telegram_id, username, daily_quota_date, daily_seconds_used
                            FROM users
                            ORDER BY daily_seconds_used DESC
                            LIMIT 200
                            """,
                        ),
                    ),
                    section(
                        "Extra balances",
                        render_query(
                            conn,
                            """
                            SELECT telegram_id, username, paid_seconds_balance
                            FROM users
                            ORDER BY paid_seconds_balance DESC
                            LIMIT 200
                            """,
                        ),
                    ),
                    section(
                        "QC rejected reasons",
                        render_query(
                            conn,
                            """
                            SELECT qc_reason, COUNT(*) AS count
                            FROM transcription_logs
                            WHERE qc_reason IS NOT NULL AND qc_reason != ''
                            GROUP BY qc_reason
                            ORDER BY count DESC
                            """,
                        ),
                    ),
                    section(
                        "Stored audio file paths",
                        render_query(
                            conn,
                            """
                            SELECT id, telegram_id, username, audio_path, processing_status,
                                   qc_reason, created_at
                            FROM transcription_logs
                            WHERE audio_path IS NOT NULL AND audio_path != ''
                            ORDER BY id DESC
                            LIMIT 200
                            """,
                        ),
                    ),
                ]
            )

    return f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="utf-8">
  <title>تقرير قاعدة بيانات البوت</title>
  <style>
    body {{ font-family: Tahoma, Arial, sans-serif; margin: 24px; background: #f7f7f7; color: #222; }}
    h1, h2 {{ color: #111; }}
    section {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px 0; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; direction: ltr; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; max-width: 360px; overflow-wrap: anywhere; }}
    th {{ background: #efefef; }}
    p {{ margin: 8px 0; }}
    .empty {{ color: #777; }}
  </style>
</head>
<body>
  <h1>تقرير قاعدة بيانات البوت</h1>
  {body}
</body>
</html>
"""


def summary_table(conn: sqlite3.Connection) -> str:
    rows = []
    for table_name in ("users", "transcription_logs", "banned_users"):
        if not table_exists(conn, table_name):
            rows.append({"table": table_name, "count": "غير متوفر"})
            continue
        count = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()["count"]
        rows.append({"table": table_name, "count": count})
    return render_rows(rows)


def render_query(conn: sqlite3.Connection, query: str) -> str:
    try:
        rows = [dict(row) for row in conn.execute(query).fetchall()]
    except sqlite3.Error:
        return paragraph("غير متوفر", css_class="empty")
    return render_rows(rows)


def render_rows(rows: list[dict[str, object]]) -> str:
    if not rows:
        return paragraph("غير متوفر", css_class="empty")
    columns = list(rows[0].keys())
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{format_cell(row.get(column))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def format_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if looks_like_datetime(text):
        text = format_datetime(text)
    return escape(text)


def looks_like_datetime(text: str) -> bool:
    return "T" in text and len(text) >= 19


def format_datetime(raw_value: str) -> str:
    try:
        value = raw_value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo("Asia/Riyadh")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw_value


def section(title: str, content: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{content}</section>"


def paragraph(text: str, css_class: str | None = None) -> str:
    class_attr = f' class="{escape(css_class)}"' if css_class else ""
    return f"<p{class_attr}>{text}</p>"


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
