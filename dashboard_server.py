import html
import json
import mimetypes
import os
import sqlite3
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from config import DATABASE_PATH
from database import Database


HOST = "127.0.0.1"
PORT = 8765
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(DATABASE_PATH)
SAFE_SETTING_KEYS = {
    "STORE_AUDIO_FILES",
    "AUDIO_RETENTION_DAYS",
    "TRANSCRIPT_QC_ENABLED",
    "SPEECH_VAD_ENABLED",
    "MIXED_LANGUAGE_MODE",
    "MIXED_LANGUAGE_PROTECTION",
    "LANGUAGE_REJECTION_SOFT",
    "CHUNK_TRANSCRIPTION_ENABLED",
    "CHUNK_SECONDS",
    "CHUNK_OVERLAP_SECONDS",
    "CHUNK_MIN_SECONDS",
}


def main() -> None:
    Database(DB_PATH).init()
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    url = f"http://{HOST}:{PORT}"
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        print(f"Live dashboard: {url}")
    except Exception:
        pass
    server.serve_forever()


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            self.send_html(render_dashboard())
            return
        if parsed.path == "/api/summary":
            self.send_json(api_summary())
            return
        if parsed.path == "/api/logs":
            self.send_json(api_logs(query))
            return
        if parsed.path.startswith("/api/log/"):
            self.send_json(api_log_detail(parsed.path))
            return
        if parsed.path == "/api/users":
            self.send_json(api_users(query))
            return
        if parsed.path == "/api/rejections":
            self.send_json(api_rejections(query))
            return
        if parsed.path == "/api/banned":
            self.send_json(api_banned())
            return
        if parsed.path == "/api/settings_safe":
            self.send_json(api_settings_safe())
            return
        if parsed.path.startswith("/media/"):
            self.send_media(parsed.path)
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, data: object) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_html(self, text: str) -> None:
        payload = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_media(self, path: str) -> None:
        try:
            log_id = int(path.rsplit("/", 1)[-1])
        except ValueError:
            self.send_error(404)
            return

        media_path = media_path_for_log(log_id)
        if media_path is None:
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(str(media_path))[0] or content_type_for_suffix(media_path.suffix)
        try:
            data = media_path.read_bytes()
        except OSError:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def connect() -> sqlite3.Connection:
    Database(DB_PATH).init()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def api_summary() -> dict[str, object]:
    today = datetime.now(ZoneInfo("Asia/Riyadh")).date().isoformat()
    with connect() as conn:
        users = scalar(conn, "SELECT COUNT(*) FROM users")
        success_today = scalar(
            conn,
            """
            SELECT COUNT(*) FROM transcription_logs
            WHERE processing_status = 'success' AND substr(created_at, 1, 10) = ?
            """,
            (today,),
        )
        rejected_today = scalar(
            conn,
            """
            SELECT COUNT(*) FROM transcription_logs
            WHERE processing_status = 'rejected' AND substr(created_at, 1, 10) = ?
            """,
            (today,),
        )
        minutes_today = scalar(
            conn,
            """
            SELECT COALESCE(SUM(duration_seconds), 0) / 60.0
            FROM transcription_logs
            WHERE processing_status = 'success' AND substr(created_at, 1, 10) = ?
            """,
            (today,),
        )
        banned = scalar(conn, "SELECT COUNT(*) FROM banned_users")
        top_reason_row = conn.execute(
            """
            SELECT qc_reason, COUNT(*) AS count
            FROM transcription_logs
            WHERE qc_reason IS NOT NULL AND qc_reason != ''
            GROUP BY qc_reason
            ORDER BY count DESC
            LIMIT 1
            """
        ).fetchone()
        status_rows = rows(conn, "SELECT processing_status AS name, COUNT(*) AS value FROM transcription_logs GROUP BY processing_status")
        reason_rows = rows(
            conn,
            """
            SELECT qc_reason AS name, COUNT(*) AS value
            FROM transcription_logs
            WHERE qc_reason IS NOT NULL AND qc_reason != ''
            GROUP BY qc_reason
            ORDER BY value DESC
            LIMIT 10
            """,
        )
        hourly_rows = rows(
            conn,
            """
            SELECT substr(created_at, 1, 13) AS name, COUNT(*) AS value
            FROM transcription_logs
            GROUP BY substr(created_at, 1, 13)
            ORDER BY name DESC
            LIMIT 24
            """,
        )
    return {
        "cards": {
            "users": users,
            "success_today": success_today,
            "rejected_today": rejected_today,
            "minutes_today": round(float(minutes_today or 0), 1),
            "top_rejection_reason": top_reason_row["qc_reason"] if top_reason_row else "-",
            "banned": banned,
        },
        "charts": {
            "status": status_rows,
            "reasons": reason_rows,
            "hourly": list(reversed(hourly_rows)),
        },
        "updated_at": datetime.now(ZoneInfo("Asia/Riyadh")).strftime("%H:%M:%S"),
    }


def api_logs(query: dict[str, list[str]]) -> dict[str, object]:
    show_qa = first(query, "qa") == "1"
    filters, params = build_log_filters(query)
    limit = bounded_int(first(query, "limit"), default=50, minimum=1, maximum=300)
    offset = bounded_int(first(query, "offset"), default=0, minimum=0, maximum=10000)
    sql = f"""
        SELECT id, created_at, telegram_id, username, duration_seconds, media_type,
               processing_status, detected_language, qc_reason,
               selected_transcript_source, transcript_preview_internal,
               audio_path, original_audio_path, converted_wav_path,
               file_extension, qc_severity, full_qc_reason, retry_qc_reason,
               chunk_valid_ratio, rejected_chunk_count, retry_used,
               suspicious_signals, protective_signals, vad_available,
               detected_speech_seconds, speech_ratio, vad_rejected,
               chunk_count, valid_chunk_count
        FROM transcription_logs
        {filters}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    with connect() as conn:
        data = [sanitize_log_row(dict(row), show_qa) for row in conn.execute(sql, params).fetchall()]
    return {"items": data, "limit": limit, "offset": offset}


def api_log_detail(path: str) -> dict[str, object]:
    try:
        log_id = int(path.rsplit("/", 1)[-1])
    except ValueError:
        return {"error": "not_found"}
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, created_at, telegram_id, username, duration_seconds, media_type,
                   processing_status, detected_language, qc_reason, qc_status,
                   qc_severity, selected_transcript_source, retry_used,
                   transcript, formatted_transcript, transcript_preview_internal,
                   audio_path, original_audio_path, converted_wav_path, file_extension,
                   suspicious_signals, protective_signals, vad_available,
                   detected_speech_seconds, speech_ratio, vad_rejected,
                   chunk_count, valid_chunk_count, rejected_chunk_count, chunk_valid_ratio,
                   full_qc_reason, retry_qc_reason
            FROM transcription_logs
            WHERE id = ?
            """,
            (log_id,),
        ).fetchone()
    if row is None:
        return {"error": "not_found"}
    item = sanitize_detail_row(dict(row))
    return {"item": item}


def api_rejections(query: dict[str, list[str]]) -> dict[str, object]:
    query = dict(query)
    query["status"] = ["rejected"]
    return api_logs(query)


def api_users(query: dict[str, list[str]]) -> dict[str, object]:
    search_id = first(query, "user_id")
    username = first(query, "username")
    filters = []
    params: list[object] = []
    if search_id:
        filters.append("CAST(u.telegram_id AS TEXT) LIKE ?")
        params.append(f"%{search_id}%")
    if username:
        filters.append("COALESCE(u.username, '') LIKE ?")
        params.append(f"%{username}%")
    where_sql = "WHERE " + " AND ".join(filters) if filters else ""
    sql = f"""
        SELECT u.telegram_id, u.username, u.daily_seconds_used,
               u.paid_seconds_balance, u.daily_quota_date,
               COUNT(t.id) AS total_requests,
               SUM(CASE WHEN t.processing_status = 'success' THEN 1 ELSE 0 END) AS successful_requests,
               SUM(CASE WHEN t.processing_status = 'rejected' THEN 1 ELSE 0 END) AS rejected_requests,
               MAX(t.created_at) AS latest_usage,
               CASE WHEN b.telegram_id IS NULL THEN 0 ELSE 1 END AS banned
        FROM users u
        LEFT JOIN transcription_logs t ON t.telegram_id = u.telegram_id
        LEFT JOIN banned_users b ON b.telegram_id = u.telegram_id
        {where_sql}
        GROUP BY u.telegram_id
        ORDER BY latest_usage DESC
        LIMIT 200
    """
    with connect() as conn:
        items = [dict(row) for row in conn.execute(sql, params).fetchall()]
    return {"items": items}


def api_banned() -> dict[str, object]:
    with connect() as conn:
        return {"items": rows(conn, "SELECT telegram_id, username, reason, banned_at FROM banned_users ORDER BY banned_at DESC")}


def api_settings_safe() -> dict[str, object]:
    env_path = BASE_DIR / ".env"
    settings: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in SAFE_SETTING_KEYS:
                settings[key] = value.strip()
    return {"settings": settings}


def build_log_filters(query: dict[str, list[str]]) -> tuple[str, list[object]]:
    filters = []
    params: list[object] = []
    mapping = {
        "status": "processing_status",
        "user_id": "CAST(telegram_id AS TEXT)",
        "username": "COALESCE(username, '')",
        "qc_reason": "COALESCE(qc_reason, '')",
        "media_type": "COALESCE(media_type, '')",
    }
    for key, column in mapping.items():
        value = first(query, key)
        if not value or value == "all":
            continue
        if key in {"user_id", "username"}:
            filters.append(f"{column} LIKE ?")
            params.append(f"%{value}%")
        else:
            filters.append(f"{column} = ?")
            params.append(value)
    date_from = first(query, "date_from")
    date_to = first(query, "date_to")
    if date_from:
        filters.append("date(created_at) >= date(?)")
        params.append(date_from)
    if date_to:
        filters.append("date(created_at) <= date(?)")
        params.append(date_to)
    where_sql = "WHERE " + " AND ".join(filters) if filters else ""
    return where_sql, params


def sanitize_log_row(row: dict[str, object], show_qa: bool) -> dict[str, object]:
    row["time"] = format_datetime(row.pop("created_at", None))
    row["quota_deducted"] = row.get("processing_status") == "success"
    row["transcript_preview_internal"] = short(row.get("transcript_preview_internal"))
    row["audio_filename"] = filename(row.get("audio_path"))
    row["audio_exists"] = path_exists(row.get("audio_path"))
    row["media_url"] = f"/media/{row.get('id')}" if row["audio_exists"] else ""
    if show_qa:
        row["qa_audio_path"] = row.get("audio_path")
        row["qa_original_audio_path"] = row.get("original_audio_path")
        row["qa_converted_wav_path"] = row.get("converted_wav_path")
    row.pop("audio_path", None)
    row.pop("original_audio_path", None)
    row.pop("converted_wav_path", None)
    return row


def sanitize_detail_row(row: dict[str, object]) -> dict[str, object]:
    row["time"] = format_datetime(row.pop("created_at", None))
    row["audio_filename"] = filename(row.get("audio_path"))
    row["audio_exists"] = path_exists(row.get("audio_path"))
    row["media_url"] = f"/media/{row.get('id')}" if row["audio_exists"] else ""
    row["transcript"] = safe_large_text(row.get("transcript"))
    row["formatted_transcript"] = safe_large_text(row.get("formatted_transcript"))
    row["transcript_preview_internal"] = safe_large_text(row.get("transcript_preview_internal"), 4000)
    row.pop("audio_path", None)
    row.pop("original_audio_path", None)
    row.pop("converted_wav_path", None)
    return row


def media_path_for_log(log_id: int) -> Path | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT audio_path, original_audio_path, converted_wav_path
            FROM transcription_logs
            WHERE id = ?
            """,
            (log_id,),
        ).fetchone()
    if row is None:
        return None
    candidates = [row["converted_wav_path"], row["original_audio_path"], row["audio_path"]]
    for candidate in candidates:
        media_path = safe_stored_media_path(candidate)
        if media_path is not None:
            return media_path
    return None


def safe_stored_media_path(value: object) -> Path | None:
    if not value:
        return None
    try:
        path = Path(str(value)).resolve()
        allowed = (BASE_DIR / "data" / "stored_audio").resolve()
        if allowed not in path.parents and path != allowed:
            return None
        if not path.is_file():
            return None
        if path.suffix.lower() not in {".ogg", ".oga", ".mp3", ".wav", ".m4a", ".mp4", ".webm", ".aac", ".flac"}:
            return None
        return path
    except OSError:
        return None


def content_type_for_suffix(suffix: str) -> str:
    return {
        ".ogg": "audio/ogg",
        ".oga": "audio/ogg",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".webm": "audio/webm",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
    }.get(suffix.lower(), "application/octet-stream")


def render_dashboard() -> str:
    return DASHBOARD_HTML


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> object:
    return conn.execute(sql, params).fetchone()[0]


def rows(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> list[dict[str, object]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def first(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key) or [""]
    return values[0].strip()


def bounded_int(raw_value: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def filename(value: object) -> str:
    if not value:
        return ""
    return Path(str(value)).name


def path_exists(value: object) -> bool:
    return bool(value and Path(str(value)).exists())


def short(value: object, limit: int = 160) -> str:
    text = "" if value is None else str(value)
    return text[:limit] + ("..." if len(text) > limit else "")


def safe_large_text(value: object, limit: int = 12000) -> str:
    text = "" if value is None else str(value)
    return text[:limit] + ("\n\n..." if len(text) > limit else "")


def format_datetime(raw: object) -> str:
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo("Asia/Riyadh")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(raw)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>لوحة مراجعة التفريغ</title>
<style>
:root{--bg:#f5f7fb;--panel:#fff;--line:#dde3ee;--text:#17202a;--muted:#64748b;--brand:#1f2937;--green:#15803d;--orange:#d97706;--red:#dc2626;--blue:#2563eb}
*{box-sizing:border-box} body{font-family:Tahoma,Arial,sans-serif;margin:0;background:var(--bg);color:var(--text)}
header{background:var(--brand);color:#fff;padding:16px 24px;display:flex;justify-content:space-between;align-items:center;gap:12px;position:sticky;top:0;z-index:5}
main{padding:20px;max-width:1680px;margin:auto}.cards{display:grid;grid-template-columns:repeat(6,minmax(145px,1fr));gap:12px}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;box-shadow:0 1px 2px rgba(15,23,42,.05)}.card span{color:var(--muted);font-size:12px}.card b{display:block;font-size:22px;margin-top:6px}.filters{display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:10px;margin:12px 0}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}input,select,button{font:inherit;border:1px solid #cbd5e1;border-radius:6px;padding:8px;background:white}button{cursor:pointer;background:var(--brand);color:#fff;border-color:var(--brand)}button.secondary{background:#fff;color:var(--brand)}button.link{background:#eef2ff;color:#1e3a8a;border-color:#c7d2fe;padding:5px 8px}.table-wrap{overflow:auto;max-height:620px;border:1px solid var(--line);border-radius:8px}table{width:100%;border-collapse:collapse;direction:rtl;font-size:13px;background:white}th,td{border-bottom:1px solid #e5e7eb;padding:9px;vertical-align:top;max-width:280px;overflow-wrap:anywhere;text-align:right}th{background:#f8fafc;position:sticky;top:0;z-index:1}.preview{max-width:340px;white-space:pre-wrap;line-height:1.55}.badge{border-radius:999px;padding:3px 8px;color:#fff;font-size:12px;display:inline-block}.success{background:var(--green)}.rejected{background:var(--orange)}.failed{background:var(--red)}.other,.admin{background:var(--blue)}.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:16px 0}.bar{height:20px;background:var(--blue);border-radius:4px;margin:4px 0;color:white;font-size:12px;text-align:center;min-width:28px}.muted{color:var(--muted)}.qa-line{font-size:12px;color:var(--muted);margin-top:4px}.modal{position:fixed;inset:0;background:rgba(15,23,42,.62);display:none;align-items:center;justify-content:center;padding:20px;z-index:20}.modal.open{display:flex}.modal-card{background:white;border-radius:10px;width:min(1180px,96vw);max-height:92vh;overflow:auto;padding:18px}.modal-head{display:flex;justify-content:space-between;gap:12px;align-items:center;border-bottom:1px solid var(--line);padding-bottom:10px}.compare{display:grid;grid-template-columns:42% 58%;gap:16px;margin-top:14px}.meta{display:grid;grid-template-columns:1fr 1fr;gap:8px}.meta div{background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:8px}.transcript{white-space:pre-wrap;line-height:1.8;background:#0f172a;color:#e5e7eb;border-radius:8px;padding:14px;max-height:560px;overflow:auto;direction:rtl}.audio-box{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fafafa}audio{width:100%;margin-top:8px}.hidden{display:none}@media(max-width:1100px){.cards,.filters,.grid,.compare{grid-template-columns:1fr}.hide-small{display:none}}
</style>
</head>
<body>
<header><h1>لوحة مراجعة التفريغ الصوتي</h1><div><button onclick="refreshAll()">تحديث الآن</button> <span id="updated">آخر تحديث: -</span></div></header>
<main>
<section class="cards" id="cards"></section>
<section class="panel"><h2>الفلاتر</h2><div class="filters"><select id="status"><option value="all">كل الحالات</option><option value="success">ناجح</option><option value="rejected">مرفوض</option><option value="failed">فشل</option></select><input id="user_id" placeholder="رقم المستخدم"><input id="username" placeholder="اسم المستخدم"><input id="date_from" type="date"><input id="date_to" type="date"><input id="qc_reason" placeholder="سبب QC"><input id="media_type" placeholder="نوع الوسيط"></div><div class="toolbar"><label><input type="checkbox" id="qa_toggle"> إظهار معلومات QA</label><label id="path_toggle_wrap" class="hidden"><input type="checkbox" id="path_toggle"> إظهار المسار المحلي</label><button class="secondary" onclick="clearFilters()">مسح الفلاتر</button></div></section>
<section class="grid"><div class="panel"><h3>توزيع الحالات</h3><div id="chart_status"></div></div><div class="panel"><h3>أسباب الرفض</h3><div id="chart_reasons"></div></div><div class="panel"><h3>الطلبات حسب الوقت</h3><div id="chart_hourly"></div></div></section>
<section class="panel"><h2>آخر العمليات</h2><div id="logs"></div></section>
<section class="panel"><h2>المرفوضات</h2><div id="rejections"></div></section>
<section class="panel"><h2>المستخدمون</h2><div id="users"></div></section>
<section class="panel"><h2>المحظورون</h2><div id="banned"></div></section>
<section class="panel"><h2>إعدادات آمنة</h2><div id="settings"></div></section>
</main>
<div class="modal" id="detail_modal"><div class="modal-card"><div class="modal-head"><h2 id="modal_title">تفاصيل العملية</h2><button onclick="closeModal()">إغلاق</button></div><div id="modal_body"></div></div></div>
<script>
const $=id=>document.getElementById(id); let latestRows={};
for (const id of ['status','user_id','username','date_from','date_to','qc_reason','media_type','qa_toggle','path_toggle']) $(id).addEventListener('input', refreshAll);
$('qa_toggle').addEventListener('input',()=>{$('path_toggle_wrap').classList.toggle('hidden',!$('qa_toggle').checked); if(!$('qa_toggle').checked)$('path_toggle').checked=false});
function qs(){let p=new URLSearchParams(); for (const id of ['status','user_id','username','date_from','date_to','qc_reason','media_type']) {let v=$(id).value; if(v) p.set(id,v)} if($('qa_toggle').checked)p.set('qa','1'); return p.toString()}
function clearFilters(){for (const id of ['user_id','username','date_from','date_to','qc_reason','media_type']) $(id).value=''; $('status').value='all'; refreshAll()}
async function get(url){return await (await fetch(url,{cache:'no-store'})).json()}
function esc(v){return String(v??'').replace(/[&<>"']/g,s=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]))}
function badge(s){let c=s==='success'?'success':s==='rejected'?'rejected':s==='failed'?'failed':'other'; return `<span class="badge ${c}">${esc(s||'-')}</span>`}
function shortText(v,n=250){let t=String(v||''); return esc(t.length>n?t.slice(0,n)+'...':t)}
function actions(v,row){let audio=row.audio_exists?`<button class="link" onclick="openDetail(${row.id},true)">تشغيل الصوت</button>`:`<span class="muted">لا يوجد ملف صوت محفوظ لهذه العملية.</span>`; return `${audio} <button class="link" onclick="openDetail(${row.id},false)">عرض المزيد</button>`}
function table(rows, cols){if(!rows.length)return '<p class="muted">غير متوفر</p>'; return '<div class="table-wrap"><table><thead><tr>'+cols.map(c=>`<th>${esc(c[0])}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>{latestRows[r.id]=r; return '<tr>'+cols.map(c=>`<td>${c[2]?c[2](r[c[1]],r):esc(r[c[1]])}</td>`).join('')+'</tr>'}).join('')+'</tbody></table></div>'}
function bars(rows){let m=Math.max(1,...rows.map(r=>r.value||0)); return rows.map(r=>`<div>${esc(r.name||'-')} (${r.value})</div><div class="bar" style="width:${Math.max(5,(r.value||0)/m*100)}%"></div>`).join('')||'<p class="muted">غير متوفر</p>'}
async function refreshAll(){let s=await get('/api/summary'); $('updated').textContent='آخر تحديث: '+s.updated_at; const c=s.cards; $('cards').innerHTML=[['عدد المستخدمين',c.users],['عمليات ناجحة اليوم',c.success_today],['عمليات مرفوضة اليوم',c.rejected_today],['إجمالي الدقائق اليوم',c.minutes_today],['أكثر سبب رفض',c.top_rejection_reason],['عدد المحظورين',c.banned]].map(x=>`<div class="card"><span>${x[0]}</span><b>${esc(x[1])}</b></div>`).join(''); $('chart_status').innerHTML=bars(s.charts.status); $('chart_reasons').innerHTML=bars(s.charts.reasons); $('chart_hourly').innerHTML=bars(s.charts.hourly); let q=qs(); let logs=await get('/api/logs?'+q); let rej=await get('/api/rejections?'+q); let users=await get('/api/users?'+q); let banned=await get('/api/banned'); let settings=await get('/api/settings_safe'); const logCols=[['رقم','id'],['الوقت','time'],['رقم المستخدم','telegram_id'],['اسم المستخدم','username'],['المدة','duration_seconds'],['النوع','media_type'],['الحالة','processing_status',(v)=>badge(v)],['اللغة','detected_language'],['سبب الرفض/الجودة','qc_reason'],['مصدر التفريغ','selected_transcript_source'],['المعاينة','transcript_preview_internal',(v)=>`<div class="preview">${shortText(v)}</div>`],['إجراءات','id',actions]]; if($('qa_toggle').checked){logCols.splice(logCols.length-1,0,['severity','qc_severity'],['retry','retry_used'],['VAD','speech_ratio'],['chunks','chunk_valid_ratio'],['file','audio_filename'],['exists','audio_exists']); if($('path_toggle').checked)logCols.splice(logCols.length-1,0,['path','qa_audio_path'])} $('logs').innerHTML=table(logs.items,logCols); $('rejections').innerHTML=table(rej.items,logCols); $('users').innerHTML=table(users.items,[['telegram_id','telegram_id'],['username','username'],['daily used','daily_seconds_used'],['extra balance','paid_seconds_balance'],['total','total_requests'],['success','successful_requests'],['rejected','rejected_requests'],['banned','banned'],['latest','latest_usage']]); $('banned').innerHTML=table(banned.items,[['telegram_id','telegram_id'],['username','username'],['reason','reason'],['banned_at','banned_at']]); $('settings').innerHTML=table(Object.entries(settings.settings).map(([k,v])=>({key:k,value:v})),[['key','key'],['value','value']])}
async function openDetail(id,play){let data=await get('/api/log/'+id); let item=data.item; if(!item){alert('غير متوفر');return} let transcript=item.formatted_transcript||item.transcript||item.transcript_preview_internal||''; let audio=item.media_url?`<audio controls ${play?'autoplay':''} src="${esc(item.media_url)}"></audio>`:`<p class="muted">لا يوجد ملف صوت محفوظ لهذه العملية.</p>`; let qa=$('qa_toggle').checked; let path=$('path_toggle').checked; $('modal_title').textContent='تفاصيل العملية #'+id; $('modal_body').innerHTML=`<div class="compare"><div><h3>النص</h3><div class="transcript">${esc(transcript)}</div></div><div><div class="audio-box"><h3>الصوت</h3>${audio}</div><div class="meta"><div><b>الحالة</b><br>${badge(item.processing_status)}</div><div><b>سبب QC</b><br>${esc(item.qc_reason||'-')}</div><div><b>المدة</b><br>${esc(item.duration_seconds||'-')}</div><div><b>النوع</b><br>${esc(item.media_type||'-')}</div><div><b>اللغة</b><br>${esc(item.detected_language||'-')}</div><div><b>الملف</b><br>${esc(item.audio_filename||'-')}</div><div><b>مصدر التفريغ</b><br>${esc(item.selected_transcript_source||'-')}</div><div><b>severity</b><br>${esc(item.qc_severity||'-')}</div></div>${qa?`<div class="qa-line"><b>signals:</b> ${esc(item.suspicious_signals||'-')}</div><div class="qa-line"><b>protective:</b> ${esc(item.protective_signals||'-')}</div><div class="qa-line"><b>VAD:</b> ${esc(item.speech_ratio||'-')} | chunks: ${esc(item.chunk_valid_ratio||'-')}</div>`:''}${qa&&path?`<div class="qa-line"><b>المسار المحلي مخفي افتراضياً:</b> لا يتم عرضه من API التفاصيل.</div>`:''}</div></div>`; $('detail_modal').classList.add('open')}
function closeModal(){$('detail_modal').classList.remove('open')}
refreshAll(); setInterval(refreshAll,3000);
</script>
</body></html>"""


DASHBOARD_HTML = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>لوحة متابعة بوت التفريغ الصوتي</title>
<style>
:root{--bg:#0f172a;--panel:#1e293b;--panel2:#111827;--line:rgba(255,255,255,.08);--text:#f8fafc;--muted:#94a3b8;--field:#0b1220;--accent:#38bdf8;--green:#22c55e;--amber:#f59e0b;--red:#ef4444;--blue:#60a5fa;--shadow:0 18px 45px rgba(0,0,0,.28)}
:root[data-theme="light"]{--bg:#f5f7fb;--panel:#fff;--panel2:#f8fafc;--line:#dde3ee;--text:#17202a;--muted:#64748b;--field:#fff;--accent:#0284c7;--shadow:0 12px 28px rgba(15,23,42,.08)}
*{box-sizing:border-box}body{font-family:Tahoma,Arial,sans-serif;margin:0;background:linear-gradient(180deg,var(--bg),#111827 460px);color:var(--text)}button,input,select{font:inherit}button{cursor:pointer;border:0;border-radius:10px;padding:9px 12px;background:var(--accent);color:#06121f;font-weight:700}button.secondary{background:transparent;color:var(--text);border:1px solid var(--line)}button.ghost{background:rgba(255,255,255,.06);color:var(--text);border:1px solid var(--line)}button.link{padding:6px 9px;background:rgba(56,189,248,.14);color:var(--accent);border:1px solid rgba(56,189,248,.25);font-size:12px}input,select{border:1px solid var(--line);border-radius:10px;padding:10px;background:var(--field);color:var(--text);min-width:0}header{position:sticky;top:0;z-index:10;background:rgba(15,23,42,.86);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}:root[data-theme="light"] header{background:rgba(255,255,255,.88)}.head{max-width:1700px;margin:auto;padding:18px 22px;display:flex;justify-content:space-between;gap:16px;align-items:center}.title h1{margin:0 0 6px;font-size:24px}.title p{margin:0;color:var(--muted)}.head-actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.badge-note{font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:6px 10px}.paused{color:#fbbf24}.new-banner{display:none;position:fixed;top:86px;left:50%;transform:translateX(-50%);z-index:30;background:#f59e0b;color:#111827;border-radius:999px;padding:10px 16px;box-shadow:var(--shadow);font-weight:700}.new-banner.show{display:block}main{max-width:1700px;margin:auto;padding:22px}.cards{display:grid;grid-template-columns:repeat(6,minmax(155px,1fr));gap:14px}.card,.panel{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:var(--shadow)}.card span{color:var(--muted);font-size:13px}.card b{display:block;font-size:25px;margin-top:9px}.filters{display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:10px;margin-top:12px}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:16px 0}.bar-row{margin:9px 0}.bar-label{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}.bar{height:9px;background:rgba(56,189,248,.16);border-radius:999px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent);border-radius:999px}.section-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px}.table-wrap{overflow:auto;max-height:660px;border:1px solid var(--line);border-radius:16px;background:var(--panel2)}table{width:100%;border-collapse:separate;border-spacing:0;direction:rtl;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:11px;vertical-align:top;text-align:right;max-width:300px;overflow-wrap:anywhere}th{background:rgba(255,255,255,.045);position:sticky;top:0;z-index:2;color:var(--muted);font-weight:700}tr:hover td{background:rgba(56,189,248,.045)}.preview{max-width:380px;white-space:pre-wrap;line-height:1.65;color:var(--text)}.badge{border-radius:999px;padding:4px 9px;color:#06121f;font-size:12px;display:inline-block;font-weight:700}.success{background:var(--green)}.rejected{background:var(--amber)}.failed{background:var(--red);color:#fff}.other,.admin{background:var(--blue)}.muted{color:var(--muted)}.qa-line{font-size:12px;color:var(--muted);margin-top:6px}.modal{position:fixed;inset:0;background:rgba(2,6,23,.72);display:none;align-items:center;justify-content:center;padding:20px;z-index:40}.modal.open{display:flex}.modal-card{background:var(--panel);border:1px solid var(--line);border-radius:20px;width:min(1220px,96vw);max-height:92vh;overflow:auto;padding:18px;box-shadow:var(--shadow)}.modal-head{display:flex;justify-content:space-between;gap:12px;align-items:center;border-bottom:1px solid var(--line);padding-bottom:12px}.compare{display:grid;grid-template-columns:44% 56%;gap:16px;margin-top:14px}.meta{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}.meta div,.audio-box{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:12px}.transcript{white-space:pre-wrap;line-height:1.85;background:#020617;color:#e5e7eb;border:1px solid var(--line);border-radius:14px;padding:15px;max-height:590px;overflow:auto;direction:rtl}.audio-box audio{width:100%;margin-top:10px}.hidden{display:none}.load-more{display:flex;justify-content:center;margin-top:12px}@media(max-width:1150px){.cards,.filters,.grid,.compare{grid-template-columns:1fr 1fr}.head{align-items:flex-start;flex-direction:column}}@media(max-width:760px){.cards,.filters,.grid,.compare,.meta{grid-template-columns:1fr}main{padding:14px}.title h1{font-size:20px}}
</style>
</head>
<body>
<div id="new_banner" class="new-banner" onclick="acceptNewLogs()">توجد عمليات جديدة — اضغط للتحديث</div>
<header><div class="head"><div class="title"><h1>لوحة متابعة بوت التفريغ الصوتي</h1><p>مراقبة العمليات، الجودة، الرصيد، وتشغيل الملفات المحفوظة محلياً</p></div><div class="head-actions"><button onclick="refreshAll({manual:true})">تحديث الآن</button><button class="secondary" id="auto_btn" onclick="toggleAutoRefresh()">إيقاف التحديث التلقائي</button><button class="ghost" onclick="toggleTheme()">الوضع الفاتح / الوضع الداكن</button><span id="paused_badge" class="badge-note hidden">التحديث التلقائي متوقف</span><span id="updated" class="badge-note">آخر تحديث: -</span></div></div></header>
<main>
<section class="cards" id="cards"></section>
<section class="panel"><div class="section-head"><h2>الفلاتر</h2><button class="secondary" onclick="clearFilters()">مسح الفلاتر</button></div><div class="filters"><select id="status"><option value="all">كل الحالات</option><option value="success">ناجح</option><option value="rejected">مرفوض</option><option value="failed">فشل</option></select><input id="user_id" placeholder="رقم المستخدم"><input id="username" placeholder="اسم المستخدم"><input id="date_from" type="date"><input id="date_to" type="date"><input id="qc_reason" placeholder="سبب QC"><input id="media_type" placeholder="نوع الوسيط"></div><div class="toolbar"><label><input type="checkbox" id="qa_toggle"> إظهار معلومات QA</label><label id="path_toggle_wrap" class="hidden"><input type="checkbox" id="path_toggle"> إظهار المسار المحلي</label></div></section>
<section class="grid"><div class="panel"><h3>توزيع الحالات</h3><div id="chart_status"></div></div><div class="panel"><h3>أسباب الرفض</h3><div id="chart_reasons"></div></div><div class="panel"><h3>الطلبات حسب الوقت</h3><div id="chart_hourly"></div></div></section>
<section class="panel"><div class="section-head"><h2>آخر العمليات</h2><span class="muted" id="logs_count"></span></div><div id="logs"></div><div class="load-more"><button class="secondary" onclick="loadMoreLogs()">تحميل المزيد</button></div></section>
<section class="panel"><h2>المرفوضات</h2><div id="rejections"></div></section>
<section class="panel"><h2>المستخدمون</h2><div id="users"></div></section>
<section class="panel"><h2>المحظورون</h2><div id="banned"></div></section>
<section class="panel"><h2>إعدادات آمنة</h2><div id="settings"></div></section>
</main>
<div class="modal" id="detail_modal"><div class="modal-card"><div class="modal-head"><h2 id="modal_title">تفاصيل العملية</h2><button onclick="closeModal()">إغلاق</button></div><div id="modal_body"></div></div></div>
<script>
const $=id=>document.getElementById(id); let latestRows={}; let logsLimit=50; let lastTopId=0; let autoRefresh=true; let timer=null; let isRefreshing=false;
function initTheme(){let theme=localStorage.getItem('dashboardTheme')||'dark'; document.documentElement.dataset.theme=theme==='light'?'light':'dark'}
function toggleTheme(){let next=document.documentElement.dataset.theme==='light'?'dark':'light'; document.documentElement.dataset.theme=next; localStorage.setItem('dashboardTheme',next)}
initTheme();
for (const id of ['status','user_id','username','date_from','date_to','qc_reason','media_type','qa_toggle','path_toggle']) $(id).addEventListener('input',()=>refreshAll({manual:true}));
$('qa_toggle').addEventListener('input',()=>{$('path_toggle_wrap').classList.toggle('hidden',!$('qa_toggle').checked); if(!$('qa_toggle').checked)$('path_toggle').checked=false});
function qs(extra={}){let p=new URLSearchParams(); for (const id of ['status','user_id','username','date_from','date_to','qc_reason','media_type']) {let v=$(id).value; if(v) p.set(id,v)} if($('qa_toggle').checked)p.set('qa','1'); for(const [k,v] of Object.entries(extra))p.set(k,v); return p.toString()}
function clearFilters(){for (const id of ['user_id','username','date_from','date_to','qc_reason','media_type']) $(id).value=''; $('status').value='all'; logsLimit=50; refreshAll({manual:true})}
function toggleAutoRefresh(){autoRefresh=!autoRefresh; $('auto_btn').textContent=autoRefresh?'إيقاف التحديث التلقائي':'تشغيل التحديث التلقائي'; $('paused_badge').classList.toggle('hidden',autoRefresh)}
function shouldPauseAuto(){let active=document.activeElement; return document.body.classList.contains('modal-open') || (active && ['INPUT','SELECT','TEXTAREA'].includes(active.tagName)) || document.querySelector('audio:not([paused])') || window.scrollY>700}
async function get(url){return await (await fetch(url,{cache:'no-store'})).json()}
function esc(v){return String(v??'').replace(/[&<>"']/g,s=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]))}
function badge(s){let c=s==='success'?'success':s==='rejected'?'rejected':s==='failed'?'failed':'other'; return `<span class="badge ${c}">${esc(s||'-')}</span>`}
function shortText(v,n=250){let t=String(v||''); return esc(t.length>n?t.slice(0,n)+'...':t)}
function actions(v,row){let audio=row.audio_exists?`<button class="link" onclick="openDetail(${row.id},true)">تشغيل الصوت</button>`:`<span class="muted">لا يوجد ملف صوت محفوظ لهذه العملية.</span>`; return `${audio} <button class="link" onclick="openDetail(${row.id},false)">عرض المزيد</button>`}
function table(rows, cols){if(!rows.length)return '<p class="muted">غير متوفر</p>'; return '<div class="table-wrap"><table><thead><tr>'+cols.map(c=>`<th>${esc(c[0])}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>{latestRows[r.id]=r; return '<tr>'+cols.map(c=>`<td>${c[2]?c[2](r[c[1]],r):esc(r[c[1]])}</td>`).join('')+'</tr>'}).join('')+'</tbody></table></div>'}
function bars(rows){let m=Math.max(1,...rows.map(r=>r.value||0)); return rows.map(r=>`<div class="bar-row"><div class="bar-label"><span>${esc(r.name||'-')}</span><b>${r.value}</b></div><div class="bar"><i style="width:${Math.max(5,(r.value||0)/m*100)}%"></i></div></div>`).join('')||'<p class="muted">غير متوفر</p>'}
async function refreshSummary(){let s=await get('/api/summary'); $('updated').textContent='آخر تحديث: '+s.updated_at; const c=s.cards; $('cards').innerHTML=[['👥 المستخدمون',c.users],['✅ الناجحة',c.success_today],['⚠️ المرفوضة',c.rejected_today],['⏱️ الدقائق',c.minutes_today],['🔎 سبب الرفض الأكثر',c.top_rejection_reason],['🚫 المحظورون',c.banned]].map(x=>`<div class="card"><span>${x[0]}</span><b>${esc(x[1])}</b></div>`).join(''); $('chart_status').innerHTML=bars(s.charts.status); $('chart_reasons').innerHTML=bars(s.charts.reasons); $('chart_hourly').innerHTML=bars(s.charts.hourly)}
async function refreshTables(){let q=qs({limit:logsLimit,offset:0}); let logs=await get('/api/logs?'+q); let rej=await get('/api/rejections?'+q); let users=await get('/api/users?'+q); let banned=await get('/api/banned'); let settings=await get('/api/settings_safe'); if(logs.items[0]){ if(lastTopId && logs.items[0].id>lastTopId && window.scrollY>300){$('new_banner').classList.add('show')} lastTopId=Math.max(lastTopId,logs.items[0].id)} const logCols=[['رقم','id'],['الوقت','time'],['رقم المستخدم','telegram_id'],['اسم المستخدم','username'],['المدة','duration_seconds'],['النوع','media_type'],['الحالة','processing_status',(v)=>badge(v)],['اللغة','detected_language'],['سبب الرفض/الجودة','qc_reason'],['مصدر التفريغ','selected_transcript_source'],['المعاينة','transcript_preview_internal',(v)=>`<div class="preview">${shortText(v)}</div>`],['إجراءات','id',actions]]; if($('qa_toggle').checked){logCols.splice(logCols.length-1,0,['severity','qc_severity'],['retry','retry_used'],['VAD','speech_ratio'],['chunks','chunk_valid_ratio'],['file','audio_filename'],['exists','audio_exists']); if($('path_toggle').checked)logCols.splice(logCols.length-1,0,['path','qa_audio_path'])} $('logs').innerHTML=table(logs.items,logCols); $('logs_count').textContent=`عرض ${logs.items.length}`; $('rejections').innerHTML=table(rej.items,logCols); $('users').innerHTML=table(users.items,[['telegram_id','telegram_id'],['username','username'],['daily used','daily_seconds_used'],['extra balance','paid_seconds_balance'],['total','total_requests'],['success','successful_requests'],['rejected','rejected_requests'],['banned','banned'],['latest','latest_usage']]); $('banned').innerHTML=table(banned.items,[['telegram_id','telegram_id'],['username','username'],['reason','reason'],['banned_at','banned_at']]); $('settings').innerHTML=table(Object.entries(settings.settings).map(([k,v])=>({key:k,value:v})),[['key','key'],['value','value']])}
async function checkNewLogs(){let data=await get('/api/logs?'+qs({limit:1,offset:0})); if(data.items[0] && lastTopId && data.items[0].id>lastTopId)$('new_banner').classList.add('show')}
async function refreshAll(opts={}){if(isRefreshing)return; isRefreshing=true; const y=window.scrollY; const preserve=opts.preserve!==false; await refreshSummary(); if(opts.tables!==false) await refreshTables(); if(preserve && !document.body.classList.contains('modal-open')) window.scrollTo(0,y); isRefreshing=false}
async function autoTick(){if(!autoRefresh)return; if(shouldPauseAuto()){await refreshSummary(); await checkNewLogs(); return} await refreshAll({tables:true})}
function loadMoreLogs(){logsLimit+=50; refreshAll({manual:true})}
function acceptNewLogs(){$('new_banner').classList.remove('show'); logsLimit=50; refreshAll({manual:true,preserve:false}).then(()=>window.scrollTo(0,0))}
async function openDetail(id,play){let data=await get('/api/log/'+id); let item=data.item; if(!item){alert('غير متوفر');return} let transcript=item.formatted_transcript||item.transcript||item.transcript_preview_internal||''; let audio=item.media_url?`<audio controls ${play?'autoplay':''} src="${esc(item.media_url)}"></audio>`:`<p class="muted">لا يوجد ملف صوت محفوظ لهذه العملية.</p>`; let qa=$('qa_toggle').checked; $('modal_title').textContent='تفاصيل العملية #'+id; $('modal_body').innerHTML=`<div class="compare"><div><div class="section-head"><h3>النص</h3><button class="secondary" onclick="copyTranscript()">نسخ النص</button></div><div class="transcript" id="detail_transcript">${esc(transcript)}</div></div><div><div class="audio-box"><h3>الصوت</h3>${audio}</div><div class="meta"><div><b>الحالة</b><br>${badge(item.processing_status)}</div><div><b>سبب QC</b><br>${esc(item.qc_reason||'-')}</div><div><b>المدة</b><br>${esc(item.duration_seconds||'-')}</div><div><b>النوع</b><br>${esc(item.media_type||'-')}</div><div><b>اللغة</b><br>${esc(item.detected_language||'-')}</div><div><b>الملف</b><br>${esc(item.audio_filename||'-')}</div><div><b>مصدر التفريغ</b><br>${esc(item.selected_transcript_source||'-')}</div><div><b>severity</b><br>${esc(item.qc_severity||'-')}</div></div>${qa?`<div class="qa-line"><b>signals:</b> ${esc(item.suspicious_signals||'-')}</div><div class="qa-line"><b>protective:</b> ${esc(item.protective_signals||'-')}</div><div class="qa-line"><b>VAD:</b> ${esc(item.speech_ratio||'-')} | chunks: ${esc(item.chunk_valid_ratio||'-')}</div>`:''}</div></div>`; document.body.classList.add('modal-open'); $('detail_modal').classList.add('open')}
function closeModal(){document.body.classList.remove('modal-open'); $('detail_modal').classList.remove('open')}
function copyTranscript(){navigator.clipboard&&navigator.clipboard.writeText($('detail_transcript').innerText)}
refreshAll({manual:true}); timer=setInterval(autoTick,3000);
</script>
</body></html>"""


if __name__ == "__main__":
    main()
