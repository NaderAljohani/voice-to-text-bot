# Operations

Run a single polling process per bot token. SQLite stores quotas and records
under `data/bot.sqlite3`. Each user has a process-local lock; inference calls
also share a lock around the model. Requests from different users can wait for
inference, but there is no persistent queue, global backlog limit, or recovery
of in-flight jobs after restart.

## Configuration

The example sets a 10-minute daily allowance, a 30-minute per-file duration
limit, and the `Asia/Riyadh` quota day. Administrators can add extra minutes.
The legacy five-minute trial constant is retained in the source, but active
quota accounting uses the daily allowance.

The code defaults to the original `large-v3` CUDA configuration if model-related
environment values are absent. Copy the complete `.env.example` for the CPU
starting configuration. CPU latency depends on model size and hardware.

For NVIDIA inference, select `WHISPER_DEVICE=cuda`, an appropriate model, and
`WHISPER_COMPUTE_TYPE=int8_float16`. Install the runtime libraries required by
the installed CTranslate2 version. Follow the upstream
[faster-whisper installation guide](https://github.com/SYSTRAN/faster-whisper#requirements).
`cuda_dll_paths.py` discovers packaged NVIDIA DLL folders on Windows.

FFmpeg and ffprobe must be available as system commands even though
faster-whisper also depends on PyAV. Video files need an audio track.
The [Telegram getFile API](https://core.telegram.org/bots/api#getfile) limits
cloud downloads separately from this bot's duration limit.

## Commands

Users can run `/start`, `/balance`, and `/myid`. Set `ADMIN_TELEGRAM_IDS` to a
comma-separated list of operator IDs to enable administration. Leave
`ADMIN_BYPASS_BALANCE=false` for ordinary quota enforcement during testing.

Administrator commands are `/admin`, `/logs`, `/userlogs`, `/user`,
`/ban`, `/unban`, `/addminutes`, and `/setbalance`. Sending a command without
arguments returns its usage. Additional balance is manually managed; payment
processing and required-channel verification are unimplemented.

## Dashboard and retention

`dashboard_server.py` binds to `127.0.0.1:8765`, reads the same database, and can
play retained audio. It has no authentication. Keep it local; do not expose it
through a public tunnel or reverse proxy without adding access control.
`view_database.py` can also produce a local `database_report.html` export.

`STORE_AUDIO_FILES=false` removes temporary audio at the end of processing.
When enabled for troubleshooting, originals and converted audio are moved to
`data/stored_audio`. `AUDIO_RETENTION_DAYS` is applied at bot startup, not by a
background schedule. Text files and database records have no automatic expiry.

Quota is deducted after sending both transcript documents. Telegram delivery
and SQLite updates are separate operations: a crash between them can deliver a
result without charging quota. Reprocessing is not idempotent across restarts.

## Scope of the release cleanup

The publication copy fixes WAV input/output path collisions, moves the first
processing acknowledgement inside the user lock, serializes inference, avoids
logging transcript contents at INFO level, and includes timezone data for
Windows. Original project copies are preserved separately.
