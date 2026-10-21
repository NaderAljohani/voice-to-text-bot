# Runtime data

Telegram transports submitted media and replies. Transcription runs on the
operator's machine after download. Model weights may be downloaded from
Hugging Face on first use; the application does not send audio to a paid
transcription endpoint.

SQLite records include Telegram identifiers, names, file metadata, transcript
text, timestamps, quality-check results, and internal errors. Transcript text
files are retained under `data/transcripts`. This also applies when optional
audio retention is disabled. Audio cleanup does not delete text or database rows.

Tell users about this retention before accepting their recordings. The operator
must manage deletion, backups, filesystem permissions, and access to the local
dashboard. The repository does not contain consent or data-deletion workflows.

Keep `.env`, `data/`, log files, database exports, and model caches outside Git.
The example environment contains empty credentials and no real administrator IDs.
