# Test scope

Run `python -m pytest -q` after installing `requirements-lock.txt`.
Tests use temporary databases, generated audio, mocked Telegram delivery, and
mocked model output. They never load the original `.env` or live user data.

Coverage targets daily quota rollover and paid-balance spillover, insufficient
balance, bans, media validation, transcript QC, chunk timing, model serialization,
WAV input conversion paths, and quota behavior after successful or failed sends.
FFmpeg integration checks skip locally when the system commands are absent.
CI installs FFmpeg and Python dependencies to run the full suite on Windows and
Linux. Subprocess tests also check timeout and cancellation cleanup.

No test downloads a Whisper model, contacts Telegram, or measures GPU inference.
The suite provides no word-error-rate or dialect benchmark. Before deploying,
the operator should test consented recordings on the intended hardware and
review both accepted and rejected transcripts.
