# Contributing

Keep credentials and runtime files out of changes. Use synthetic or consented
test inputs and avoid transcripts containing personal information.

Install `requirements-lock.txt`, run `python scripts/check_release.py`, and run
`python -m pytest -q` before submitting a change. Changes to quality thresholds
should include examples of intended acceptance and rejection behavior.
