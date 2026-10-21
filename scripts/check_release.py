"""Check the exact Git index without printing credential values."""
import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(rb'\b\d{6,12}:[A-Za-z0-9_-]{30,}\b'),
    re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})'),
    re.compile(rb'(?:gsk_[A-Za-z0-9]{24,}|AIza[A-Za-z0-9_-]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
]


def main():
    result = subprocess.run(['git', 'ls-files', '--stage', '-z'], cwd=ROOT, capture_output=True, check=True)
    issues, count, total = [], 0, 0
    blobs = {}
    for entry in result.stdout.decode().split('\0'):
        if not entry:
            continue
        meta, name = entry.split('\t', 1)
        mode, blob_id, stage = meta.split()
        path = Path(name)
        if mode not in {'100644', '100755'} or stage != '0':
            issues.append(name + ': unsupported Git entry')
            continue
        blob = subprocess.run(['git', 'cat-file', 'blob', blob_id], cwd=ROOT, capture_output=True, check=True).stdout
        blobs[name] = blob
        count += 1
        total += len(blob)
        if path.parts[0] in {'data', 'models', '.venv', '__pycache__'} or path.suffix in {'.wav', '.ogg', '.mp3', '.mp4', '.sqlite3', '.db', '.log', '.zip'}:
            issues.append(name + ': runtime data is included')
        if path.name.startswith('.env') and path.name != '.env.example':
            issues.append(name + ': private environment is included')
        if any(pattern.search(blob) for pattern in PATTERNS):
            issues.append(name + ': credential pattern detected; value omitted')
        if len(blob) > 2 * 1024**2:
            issues.append(name + ': unexpected large file')
        if path.suffix == '.py':
            try:
                ast.parse(blob.decode('utf-8-sig'), filename=name)
            except (SyntaxError, UnicodeError):
                issues.append(name + ': Python syntax check failed')
    if not count:
        issues.append('No staged files found')
    for name in ['README.md', 'README_AR.md']:
        for target in re.findall(r'\]\(([^)]+)\)', blobs.get(name, b'').decode()):
            target = target.split('#')[0]
            if target and '://' not in target and target not in blobs:
                issues.append(name + ': missing documentation link')
    if issues:
        print('\n'.join(issues))
        raise SystemExit(1)
    print(json.dumps({'status': 'passed', 'indexed_files': count, 'bytes': total,
                      'credential_patterns': 'none detected', 'runtime_files': 'excluded'}))


if __name__ == '__main__':
    main()
