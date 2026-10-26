"""Download pinned source bytes; no third-party dependencies or source execution."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def verify(data, entry):
    if len(data) != entry['expected_bytes']:
        raise ValueError('byte count mismatch')
    digest = hashlib.sha256(data).hexdigest()
    if entry.get('git_blob_sha1'):
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != entry['git_blob_sha1']:
            raise ValueError('Git blob mismatch')
    if entry.get('expected_sha256') and digest != entry['expected_sha256']:
        raise ValueError('SHA-256 mismatch; remote snapshot changed')
    return digest


def main():
    entries = json.loads((ROOT / 'manifests/source-lock.json').read_text())
    old = ROOT / 'manifests/sources.json'
    prior = {e['local_path']: e for e in json.loads(old.read_text())} if old.exists() else {}
    results = []
    for e in entries:
        e = dict(e)
        path = ROOT / e['local_path']
        try:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name + '.part')
                subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                                '--retry', '3', '--retry-delay', '3', '--max-time', '300',
                                e['url'], '-o', str(temp)], check=True)
                verify(temp.read_bytes(), e)
                temp.replace(path)
                prior[e['local_path']] = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}
            data = path.read_bytes()
            e.update(status='ok', bytes=len(data), sha256=verify(data, e),
                     retrieved_at_utc=prior.get(e['local_path'], {}).get('retrieved_at_utc'),
                     verified_at_utc=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            e.update(status='failed', error=str(exc))
        results.append(e)
        old.write_text(json.dumps(results, indent=2) + '\n')
        print(e['status'], e['repository'], e['path'], flush=True)
    write_inventory(results)
    return int(any(e['status'] != 'ok' for e in results))


def write_inventory(results):
    unique = {e['local_path']: e for e in results}
    lines = ['# Preserved files', '',
             'Each path below is one local byte sequence. All source-location aliases, URLs, commit IDs, retrieval timestamps and SHA-256 hashes are in [sources.json](../manifests/sources.json).', '',
             '| Local file | Bytes | Status |', '|---|---:|---|']
    for path, e in sorted(unique.items()):
        lines.append(f"| `{path}` | {e.get('bytes', '—')} | {e['status']} |")
    lines.extend(['', f"{len(results)} source-location entries; {len(unique)} local files; "
                  f"{sum(e.get('bytes',0) for e in unique.values()):,} retrieved bytes.", ''])
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports/acquired-files.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    raise SystemExit(main())
