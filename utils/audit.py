"""Local configuration audit. No writes at import or during diagnostics."""
from __future__ import annotations
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from utils.runtime_config import ROOT

AUDIT_PATH = ROOT / 'data/audit_log.csv'
COLUMNS = ['ts', 'actor', 'event', 'details']
_LOCK = Lock()
_MASK = '[masque]'
_SECRET_FIELD = re.compile(r'key|token|secret|pass(?:word)?|authorization|credential', re.I)


def redact_details(value):
    """Redact nested JSON and log text, including credentials not in this process."""
    from utils.health_diagnostics import secrets
    known = sorted(secrets(), key=len, reverse=True)

    def clean(item):
        if isinstance(item, dict):
            return {str(key): _MASK if _SECRET_FIELD.search(str(key)) else clean(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(val) for val in item]
        if not isinstance(item, str):
            return item if item is None or isinstance(item, (bool, int, float)) else '[type non affichable]'
        try:
            parsed = json.loads(item)
            if isinstance(parsed, (dict, list)):
                return clean(parsed)
        except (ValueError, TypeError):
            pass
        for secret in known:
            item = item.replace(secret, _MASK)
        item = re.sub(r'(https?://)[^/\s@]+@', r'\1[masque]@', item)
        item = re.sub(r'(?i)(bearer\s+)[^\s,;"\']+', r'\1[masque]', item)
        item = re.sub(r'(?i)((?:api[_-]?key|key|access[_-]?token|refresh[_-]?token|token|password|passwd|pwd|smtp[_-]?pass|secret|authorization)[\"\']?\s*[=:]\s*)("[^"\r\n]*"|\'[^\'\r\n]*\'|[^\s,;&]+)', r'\1[masque]', item)
        item = re.sub(r'\b(?:gsk_[A-Za-z0-9]{16,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})', _MASK, item)
        return item[:4000]
    return clean(value)


def append_audit(event: str, actor: str = 'session_non_authentifiee', details: dict | None = None) -> None:
    if not isinstance(event, str) or not event.strip():
        raise ValueError('Evenement audit requis.')
    row = [datetime.now(timezone.utc).isoformat(timespec='seconds'), actor, event,
           json.dumps(redact_details(details or {}), ensure_ascii=False, allow_nan=False)]
    with _LOCK:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        exists = AUDIT_PATH.exists() and AUDIT_PATH.stat().st_size > 0
        if exists:
            with AUDIT_PATH.open(encoding='utf-8-sig', newline='') as stream:
                if next(csv.reader(stream), None) != COLUMNS:
                    raise ValueError('Schema audit incompatible ; aucun remplacement effectue.')
        needs_newline = False
        if exists:
            with AUDIT_PATH.open('rb') as stream:
                stream.seek(-1, 2)
                needs_newline = stream.read(1) not in (b'\n', b'\r')
        with AUDIT_PATH.open('a', encoding='utf-8', newline='') as stream:
            if needs_newline:
                stream.write('\n')
            writer = csv.writer(stream)
            if not exists:
                writer.writerow(COLUMNS)
            writer.writerow(row)
