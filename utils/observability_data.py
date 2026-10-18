"""Read-only observability snapshots. Unknown values never become successful calls."""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
import pandas as pd
from utils.client_dashboard_data import load_history
from utils.global_dashboard_data import portfolio_kpis, daily_summary
from utils.audit import redact_details


def source_info(path):
    try:
        path = Path(path)
        stat = path.stat()
        if not path.is_file():
            raise OSError()
        return dict(state='empty' if stat.st_size == 0 else 'present', size=stat.st_size,
                    modified=pd.Timestamp(stat.st_mtime, unit='s', tz='UTC').isoformat())
    except FileNotFoundError:
        return dict(state='missing', size=None, modified=None)
    except OSError:
        return dict(state='unavailable', size=None, modified=None)


def prediction_snapshot(path):
    source = source_info(path)
    result = dict(source=source, frame=pd.DataFrame(), kpis=None, daily=pd.DataFrame(), warnings=[])
    if source['state'] in {'missing', 'unavailable'}:
        return result
    try:
        from utils.prediction_logs import COLUMNS
        raw = Path(path).read_bytes()
        if raw:
            try:
                text = raw.decode('utf-8-sig')
            except UnicodeDecodeError:
                text = raw.decode('latin-1')
            header = next(csv.reader(io.StringIO(text, newline=''), strict=True))
            if not set(COLUMNS[:7]).issubset(header):
                raise ValueError()
        frame = load_history(path)
        result.update(frame=frame, kpis=portfolio_kpis(frame), daily=daily_summary(frame))
        result['source']['state'] = 'empty' if frame.empty else 'available'
        if '_history_warning' in frame:
            result['warnings'] = list(frame['_history_warning'].dropna().unique())
    except Exception:
        result['source']['state'] = 'invalid'
        result['warnings'] = ['Journal de predictions illisible ou schema invalide ; aucune ligne ignoree ou reecrite.']
    return result


def csv_snapshot(path, required_columns=()):
    source = source_info(path)
    result = dict(source=source, frame=pd.DataFrame(), warnings=[])
    if source['state'] != 'present':
        return result
    try:
        raw = Path(path).read_bytes()
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw.decode('latin-1')
            result['warnings'].append('Encodage latin-1 detecte.')
        rows = csv.reader(io.StringIO(text, newline=''), strict=True)
        header = next(rows)
        if not header or len(set(header)) != len(header) or any(not name.strip() for name in header):
            raise ValueError()
        if not set(required_columns).issubset(header):
            raise ValueError()
        records = []
        for values in rows:
            if not values:
                continue
            if len(values) != len(header):
                raise ValueError()
            records.append(values)
        result['frame'] = pd.DataFrame(records, columns=header)
        result['source']['state'] = 'available' if records else 'empty'
    except Exception:
        result['source']['state'] = 'invalid'
        result['warnings'].append('CSV illisible ou schema invalide ; aucune ligne ignoree ou reecrite.')
    return result


def agent_statistics(frame):
    def known_success(value):
        value = str(value).strip().lower()
        return True if value in {'true', '1'} else False if value in {'false', '0'} else None
    success = frame['success'].map(known_success) if 'success' in frame else pd.Series(None, index=frame.index, dtype=object)
    known = int(success.notna().sum())
    tools = frame['tool'].str.strip() if 'tool' in frame else pd.Series('', index=frame.index, dtype=str)
    # Redact names before they reach chart legends or metric labels.
    tools = pd.Series([str(value) for value in redact_details(tools.tolist())], index=tools.index, dtype=str)
    counts = tools[tools != ''].value_counts()
    timestamp = pd.to_datetime(frame['timestamp'], utc=True, format='mixed', errors='coerce') if 'timestamp' in frame else pd.Series(pd.NaT, index=frame.index, dtype='datetime64[ns, UTC]')
    valid = timestamp.notna() & tools.ne('')
    grouped = pd.DataFrame({'day': timestamp[valid].dt.floor('D'), 'tool': tools[valid]})
    daily = grouped.groupby(['day', 'tool']).size().unstack(fill_value=0) if not grouped.empty else pd.DataFrame()
    return dict(total=len(frame), known_success=known, unknown_success=len(frame)-known,
                success_rate=int(success.eq(True).sum())/known if known else None,
                tool_counts=counts.to_dict(), most_used=str(counts.index[0]) if not counts.empty else None,
                undated=int(timestamp.isna().sum()), daily=daily)


def safe_frame(frame, limit=300):
    """Preserve numeric values in metrics, redact display records separately."""
    records = redact_details(frame.head(limit).to_dict('records'))
    for record in records:
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                record[key] = json.dumps(value, ensure_ascii=False, default=str)
    return pd.DataFrame(records)
