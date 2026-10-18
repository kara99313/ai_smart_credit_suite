"""Append-only prediction history, including legacy seven/nine-column CSV files."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path
from threading import Lock

COLUMNS = ["timestamp", "client_id", "prob_default", "pred_label", "threshold",
           "inputs_json", "model_tag", "rating", "decision"]
_LOCK = Lock()


def ensure_prediction_log(path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(COLUMNS)


def _raw_records(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return COLUMNS.copy(), [], "utf-8"
    content = path.read_bytes()
    try:
        text = content.decode("utf-8-sig")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text, encoding = content.decode("latin-1"), "latin-1"
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
        values_list = list(reader)
    except (csv.Error, StopIteration) as error:
        raise ValueError(f"Invalid prediction CSV; no row was discarded: {error}") from error
    if len(set(header)) != len(header):
        raise ValueError("Duplicate prediction-log columns.")
    records = []
    for values in values_list:
        if not values:
            continue
        names = header
        # Older writers appended rating/decision beneath a seven-column header.
        if header == COLUMNS[:7] and len(values) == 9:
            names = COLUMNS
        if len(values) > len(names):
            raise ValueError(f"Unrecognized CSV width at record {len(records) + 1}; no row was discarded.")
        row = dict(zip(names, values))
        for name in COLUMNS:
            row.setdefault(name, None)
        row["_record_number"] = len(records) + 1
        records.append(row)
    return header, records, encoding


def _key(row):
    values = ["" if row.get(name) is None else str(row[name]) for name in COLUMNS]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


def snapshot_path(path):
    return Path(path).with_suffix(".results.jsonl")


def read_prediction_records(path):
    _, records, _ = _raw_records(path)
    archive = snapshot_path(path)
    snapshots, warnings = {}, []
    if archive.exists():
        for number, line in enumerate(archive.read_text(encoding="utf-8").splitlines(), 1):
            try:
                item = json.loads(line)
                if not isinstance(item["result"], dict):
                    raise ValueError("Invalid result snapshot")
                snapshots[(int(item["record_number"]), item["csv_key"])] = item["result"]
            except (ValueError, KeyError, TypeError):
                warnings.append(f"Result snapshot {number} is unreadable; CSV records remain available.")
    for row in records:
        result = snapshots.get((row["_record_number"], _key(row)))
        if result:
            # CSV values remain authoritative where already recorded.
            for name, value in result.items():
                if name not in COLUMNS or row.get(name) in (None, ""):
                    row[name] = value
            row["_result_snapshot"] = True
        if warnings:
            row["_history_warning"] = " ".join(warnings)
    return records


def append_prediction_record(path, row, scoring_result=None):
    """Keep the existing CSV header; archive full results without migrating old rows."""
    path = Path(path)
    with _LOCK:
        ensure_prediction_log(path)
        header, records, encoding = _raw_records(path)
        if not set(COLUMNS[:7]).issubset(header):
            raise ValueError("Prediction-log header lacks required columns.")
        buffer = io.StringIO(newline="")
        csv.writer(buffer).writerow([row.get(name, "") for name in header])
        encoded = buffer.getvalue().encode(encoding)
        persisted = {name: row.get(name) for name in header}
        result = dict(row)
        if scoring_result:
            result.update(scoring_result)
        result["timestamp"] = row["timestamp"]
        result["client_id"] = row["client_id"]
        result["prob_default"] = row["prob_default"]
        archive = {"record_number": len(records) + 1, "csv_key": _key(persisted), "result": result}
        # Validate serialization before appending either file.
        snapshot = json.dumps(archive, ensure_ascii=False, allow_nan=False) + "\n"
        previous = path.read_bytes()
        with path.open("ab") as stream:
            if previous and previous[-1:] not in (b"\n", b"\r"):
                stream.write(b"\n")
            stream.write(encoded)
        with snapshot_path(path).open("a", encoding="utf-8", newline="") as stream:
            stream.write(snapshot)
        return result
