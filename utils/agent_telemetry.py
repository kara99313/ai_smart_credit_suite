"""Content-free Agent lifecycle snapshots, isolated from prediction telemetry."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

STORE_PATH = Path(__file__).resolve().parents[1] / 'logs' / 'agent_telemetry.sqlite3'
RETENTION_DAYS = 30
TOOLS = {'predict_credit', 'rag_retrieve', 'search_web', 'build_report',
         'now', 'client_context'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def model_metadata(agent):
    """Inspect only actual chat model configuration; never model dumps or payloads."""
    from langchain_core.language_models.chat_models import BaseChatModel
    steps = getattr(getattr(getattr(agent, 'agent', None), 'runnable', None), 'steps', [])
    if not isinstance(steps, (list, tuple)):
        return None, None
    for step in steps:
        llm = getattr(step, 'bound', step)
        if isinstance(llm, BaseChatModel):
            provider = {'ChatGroq': 'groq', 'ChatOpenAI': 'openai',
                        'ChatOllama': 'ollama', 'ChatAnthropic': 'anthropic'}.get(type(llm).__name__)
            model = getattr(llm, 'model_name', None) or getattr(llm, 'model', None)
            model = model if isinstance(model, str) and re.fullmatch(r'[A-Za-z0-9_./:-]{1,100}', model) else None
            return provider, model
    return None, None


def tool_status(name, output):
    """Classify in memory; persist only a fixed status, never the inspected text."""
    if getattr(output, 'status', None) == 'error':
        return 'error'
    text = getattr(output, 'content', output)
    if not isinstance(text, str):
        return 'unknown'
    if name == 'build_report':
        return 'success' if (re.match(r'(?is)\s*(?:<!doctype html[^>]*>\s*)?<html\b', text)
                             and re.search(r'(?is)</html>\s*$', text)) else 'error'
    # Source bodies may contain words such as "error" without indicating failure.
    source = {'rag_retrieve': 'D', 'search_web': 'W'}.get(name)
    if source and re.match(r'\s*\[' + source + r'\d+\]', text):
        return 'success'
    normalized = ''.join(c for c in unicodedata.normalize('NFKD', text.lower())
                         if not unicodedata.combining(c))
    normalized = normalized.strip()
    # Explicit error envelopes and exact tool failure contracts only, never body words.
    failures = {
        'predict_credit': r"(?:format json invalide pour predict_credit\b|(?:❌\s*)?echec d'appel /api/predict\s*:)",
        'rag_retrieve': r'(?:⚠\ufe0f?\s*)?rag indisponible\s*:',
        'search_web': r'(?:⚠\ufe0f?\s*)?recherche web indisponible\s*:',
    }
    if (re.match(r'(?:error|erreur)\s*:', normalized)
            or (name in failures and re.match(failures[name], normalized))):
        return 'error'
    no_results = {
        'rag_retrieve': ('aucun index rag disponible', 'aucun passage pertinent'),
        'search_web': ('aucun resultat web trouve', 'aucun extrait web exploitable'),
        'client_context': ('aucun contexte client disponible',),
    }
    if not normalized or normalized.startswith(no_results.get(name, ())):
        return 'no_result'
    return 'success'


def persist(table, values):
    """Fixed schema, bounded locking, idempotent upserts and best effort retention."""
    try:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(STORE_PATH, timeout=.05)) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS agent_runs (
                request_id TEXT PRIMARY KEY, schema_version INTEGER, timestamp_utc TEXT,
                duration_ms REAL, status TEXT, attempts INTEGER, provider TEXT, model TEXT,
                tools_started INTEGER, tools_completed INTEGER, tools_errors INTEGER)''')
            db.execute('''CREATE TABLE IF NOT EXISTS agent_attempts (
                request_id TEXT, attempt INTEGER, timestamp_utc TEXT, duration_ms REAL,
                status TEXT, PRIMARY KEY(request_id, attempt))''')
            db.execute('''CREATE TABLE IF NOT EXISTS agent_tools (
                request_id TEXT, attempt INTEGER, tool_id INTEGER, timestamp_utc TEXT,
                tool_name TEXT, duration_ms REAL, status TEXT,
                PRIMARY KEY(request_id, tool_id))''')
            sizes = {'agent_runs': 11, 'agent_attempts': 5, 'agent_tools': 7}
            db.execute(f"INSERT OR REPLACE INTO {table} VALUES ({','.join('?' for _ in range(sizes[table]))})", values)
            cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
            for name in sizes:
                db.execute(f'CREATE INDEX IF NOT EXISTS {name}_timestamp ON {name}(timestamp_utc)')
                db.execute(f'DELETE FROM {name} WHERE timestamp_utc < ?', (cutoff,))
    except Exception:
        pass


class AgentTelemetry:
    def __init__(self, request_id, agent):
        self.request_id, self.timestamp, self.started = request_id, utc(), time.perf_counter()
        self.attempt = 0
        self.tools = {}
        try:
            self.provider, self.model = model_metadata(agent)
        except Exception:
            self.provider, self.model = None, None
        self.run('running')

    def write(self, table, values):
        try:
            persist(table, values)
        except Exception:
            pass

    def run(self, status):
        if status in {'success', 'error'}:
            for run_id, tool in self.tools.items():
                if tool['status'] == 'running':
                    tool['status'] = 'interrupted'
                    self.save_tool(run_id)
        self.write('agent_runs', (self.request_id, 1, self.timestamp,
            max(0., (time.perf_counter() - self.started) * 1000), status, self.attempt,
            self.provider, self.model, len(self.tools),
            sum(t['status'] not in {'running', 'interrupted'} for t in self.tools.values()),
            sum(t['status'] == 'error' for t in self.tools.values())))

    def begin_attempt(self, attempt):
        self.attempt, self.attempt_timestamp, self.attempt_started = attempt, utc(), time.perf_counter()
        self.attempt_status = 'running'
        self.end_attempt('running')
        self.run('running')

    def end_attempt(self, status):
        if self.attempt_status != 'running':
            return
        self.attempt_status = status
        self.write('agent_attempts', (self.request_id, self.attempt, self.attempt_timestamp,
            max(0., (time.perf_counter() - self.attempt_started) * 1000), status))

    def start_tool(self, run_id, name):
        if run_id in self.tools:
            return
        self.tools[run_id] = dict(id=len(self.tools) + 1, attempt=self.attempt,
            timestamp=utc(), started=time.perf_counter(), name=name if name in TOOLS else 'unknown', status='running')
        self.save_tool(run_id)
        self.run('running')

    def save_tool(self, run_id):
        t = self.tools[run_id]
        self.write('agent_tools', (self.request_id, t['attempt'], t['id'], t['timestamp'],
            t['name'], max(0., (time.perf_counter() - t['started']) * 1000), t['status']))

    def end_tool(self, run_id, status):
        if run_id not in self.tools or self.tools[run_id]['status'] != 'running':
            return
        self.tools[run_id]['status'] = status
        self.save_tool(run_id)
        self.run('running')
