"""Current archived context for RAG/tools; never fabricate missing business results."""
import json
from utils.client_dashboard_data import load_history, valid_number
from utils.report_generator import _optional_text
LOG_PATH='data/predictions_log.csv'


def _iter_rows():
    return load_history(LOG_PATH).to_dict('records')


def load_client_record(client_id=None):
    rows=_iter_rows()
    return next((row for row in rows if client_id is None or str(row.get('client_id') or '').strip()==client_id.strip()),None)


def _archived(row):
    value={name:None if _optional_text(row.get(name)) is None else row.get(name) for name in
        ('client_id','timestamp','prob_default','score_1000','threshold','rating','decision','risk_level','model_tag')}
    value['prob_default']=valid_number(row.get('prob_default'))
    return value


def build_client_context(client_id=None,max_chars=900):
    row=load_client_record(client_id)
    if row is None:return '',None
    text='[C] Valeurs archivees ; null signifie indisponible. '+json.dumps(_archived(row),ensure_ascii=False,default=str)
    return text[:max_chars],row


def build_logs_summary(n=10,max_chars=900):
    rows=_iter_rows()[:n]
    return ('[HISTORY] Valeurs archivees ; null signifie indisponible.\n'+'\n'.join(json.dumps(_archived(row),ensure_ascii=False,default=str) for row in rows))[:max_chars] if rows else ''
