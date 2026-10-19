"""Explicit credit input contract for the Agent, independent of scoring."""
import json
import re
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar

FEATURES = ('DTIRatio', 'TrustScorePsychometric', 'HouseholdSize', 'NumCreditLines',
    'Income', 'CommunityGroupMember', 'HasMortgage', 'MonthsEmployed', 'HasSocialAid',
    'MobileMoneyTransactions', 'Age', 'InterestRate', 'LoanTerm', 'LoanAmount', 'InformalIncome')
CONTRACT = (
    'Les 15 variables obligatoires sont : ' + ', '.join(FEATURES) + '. '
    'threshold est un parametre de decision optionnel (defaut 0.10), hors des 15 variables. '
    'Ne jamais inventer de variable ou de valeur. Lister les variables manquantes et demander '
    'leurs valeurs sans appeler predict_credit. Appeler cet outil uniquement pour une prediction '
    'avec les 15 valeurs explicitement fournies par utilisateur. Une question sur les variables '
    'requises doit recevoir la liste exacte sans appel outil. payload_json est une chaine JSON objet.')
_supplied = ContextVar('credit_user_values', default=None)


def _natural_values(text):
    """Collect numbers attached to explicit French labels, without inference."""
    text = ''.join(c for c in unicodedata.normalize('NFKD', str(text).lower())
                   if not unicodedata.combining(c))
    labels = {
        'Age': r'(?:age|client)\s+de\s+',
        'Income': r'revenu\s*(?:de\s*)?',
        'LoanAmount': r'montant\s+(?:de\s+)?pret\s*(?:de\s*)?',
        'DTIRatio': r'dti\s*(?:de\s*)?',
        'TrustScorePsychometric': r'score\s+psychometrique\s*(?:de\s*)?',
    }
    values = {}
    number = r'([-+]?\d+(?:[.,]\d+)?)(?!\w|[.,]\d|\s*\d)'
    for name, label in labels.items():
        pattern = r'\b(?:' + label + r')\s*(?:[:=]\s*)?' + number
        for match in re.finditer(pattern, text):
            if name == 'Age' and not re.match(r'\s*ans\b', text[match.end():]):
                continue
            raw = match[1].replace(',', '.')
            values[name] = float(raw) if '.' in raw else int(raw)
    return values


def supplied_values(text):
    """Only explicit named scalar values are collected, never inferred."""
    values = _natural_values(text)
    names = '|'.join((*FEATURES, 'threshold'))
    pattern = r'["\x27]?\b(' + names + r')\b["\x27]?\s*[:=]\s*("[^"\n]*"|\x27[^\x27\n]*\x27|true\b|false\b|null\b|oui\b|non\b|yes\b|no\b|[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)(?!\w|[.,]\d|\.\S|\s*\d)'
    for match in re.finditer(pattern, str(text), re.I):
        name = next(n for n in (*FEATURES, 'threshold') if n.lower() == match[1].lower())
        raw = match[2]
        # Keep ambiguous thousands-style tokens unparsed; never truncate them.
        if re.fullmatch(r'[-+]?[1-9]\d{0,2},\d{3}', raw):
            continue
        if re.fullmatch(r'[-+]?\d+,\d+(?:[eE][-+]?\d+)?', raw):
            raw = raw.replace(',', '.')
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw.strip(chr(34) + chr(39))
        values[name] = value
    return values


def user_values(query, history):
    values = {}
    for message in history or []:
        if getattr(message, 'type', None) == 'human':
            values.update(supplied_values(message.content))
        elif isinstance(message, (tuple, list)) and len(message) == 2 and message[0] == 'user':
            values.update(supplied_values(message[1]))
    values.update(supplied_values(query))
    return values


def missing_values(payload):
    return [n for n in FEATURES if payload.get(n) is None or payload.get(n) == '']


def missing_reply(missing):
    return 'Variables manquantes : ' + ', '.join(missing) + '. Veuillez fournir leurs valeurs. threshold est optionnel (defaut 0.10).'


def preflight(query, values):
    from utils.agent_preflight_diagnostics import mark_preflight
    mark_preflight()
    text = ''.join(c for c in unicodedata.normalize('NFKD', query.lower()) if not unicodedata.combining(c))
    question = re.search(r'quelles?|quels?|what|which|liste|list\b', text)
    if question and re.search(r'variable|champ|field|input|parametre', text) and not supplied_values(query):
        mark_preflight('schema_question')
        return CONTRACT.split('Ne jamais')[0].strip()
    if re.search(r'predict|predi|scor|calcul.*credit', text) or supplied_values(query):
        missing = missing_values(values)
        if missing:
            mark_preflight('missing_features')
            return missing_reply(missing)
        mark_preflight('complete_features')
    else:
        mark_preflight('unrecognized_request')
    return None


@contextmanager
def prediction_scope(values):
    token = _supplied.set(values)
    try:
        yield
    finally:
        _supplied.reset(token)


def payload_error(payload):
    missing = missing_values(payload)
    if missing:
        return 'ERROR: ' + missing_reply(missing)
    unknown = set(payload) - set(FEATURES) - {'threshold'}
    if unknown:
        return 'ERROR: Variables inconnues : ' + ', '.join(sorted(unknown))
    supplied = _supplied.get()
    if supplied is not None:
        if any(name not in supplied or supplied[name] != value for name, value in payload.items() if name != 'threshold'):
            return 'ERROR: Valeurs non fournies par utilisateur. Demander leurs valeurs explicites.'
        if 'threshold' in payload and payload['threshold'] != supplied.get('threshold', .10):
            return 'ERROR: threshold non fourni par utilisateur (defaut 0.10).'
    return None
