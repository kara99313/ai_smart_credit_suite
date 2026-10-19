import json
from unittest.mock import Mock, patch
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from utils.credit_agent_contract import FEATURES, CONTRACT, user_values, prediction_scope
from utils.agents import run_agent_once
from utils import agent_tools, multitask_agent

PAYLOAD = dict.fromkeys(FEATURES, 1)


def test_contract_matches_fastapi():
    import server
    assert tuple(server.ScoringInputs.model_fields) == FEATURES
    assert server.PredictRequest.model_fields['threshold'].default == .10


@pytest.mark.parametrize('query', ['Fais une prediction de credit', 'Prediction : {"Income": 12}', 'Calcule le score avec Income=12, Age=30'])
def test_incomplete_never_invokes_provider(query):
    agent = Mock()
    result = run_agent_once(agent, query, [])
    agent.invoke.assert_not_called()
    assert 'Veuillez fournir' in result['output']
    for name in FEATURES:
        if name not in query:
            assert name in result['output']


def test_schema_question_without_tools():
    agent = Mock()
    result = run_agent_once(agent, 'Quelles variables sont necessaires pour une prediction ?', [])
    agent.invoke.assert_not_called()
    assert all(name in result['output'] for name in FEATURES)
    assert 'optionnel' in result['output']


@pytest.mark.parametrize('threshold', [None, .2])
def test_complete_payload_allowed(threshold):
    payload = dict(PAYLOAD)
    if threshold is not None:
        payload['threshold'] = threshold
    response = dict(pd=.3, score_1000=700, rating='CCC', decision='REVIEW/REJECT', threshold=.1, used_features=list(FEATURES))
    def invoke(request, config):
        return {'output': agent_tools.predict_credit_tool(json.dumps(payload))}
    agent = Mock(invoke=Mock(side_effect=invoke))
    with patch.object(agent_tools, 'predict_credit_api', return_value=response) as api:
        result = run_agent_once(agent, 'Prediction ' + json.dumps(payload), [])
    api.assert_called_once_with(payload, timeout=30.0)
    assert '30.00%' in result['output']


@pytest.mark.parametrize('payload', [{}, {'Income': 1}, dict(PAYLOAD, Age=None), dict(PAYLOAD, Fake=2)])
def test_tool_rejects_missing_and_unknown(payload):
    with patch.object(agent_tools, 'predict_credit_api') as api:
        assert 'ERROR' in agent_tools.predict_credit_tool(json.dumps(payload))
    api.assert_not_called()


def test_never_trust_assistant_or_invented_values():
    values = user_values('Age=30', [AIMessage(json.dumps(PAYLOAD)), HumanMessage('Income=12')])
    assert values == {'Income': 12, 'Age': 30}
    with prediction_scope(values), patch.object(agent_tools, 'predict_credit_api') as api:
        assert 'ERROR' in agent_tools.predict_credit_tool(json.dumps(PAYLOAD))
    api.assert_not_called()


def test_multiturn_explicit_values():
    history = [HumanMessage(json.dumps({k: v for k, v in PAYLOAD.items() if k != 'Age'}))]
    agent = Mock(invoke=Mock(return_value={'output': 'ok'}))
    run_agent_once(agent, 'Age=30', history)
    agent.invoke.assert_called_once()


def test_now_schema_and_execution_unchanged():
    tool = next(t for t in multitask_agent._tools() if t.name == 'now')
    assert list(tool.args_schema.model_fields) == ['fmt']
    assert len(tool.invoke({})) == 19
    assert tool.invoke({'fmt': '%Y'}).isdigit()
    agent = Mock(invoke=Mock(return_value={'output': 'heure'}))
    run_agent_once(agent, 'Quelle heure est-il ?', [])
    agent.invoke.assert_called_once()


def test_prediction_description_exposes_contract():
    tool = next(t for t in multitask_agent._tools() if t.name == 'predict_credit')
    assert tool.description == CONTRACT
    assert all(name in tool.args_schema.model_fields['payload_json'].description for name in FEATURES)


def test_complete_but_changed_value_rejected_and_scope_reset():
    with prediction_scope(PAYLOAD), patch.object(agent_tools, 'predict_credit_api') as api:
        result = agent_tools.predict_credit_tool(json.dumps(dict(PAYLOAD, Income=999)))
    assert 'ERROR' in result
    api.assert_not_called()
    from utils.credit_agent_contract import payload_error
    assert payload_error(dict(PAYLOAD, Income=999)) is None


def test_threshold_cannot_be_invented():
    from utils.credit_agent_contract import payload_error
    with prediction_scope(PAYLOAD):
        assert payload_error(dict(PAYLOAD, threshold=.1)) is None
        assert 'ERROR' in payload_error(dict(PAYLOAD, threshold=.5))


def test_ambiguous_number_is_not_truncated():
    assert 'Income' not in user_values('Income=1,000', [])
    assert 'Income' not in user_values('Income=12oops', [])


def test_french_partial_prediction_requests_only_missing_values():
    query = ('Prédit le risque de crédit pour un client de 32 ans avec un revenu de 300000, '
             'un montant de prêt de 100000, un DTI de 0,35 et un score psychométrique de 0,62.')
    expected = dict(Age=32, Income=300000, LoanAmount=100000,
                    DTIRatio=.35, TrustScorePsychometric=.62)
    assert user_values(query, []) == expected
    agent = Mock()
    with patch.object(agent_tools, 'predict_credit_tool') as tool, patch.object(agent_tools, 'predict_credit_api') as api:
        result = run_agent_once(agent, query, [])
    agent.invoke.assert_not_called()
    tool.assert_not_called()
    api.assert_not_called()
    missing = result['output'].split('Variables manquantes : ', 1)[1].split('. ', 1)[0]
    assert set(missing.split(', ')) == {
        'HouseholdSize', 'NumCreditLines', 'CommunityGroupMember', 'HasMortgage',
        'MonthsEmployed', 'HasSocialAid', 'MobileMoneyTransactions', 'InterestRate',
        'LoanTerm', 'InformalIncome',
    }


PRODUCTION_QUERY = ('DTIRatio = 0,35 ; TrustScorePsychometric = 0,62 ; HouseholdSize = 4 ; '
    'NumCreditLines = 2 ; Income = 300000 ; CommunityGroupMember = Yes ; HasMortgage = No ; '
    'MonthsEmployed = 36 ; HasSocialAid = No ; MobileMoneyTransactions = 20 ; Age = 32 ; '
    'InterestRate = 12 ; LoanTerm = 24 ; LoanAmount = 100000 ; InformalIncome = 50000.')
PRODUCTION_VALUES = dict(DTIRatio=.35, TrustScorePsychometric=.62, HouseholdSize=4,
    NumCreditLines=2, Income=300000, CommunityGroupMember='Yes', HasMortgage='No',
    MonthsEmployed=36, HasSocialAid='No', MobileMoneyTransactions=20, Age=32,
    InterestRate=12, LoanTerm=24, LoanAmount=100000, InformalIncome=50000)


def test_exact_production_request_extracts_all_values_and_allows_preflight(caplog):
    from utils.credit_agent_contract import missing_values, preflight, payload_error
    import logging
    values = user_values(PRODUCTION_QUERY, [])
    recognized_features_count = sum(name in values for name in FEATURES)
    missing_features_count = len(missing_values(values))
    assert recognized_features_count == 15
    assert missing_features_count == 0
    assert values == PRODUCTION_VALUES
    assert 'threshold' not in values
    assert preflight(PRODUCTION_QUERY, values) is None
    agent = Mock(invoke=Mock(return_value={'output': 'ok'}))
    caplog.clear()
    with caplog.at_level(logging.INFO):
        run_agent_once(agent, PRODUCTION_QUERY, [])
    agent.invoke.assert_called_once()
    diagnostic, = [json.loads(record.getMessage().partition('Agent preflight ')[2])
                   for record in caplog.records if record.getMessage().startswith('Agent preflight ')]
    assert diagnostic['preflight_executed'] is True
    assert diagnostic['recognized_features_count'] == 15
    assert diagnostic['missing_features_count'] == 0
    assert diagnostic['history_messages_count'] == 0
    assert diagnostic['reason'] == 'complete_features'
    assert diagnostic['invoke_allowed'] is True
    with prediction_scope(values):
        assert payload_error(PRODUCTION_VALUES) is None
        assert payload_error(dict(PRODUCTION_VALUES, Income=300001)) is not None


@pytest.mark.parametrize('separator', [' = ', ': '])
@pytest.mark.parametrize('decimal_separator', [',', '.'])
def test_production_values_preserve_named_formats(separator, decimal_separator):
    query = PRODUCTION_QUERY.replace(' = ', separator)
    if decimal_separator == '.':
        query = query.replace('0,35', '0.35').replace('0,62', '0.62')
    assert user_values(query, []) == PRODUCTION_VALUES


@pytest.mark.parametrize('query,expected', [
    ('DTIRatio=0,35', {'DTIRatio': .35}),
    ('TrustScorePsychometric: 0,62', {'TrustScorePsychometric': .62}),
    ('InformalIncome=50000.', {'InformalIncome': 50000}),
    ('DTIRatio=0,350', {'DTIRatio': .350}),
    ('Income=1,000', {}),
    ('Income=12,345', {}),
    ('Income=300,000', {}),
    ('Income=1,000,000', {}),
    ('Income=12oops', {}),
    ('Income=12.34oops', {}),
    ('Income=12,34oops', {}),
    ('Income=12..34', {}),
    ('Income=12 000', {}),
    ('Income=12.34.56', {}),
    ('Income=12,34,56', {}),
    ('DTIRatio=0,35 ; threshold: 0,20.', {'DTIRatio': .35, 'threshold': .20}),
    ('CommunityGroupMember=Yes ; HasMortgage=No ; HasSocialAid=No.',
     {'CommunityGroupMember': 'Yes', 'HasMortgage': 'No', 'HasSocialAid': 'No'}),
    ('CommunityGroupMember="oui, explicitement"', {'CommunityGroupMember': 'oui, explicitement'}),
    ('DTIRatio ; TrustScorePsychometric ; InformalIncome', {}),
])
def test_explicit_tokens_are_not_truncated_or_invented(query, expected):
    assert user_values(query, []) == expected


@pytest.mark.parametrize('decimal_separator', [',', '.'])
def test_existing_natural_french_values_keep_both_decimal_formats(decimal_separator):
    query = ('Predit le risque de credit pour un client de 32 ans avec un revenu de 300000, '
             'un montant de pret de 100000, un DTI de 0,35 et un score psychometrique de 0,62.')
    if decimal_separator == '.':
        query = query.replace('0,35', '0.35').replace('0,62', '0.62')
    assert user_values(query, []) == dict(Age=32, Income=300000, LoanAmount=100000,
                                        DTIRatio=.35, TrustScorePsychometric=.62)


def test_production_missing_value_is_not_filled_from_assistant_history():
    from utils.credit_agent_contract import missing_values, preflight
    query = PRODUCTION_QUERY.replace(' ; InformalIncome = 50000.', '.')
    values = user_values(query, [AIMessage('InformalIncome=50000')])
    assert values == {name: value for name, value in PRODUCTION_VALUES.items() if name != 'InformalIncome'}
    assert missing_values(values) == ['InformalIncome']
    assert 'InformalIncome' in preflight(query, values)
    assert 'threshold' not in values
