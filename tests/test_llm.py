import json
import importlib
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from openai import OpenAI


def response(text='{"ok": true}', status='completed', refusal=False):
    content = [{"type": "refusal", "refusal": "Cannot comply"}] if refusal else [
        {"type": "output_text", "text": text, "annotations": []}]
    return {"id": "resp_test", "object": "response", "created_at": 1,
            "model": "test-model", "status": status,
            "incomplete_details": {"reason": "max_output_tokens"} if status == 'incomplete' else None,
            "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                        "status": "completed", "content": content}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                      "input_tokens_details": {"cached_tokens": 40},
                      "output_tokens_details": {"reasoning_tokens": 5}}}


@pytest.fixture
def api(monkeypatch):
    # Only the HTTP boundary is replaced; calls pass through the real OpenAI SDK.
    monkeypatch.setenv('OPENAI_MODEL', 'test-model')
    from utils import llm
    llm = importlib.reload(llm)
    requests, replies = [], []
    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=replies.pop(0) if replies else response())
    client = OpenAI(api_key='offline-test', http_client=httpx.Client(
        transport=httpx.MockTransport(handler)), max_retries=0)
    monkeypatch.setattr(llm, 'client', client)
    yield llm, requests, replies
    client.close()


def test_json_and_block_conversion(api):
    llm, requests, _ = api
    blocks = [{"type": "text", "text": "profile", "cache_control": {"type": "ephemeral"}},
              {"type": "text", "text": "history"}]
    assert llm.call_llm_json('Return a decision', blocks, 150) == {"ok": True}
    path, payload = requests[0]
    assert path == '/v1/responses'
    assert payload['model'] == 'test-model'
    assert payload['input'][0]['content'] == [
        {"type": "input_text", "text": "profile"}, {"type": "input_text", "text": "history"},
        {"type": "input_text", "text": "Return a valid JSON object."}]
    assert payload['text']['format'] == {"type": "json_object"}
    assert payload['max_output_tokens'] == 150
    assert payload['store'] is False
    assert 'cache_control' in blocks[0]  # don't mutate caller's context


def test_plain_text_and_model_override(api):
    llm, requests, replies = api
    replies.append(response('Style notes'))
    assert llm.call_llm('Analyze style', 'meeting', model='other-model') == 'Style notes'
    assert requests[0][1]['model'] == 'other-model'
    assert 'text' not in requests[0][1]


@pytest.mark.parametrize('reply,match', [
    (response(status='incomplete'), 'incomplete'),
    (response(refusal=True), 'refus'),
    (response(''), 'empty'),
])
def test_invalid_completion_is_not_accepted(api, reply, match):
    llm, _, replies = api
    replies.append(reply)
    with pytest.raises(ValueError, match=match):
        llm.call_llm_json('JSON please', 'meeting')


def test_retry_json_but_reject_non_object(api):
    llm, requests, replies = api
    replies.extend([response('[]'), response('not json'), response('{"valid": 1}')])
    assert llm.call_llm_json('JSON please', 'meeting') == {"valid": 1}
    assert len(requests) == 3


def test_usage_parallel_calls_no_claude_pricing(api):
    llm, _, _ = api
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: llm.call_llm('style', 'meeting'), range(3)))
    report = llm.usage_report()
    assert '300 input' in report
    assert '120 cached' in report
    assert '60 output' in report
    assert '$' not in report


def test_missing_config_fails_before_network(api, monkeypatch):
    llm, requests, _ = api
    monkeypatch.delenv('OPENAI_MODEL')
    with pytest.raises(ValueError, match='OPENAI_MODEL'):
        llm.call_llm('style', 'meeting')
    assert requests == []


def test_invalid_json_exhausts_retry_budget(api):
    llm, requests, replies = api
    replies.extend([response('[]')] * 3)
    with pytest.raises(ValueError, match='3 attempts'):
        llm.call_llm_json('JSON please', 'meeting')
    assert len(requests) == 3


def test_reasoning_budget_and_effort(api, monkeypatch):
    llm, requests, _ = api
    monkeypatch.setenv('OPENAI_MIN_OUTPUT_TOKENS', '2048')
    monkeypatch.setenv('OPENAI_REASONING_EFFORT', 'low')
    llm.call_llm_json('JSON please', 'meeting', max_tokens=150)
    assert requests[0][1]['max_output_tokens'] == 2048
    assert requests[0][1]['reasoning'] == {'effort': 'low'}


def test_missing_key_is_actionable(api, monkeypatch):
    llm, requests, _ = api
    monkeypatch.setattr(llm, 'client', None)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        llm.call_llm('style', 'meeting')
    assert requests == []


def test_json_mode_mentions_json_in_input(api):
    llm, requests, _ = api
    llm.call_llm_json('JSON requested in instructions only', 'Return an object')
    assert 'json' in requests[0][1]['input'][0]['content'].lower()
