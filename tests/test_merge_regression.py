"""Offline regression checks for Lexi behavior inside Rita's series runner."""
from types import SimpleNamespace

import pytest
from schema import Turn
from tests.test_multi_meeting import run_series
from tests.test_task_pipeline import agent3


@pytest.mark.parametrize('task_id,count', [('integration_02', 2), ('self_review_01', 3)])
def test_series_preserves_closing_and_exploration_metrics(monkeypatch, task_id, count):
    original = agent3.speak
    closing_intents = []

    def speak(speaker, content, shared_context, brief, style, transcript, intent, extra_note=''):
        if intent in ('close', 'goodbye'):
            closing_intents.append(intent)
            return 'Thanks, let us stop here.' if intent == 'close' else 'See you.'
        return original(speaker, content, shared_context, brief, style, transcript, intent, extra_note)

    monkeypatch.setattr(agent3, 'speak', speak)
    result, fake = run_series(task_id)
    assert closing_intents.count('close') == count
    assert closing_intents.count('goodbye') >= count
    assert fake.calls['moves'] == count
    for meeting in result['meetings']:
        texts = [t['text'] for t in meeting['transcript']]
        assert 'Thanks, let us stop here.' in texts
        assert texts[-1] == 'See you.'
        feedback = meeting['qa_feedback']
        assert feedback['exploration_share'] is not None
        assert feedback['turn_moves']
        assert meeting['qa_history'][-1]['exploration_share'] == feedback['exploration_share']


def test_speech_rewrite_cannot_drop_required_date(monkeypatch):
    original = 'We expect approval on March 1, 2027, and we still need to check the clinic schedule together.'
    replacement = 'We expect approval quite soon, and we still need to check the clinic schedule together before deciding.'
    monkeypatch.setattr(agent3, 'call_llm_json', lambda *a, **k: {'text': replacement})
    monkeypatch.setattr(agent3, 'check_seed_leak', lambda *a, **k: (False, ''))
    content = SimpleNamespace(seed=SimpleNamespace(benchmark_task='State the approval date.'))
    result = agent3.speech_pass(content, [Turn(speaker='A', text=original)], facts=[
        {'speaker': 'A', 'fact': 'Approval is on March 1, 2027.', 'cue': 'March 1, 2027'}])
    assert result[0].text == original
