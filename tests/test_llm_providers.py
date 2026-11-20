"""
Offline checks for the Claude path in utils/llm.py and the generator/judge
split. A fake Anthropic client stands in for the API.

    python -m pytest tests/test_llm_providers.py -q
"""
import os
from types import SimpleNamespace as NS

os.environ.setdefault("OPENAI_API_KEY", "offline-test")

import utils.llm as llm  # noqa: E402


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kw):
        self.requests.append(kw)
        text = self.replies.pop(0)
        return NS(
            content=[NS(type="thinking", thinking="private"), NS(type="text", text=text)],
            stop_reason="end_turn",
            usage=NS(input_tokens=100, cache_creation_input_tokens=50,
                     cache_read_input_tokens=400, output_tokens=30),
        )


def use_fake(monkeypatch, replies):
    fake = NS(messages=FakeMessages(replies))
    monkeypatch.setitem(llm._clients, "anthropic", fake)
    llm._usage.clear()
    return fake.messages


def test_claude_call_shape_and_text(monkeypatch):
    msgs = use_fake(monkeypatch, ["hello"])
    out = llm.call_llm("sys", "hi", max_tokens=150, model="claude-sonnet-5-5")
    req = msgs.requests[0]
    assert out == "hello"                        # thinking block skipped
    assert req["system"] == "sys"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": llm.CLAUDE_EFFORT}
    assert req["max_tokens"] == 150 + llm.CLAUDE_THINKING_HEADROOM


def test_cache_breakpoints_become_cache_control_max_four(monkeypatch):
    msgs = use_fake(monkeypatch, ["{}"])
    blocks = [{"type": "text", "text": f"b{i}", "prompt_cache_breakpoint": llm.CACHE_BREAKPOINT}
              for i in range(6)] + [{"type": "text", "text": "tail"}]
    llm.call_llm_json("sys", blocks, model="claude-sonnet-5-5")
    sent = msgs.requests[0]["messages"][0]["content"]
    assert all(set(b) <= {"type", "text", "cache_control"} for b in sent)
    assert [i for i, b in enumerate(sent) if "cache_control" in b] == [2, 3, 4, 5]


def test_json_with_surrounding_text_is_parsed_and_bad_json_retried(monkeypatch):
    use_fake(monkeypatch, ['Sure! {"a": 1} hope that helps', "not json", '{"b": 2}'])
    assert llm.call_llm_json("s", "u", model="claude-sonnet-5-5") == {"a": 1}
    assert llm.call_llm_json("s", "u", model="claude-sonnet-5-5") == {"b": 2}


def test_usage_and_cost(monkeypatch):
    use_fake(monkeypatch, ["x"])
    llm.call_llm("s", "u", model="claude-sonnet-5-5")
    report = llm.usage_report()
    # 100*2 + 50*2.5 + 400*0.2 + 30*10 = 705 per million
    assert "100 input, 50 cache write, 400 cache read, 30 output" in report
    assert "$0.00" in report and "total" in report


def test_judge_calls_use_judge_model(monkeypatch):
    """Agent 4 and series QA pass JUDGE_MODEL; the generators don't."""
    import tests.test_multi_meeting as tm
    seen = []
    orig = tm.SeriesFake.json

    def spy(self, system, user, max_tokens=4000, model=None):
        seen.append((system[:40], model))
        return orig(self, system, user, max_tokens, model)

    monkeypatch.setattr(tm.SeriesFake, "json", spy)
    tm.run_series("adaptation_01")
    judge = [m for s, m in seen if m == llm.JUDGE_MODEL]
    gen = [m for s, m in seen if m is None]
    assert judge and gen
    assert all(m in (None, llm.JUDGE_MODEL) for _, m in seen)


def test_output_root_separates_generators(monkeypatch):
    monkeypatch.setenv("GEN_MODEL", "claude-sonnet-5-5")
    monkeypatch.setattr(llm, "GEN_MODEL", "claude-sonnet-5-5")
    monkeypatch.delenv("RUN_TAG", raising=False)
    assert llm.output_root().parts[-3:] == ("data", "outputs", "claude-sonnet-5-5")
    monkeypatch.setenv("RUN_TAG", "v2")
    assert llm.output_root().name == "claude-sonnet-5-5__v2"
    monkeypatch.delenv("GEN_MODEL")
    assert llm.output_root().parts[-2:] == ("data", "outputs")
