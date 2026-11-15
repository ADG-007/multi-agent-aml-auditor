"""The LLM client: schema output, one retry, cache modes, model lock, audit log."""

import json

import httpx
import pytest
from pydantic import BaseModel

from src.llm import LLMClient, LLMFailureReason, LLMMode, LLMStageFailure, Role, fingerprint


class Answer(BaseModel):
    level: str
    count: int


class FakeOllama:
    """Replies in order; records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, path: str, payload: dict) -> dict:
        self.requests.append((path, payload))
        if path != "/api/chat":
            return {}
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"message": {"content": reply}, "eval_count": 7, "prompt_eval_count": 11}

    def chats(self) -> list[dict]:
        return [p for path, p in self.requests if path == "/api/chat"]


def client(tmp_path, fake, mode=LLMMode.LIVE) -> LLMClient:
    return LLMClient(
        mode, cache_root=tmp_path / "cache", audit_path=tmp_path / "audit.jsonl", transport=fake
    )


GOOD = '{"level": "HIGH", "count": 2}'


def test_valid_answer_is_parsed_and_sent_with_the_schema(tmp_path):
    fake = FakeOllama(GOOD)
    result = client(tmp_path, fake).call(Role.VISION, "prompt", Answer, "v1")
    assert result.value == Answer(level="HIGH", count=2)
    assert (result.cached, result.attempts) == (False, 1)
    sent = fake.chats()[0]
    assert sent["format"] == Answer.model_json_schema()
    assert sent["think"] is False and sent["options"]["temperature"] == 0.0


def test_one_retry_shows_the_model_its_error(tmp_path):
    fake = FakeOllama('{"level": "HIGH"}', "```json\n" + GOOD + "\n```")
    result = client(tmp_path, fake).call(Role.VISION, "prompt", Answer, "v1")
    assert result.attempts == 2 and result.value.count == 2
    retry = fake.chats()[1]["messages"]
    assert retry[-2]["role"] == "assistant"
    assert "not valid" in retry[-1]["content"] and "count" in retry[-1]["content"]


def test_two_bad_answers_fail_and_nothing_is_saved(tmp_path):
    fake = FakeOllama("nonsense", '{"level": 1}')
    c = client(tmp_path, fake)
    with pytest.raises(LLMStageFailure) as failure:
        c.call(Role.VISION, "prompt", Answer, "v1")
    assert failure.value.reason == LLMFailureReason.SCHEMA_VALIDATION_FAILED
    assert c.cache.count("vision") == 0
    assert c.calls[-1]["ok"] is False


def test_saved_answer_replays_without_a_call(tmp_path):
    client(tmp_path, FakeOllama(GOOD)).call(Role.VISION, "prompt", Answer, "v1")
    fake = FakeOllama()
    replay = client(tmp_path, fake, LLMMode.CACHED).call(Role.VISION, "prompt", Answer, "v1")
    assert replay.cached and replay.value.level == "HIGH"
    assert fake.requests == []


def test_cache_key_covers_model_version_prompt_and_image(tmp_path):
    image = tmp_path / "card.png"
    image.write_bytes(b"one")
    fake = FakeOllama(GOOD, GOOD, GOOD, GOOD)
    c = client(tmp_path, fake)
    c.call(Role.VISION, "prompt", Answer, "v1")
    c.call(Role.VISION, "prompt", Answer, "v2")
    c.call(Role.VISION, "prompt", Answer, "v1", images=[image])
    image.write_bytes(b"two")
    c.call(Role.VISION, "prompt", Answer, "v1", images=[image])
    assert len(fake.chats()) == 4
    assert fake.chats()[2]["messages"][-1]["images"]
    assert fingerprint("m", "v1", "p") != fingerprint("n", "v1", "p")


def test_cached_mode_miss_and_off_mode_raise(tmp_path):
    with pytest.raises(LLMStageFailure) as miss:
        client(tmp_path, FakeOllama(), LLMMode.CACHED).call(Role.VISION, "p", Answer, "v1")
    assert miss.value.reason == LLMFailureReason.CACHE_MISS
    with pytest.raises(LLMStageFailure) as off:
        client(tmp_path, FakeOllama(), LLMMode.OFF).call(Role.VISION, "p", Answer, "v1")
    assert off.value.reason == LLMFailureReason.MODEL_OFF


def test_only_one_llm_is_loaded_at_a_time(tmp_path):
    fake = FakeOllama(GOOD, GOOD, GOOD)
    c = client(tmp_path, fake)
    c.call(Role.VISION, "a", Answer, "v1")
    c.call(Role.WEB_RESEARCH, "b", Answer, "v1")  # same model: no unload
    c.call(Role.AUDITOR, "c", Answer, "v1")
    unloads = [p for path, p in fake.requests if path == "/api/generate"]
    assert unloads == [{"model": c.model_for(Role.VISION), "keep_alive": 0}]


def test_unreachable_and_timeout_are_typed(tmp_path):
    down = client(tmp_path, FakeOllama(httpx.ConnectError("refused")))
    with pytest.raises(LLMStageFailure) as failure:
        down.call(Role.VISION, "p", Answer, "v1")
    assert failure.value.reason == LLMFailureReason.PROVIDER_UNAVAILABLE
    slow = client(tmp_path, FakeOllama(httpx.ReadTimeout("slow")))
    with pytest.raises(LLMStageFailure) as failure:
        slow.call(Role.VISION, "p", Answer, "v1")
    assert failure.value.reason == LLMFailureReason.TIMED_OUT


def test_audit_log_has_one_line_per_call(tmp_path):
    c = client(tmp_path, FakeOllama(GOOD))
    c.call(Role.VISION, "prompt", Answer, "v1", agent="vision_kyc")
    c.call(Role.VISION, "prompt", Answer, "v1", agent="vision_kyc")
    lines = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [x["cached"] for x in lines] == [False, True]
    assert lines[0]["agent"] == "vision_kyc" and lines[0]["output_tokens"] == 7


def test_embedding_runs_on_the_cpu(tmp_path):
    def fake(path, payload):
        assert path == "/api/embed" and payload["options"] == {"num_gpu": 0}
        return {"embeddings": [[0.1, 0.2] for _ in payload["input"]]}

    assert client(tmp_path, fake).embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
