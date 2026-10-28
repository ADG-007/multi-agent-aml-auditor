"""Saved model answers: the key and the cache folder."""

from src.llm.cache import ResponseCache, fingerprint


def test_fingerprint_is_stable_and_changes_with_any_input():
    key = fingerprint("model-a", "kyc-v1", "prompt")

    assert key == fingerprint("model-a", "kyc-v1", "prompt")
    assert key != fingerprint("model-b", "kyc-v1", "prompt")
    assert key != fingerprint("model-a", "kyc-v2", "prompt")
    assert key != fingerprint("model-a", "kyc-v1", "prompt.")


def test_a_saved_answer_loads_only_with_its_own_key(tmp_path):
    cache = ResponseCache("kyc", root=tmp_path)

    assert cache.load("ACC-T001", "k1") is None

    cache.save(
        "ACC-T001",
        {
            "key": "k1",
            "prompt_version": "kyc-v1",
            "answer": "HIGH",
        },
    )

    assert cache.load("ACC-T001", "k1")["answer"] == "HIGH"
    assert cache.load("ACC-T001", "k2") is None


def test_count_only_includes_the_given_prompt_version(tmp_path):
    cache = ResponseCache("kyc", root=tmp_path)

    assert cache.count("kyc-v1") == 0

    cache.save("ACC-T001", {"key": "a", "prompt_version": "kyc-v1"})
    cache.save("ACC-T002", {"key": "b", "prompt_version": "kyc-v1"})
    cache.save("ACC-T003", {"key": "c", "prompt_version": "kyc-v2"})

    assert cache.count("kyc-v1") == 2