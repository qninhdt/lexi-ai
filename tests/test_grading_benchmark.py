"""Small offline grading runs: existing prompts, single model, measured accounting."""

import asyncio
import json
import os
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from typesafe_sdk import Noul

from benchmarks.run import (
    TASKS,
    configured_model,
    load_config,
    load_dataset,
    load_settings,
    prepare_case,
    run_benchmark,
)
from benchmarks.scoring import case_cost, summarize
from examples._config import DEFAULT_ENV_FILE, PROVIDER_VARIABLES, load_provider_values
from lexi_ai import DecisionConfig, DecisionMode, LLMConfig
from lexi_ai.inference.decision import DecisionModel

ROOT = Path(__file__).resolve().parents[1]
PRICING = {
    "currency": "USD",
    "model_ids": ["fixture-model"],
    "input_per_million": 1,
    "output_per_million": 3,
    "cache_read_per_million": 0.5,
}


@pytest.mark.parametrize(
    "profile,prefix,mode",
    [
        ("grading_decision", "DECISION", DecisionMode.DECISION_ONLY),
        ("grading_llm", "LLM", DecisionMode.LLM_ONLY),
    ],
)
def test_examples_and_benchmark_share_env_without_provider_defaults(
    tmp_path, monkeypatch, profile, prefix, mode
):
    for name in PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=llm-key\nLLM_BASE_URL=https://llm.test/v1\nLLM_MODEL=llm-custom\n"
        "DECISION_API_KEY=decision-key\nDECISION_BASE_URL=https://decision.test\n"
        "DECISION_MODEL=decision-custom\n"
    )
    settings = json.loads((ROOT / "benchmarks" / "config.json").read_text())
    settings["pricing"][f"{prefix.lower()}-custom"] = PRICING
    settings_path = tmp_path / "config.json"
    settings_path.write_text(json.dumps(settings))
    config = load_config(settings_path, profile, env_file=env_file)
    shared = load_provider_values(env_file)
    assert config["mode"] == mode
    assert config["api_key"] == shared[f"{prefix}_API_KEY"]
    assert config["base_url"] == shared[f"{prefix}_BASE_URL"]
    assert config["model"] == shared[f"{prefix}_MODEL"]
    assert config["pricing"] == PRICING
    if prefix == "LLM":
        assert config["structured_outputs"] is True
    assert DEFAULT_ENV_FILE == ROOT / ".env"
    assert f"{prefix}_API_KEY" not in os.environ
    monkeypatch.setenv(f"{prefix}_MODEL", "process-model")
    assert load_config(settings_path, profile, env_file=env_file)["model"] == "process-model"
    assert load_provider_values(env_file)[f"{prefix}_MODEL"] == "process-model"
    monkeypatch.setenv(f"{prefix}_MODEL", "")
    with pytest.raises(ValueError, match=f"{prefix}_MODEL is required"):
        load_config(settings_path, profile, env_file=env_file)


def test_each_benchmark_profile_can_select_a_price_snapshot(tmp_path, monkeypatch):
    for name in PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=llm-key\nLLM_BASE_URL=https://llm.test/v1\nLLM_MODEL=requested-alias\n"
    )
    settings = {
        "pricing": {"special-tier": PRICING},
        "benchmarks": {
            "custom": {
                "mode": "llm_only",
                "threshold": 0.9,
                "timeout": 15,
                "pricing": "special-tier",
            }
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings))
    config = load_config(path, "custom", env_file=env_file)
    assert config["model"] == "requested-alias"
    assert config["pricing"] == PRICING
    assert config["threshold"] == 0.9 and config["timeout"] == 15


@pytest.mark.parametrize(
    "env_value,override,expected",
    [
        ("true", None, True),
        ("false", None, False),
        ("true", False, False),
        ("false", True, True),
    ],
)
async def test_benchmark_structured_outputs_uses_env_or_profile_override(
    tmp_path, monkeypatch, env_value, override, expected
):
    for name in PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=fake\nLLM_BASE_URL=https://llm.test/v1\nLLM_MODEL=test\n"
        f"LLM_STRUCTURED_OUTPUTS={env_value}\n"
    )
    profile = {"mode": "llm_only", "threshold": 0.8, "timeout": 30}
    if override is not None:
        profile["structured_outputs"] = override
    settings_path = tmp_path / "config.json"
    settings_path.write_text(json.dumps({"benchmarks": {"test": profile}}))
    config = load_config(settings_path, "test", env_file=env_file)
    assert config["structured_outputs"] is expected
    async with configured_model(config) as model:
        assert model.llm_config.structured_outputs is expected
        assert model.fallback.structured_outputs is expected
        assert model.primary is None


@pytest.mark.parametrize(
    "override,value,expected",
    [
        (False, None, 0.7),
        (True, 0, 0),
        (True, None, None),
    ],
)
async def test_benchmark_temperature_uses_env_or_profile_override(
    tmp_path, monkeypatch, override, value, expected
):
    for name in PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=fake\nLLM_BASE_URL=https://llm.test/v1\nLLM_MODEL=test\n"
        "LLM_STRUCTURED_OUTPUTS=false\nLLM_TEMPERATURE=0.7\n"
    )
    profile = {"mode": "llm_only", "threshold": 0.8, "timeout": 30}
    if override:
        profile["temperature"] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"benchmarks": {"test": profile}}))
    config = load_config(path, "test", env_file=env_file)
    assert config["temperature"] == expected
    async with configured_model(config) as model:
        assert model.llm_config.temperature == expected
        assert model.fallback.structured_outputs is False
        if expected is not None:
            assert model._fallback_provider.temperature == expected


@pytest.mark.parametrize("value", [True, "0", -1, 2.1, float("nan")])
def test_benchmark_rejects_invalid_temperature(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "benchmarks": {
                    "test": {
                        "mode": "llm_only",
                        "threshold": 0.8,
                        "timeout": 30,
                        "temperature": value,
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="temperature"):
        load_settings(path, "test")


def test_temperature_is_not_silently_applied_to_decision_only(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "benchmarks": {
                    "test": {
                        "mode": "decision_only",
                        "threshold": 0.8,
                        "timeout": 30,
                        "temperature": 0,
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="only to llm_only"):
        load_settings(path, "test")


@pytest.mark.parametrize("value", ["false", "", "0", "invalid"])
def test_output_config_rejects_invalid_flags(tmp_path, monkeypatch, value):
    for name in PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    profile = {
        "mode": "llm_only",
        "threshold": 0.8,
        "timeout": 30,
        "structured_outputs": value,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"benchmarks": {"test": profile}}))
    with pytest.raises(ValueError, match="boolean"):
        load_settings(path, "test")


def example_cases():
    cases = []
    for task in TASKS:
        case = json.loads((ROOT / "benchmarks" / "templates" / f"{task}.template.jsonl").read_text().splitlines()[0])
        case.pop("example")
        cases.append(case)
    return cases


class Transport:
    def __init__(self, cases):
        self.cases = iter(cases)
        self.calls = []

    async def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        expected = next(self.cases)["expected"]
        return SimpleNamespace(
            model="fixture-model",
            usage=SimpleNamespace(
                input_tokens=100,
                cache_read_tokens=25,
                cache_write_tokens=0,
                output_tokens=10,
            ),
            nouls={
                name: SimpleNamespace(noul=0.8 if expected[name] else 0.2)
                for name, question in questions.items()
                if isinstance(question, Noul)
            },
            choices={
                name: SimpleNamespace(choice=expected[name], confidence=0.1)
                for name, question in questions.items()
                if not isinstance(question, Noul)
            },
        )


@pytest.mark.parametrize("mode", [DecisionMode.DECISION_ONLY, DecisionMode.LLM_ONLY])
@pytest.mark.parametrize("structured_outputs", [True, False])
async def test_one_model_run_across_all_six_grading_tasks(tmp_path, mode, structured_outputs):
    cases = example_cases()
    active, unused = Transport(cases), Transport([])
    model = DecisionModel(
        DecisionConfig(0.8),
        primary=active if mode == DecisionMode.DECISION_ONLY else unused,
        fallback=active if mode == DecisionMode.LLM_ONLY else unused,
        llm_config=LLMConfig(structured_outputs=structured_outputs, temperature=0.7),
    )
    output = tmp_path / "run"
    summary = await run_benchmark(
        cases,
        model,
        mode=mode,
        timeout=5,
        output=output,
        pricing=PRICING,
        metadata={"model": "requested-alias"},
    )
    assert summary["accuracy"] == 1
    assert summary["total_tokens"] == 660
    assert summary["total_cost"] == pytest.approx(0.000705)
    assert summary["latency_p95_ms"] >= 0
    assert summary["actual_model_ids"] == ["fixture-model"]
    assert summary["mode"] == mode
    assert len(active.calls) == 6 and not unused.calls
    for case, (state, questions) in zip(cases, active.calls, strict=True):
        assert (state, questions) == prepare_case(case)
        assert "expected" not in state
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert summarize(records, PRICING)["total_cost"] == summary["total_cost"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["mode"] == mode
    assert manifest["structured_outputs"] == (
        structured_outputs if mode == DecisionMode.LLM_ONLY else None
    )
    assert manifest["temperature"] == (0.7 if mode == DecisionMode.LLM_ONLY else None)
    assert len(manifest["prompt_sha256"]) == 6
    assert "api_key" not in manifest
    with pytest.raises(FileExistsError):
        await run_benchmark(cases, model, mode=mode, timeout=5, output=output)
    assert len(active.calls) == 6


def record(task="task_a", *, prediction=None, status="ok"):
    return {
        "task": task,
        "expected": {"meaning": "correct", "form": "correct"},
        "prediction": prediction or {"meaning": "correct", "form": "correct"},
        "status": status,
        "latency_ms": 10,
        "usage": [
            {
                "model_id": "fixture-model",
                "input_tokens": 100,
                "cache_read_tokens": 25,
                "cache_write_tokens": 0,
                "output_tokens": 10,
            }
        ],
    }


def test_macro_exact_match_includes_errors_and_does_not_weight_by_case_count():
    rows = [
        record(),
        record(prediction={"meaning": "correct", "form": "form_error"}),
        record("task_b"),
    ]
    summary = summarize(rows, PRICING)
    assert summary["accuracy"] == 0.75
    assert summary["by_task"]["task_a"]["accuracy"] == 0.5
    rows[1]["status"] = "error"
    rows[1]["prediction"] = rows[1]["expected"]
    assert summarize(rows, PRICING)["accuracy"] == 0.75
    assert summarize(rows, PRICING)["errors"] == 1


def test_cost_replaces_cache_rate_without_counting_cached_input_twice():
    assert case_cost(record(), PRICING) == Decimal("0.0001175")
    write_pricing = {**PRICING, "cache_write_per_million": 2}
    row = record()
    row["usage"][0]["cache_write_tokens"] = 10
    assert case_cost(row, write_pricing) == Decimal("0.0001275")
    request_pricing = {"currency": "USD", "model_ids": ["fixture-model"], "per_request": 0.01}
    assert summarize([row], request_pricing)["total_cost"] == 0.01
    row["status"] = "error"
    assert summarize([row], request_pricing)["total_cost"] is None


def test_unknown_usage_and_unpriced_actual_model_remain_unknown():
    row = record()
    row["usage"][0]["cache_read_tokens"] = None
    assert summarize([row], PRICING)["total_cost"] is None
    assert summarize([row], PRICING)["total_tokens"] == 110
    assert case_cost(row, {**PRICING, "cache_read_per_million": 1}) == Decimal("0.00013")
    row["usage"][0]["input_tokens"] = None
    summary = summarize([record(), row], PRICING)
    assert summary["total_tokens"] is None and not summary["tokens_complete"]
    assert summary["known_token_counts"]["input_tokens"] == 100
    assert summary["token_known_cases"] == 1
    assert summary["known_cost"] == pytest.approx(0.0001175)
    row = record()
    row["usage"][0]["model_id"] = "different-model"
    assert summarize([row], PRICING)["total_cost"] is None


async def test_timeout_counts_as_wrong_and_does_not_cancel_remaining_cases(tmp_path):
    cases = [example_cases()[0], deepcopy(example_cases()[0])]
    cases[1]["id"] = "second-case"

    class SlowFirst(Transport):
        async def system_one(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                await asyncio.sleep(10)
            return await super().system_one(**kwargs)

    model = DecisionModel(DecisionConfig(0.8), primary=SlowFirst(cases))
    summary = await run_benchmark(
        cases,
        model,
        mode=DecisionMode.DECISION_ONLY,
        timeout=0.01,
        output=tmp_path / "timeout",
        pricing=PRICING,
    )
    assert summary["accuracy"] == 0.5 and summary["errors"] == 1
    assert summary["total_cost"] is None and summary["total_tokens"] is None
    rows = (tmp_path / "timeout" / "results.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["status"] == "timeout"
    assert json.loads(rows[1])["status"] == "ok"


def test_dataset_templates_are_valid_but_not_silently_treated_as_verified_gt(tmp_path):
    assert len(load_dataset(ROOT / "benchmarks" / "templates", allow_examples=True)) == 10
    with pytest.raises(ValueError, match="template case"):
        load_dataset(ROOT / "benchmarks" / "templates")
    cases = example_cases()
    path = tmp_path / "grading.jsonl"
    path.write_text("\n".join(json.dumps(case) for case in cases))
    assert load_dataset(path) == cases
    cases[0]["expected"]["task_fit"] = 0.8
    with pytest.raises(ValueError, match="boolean"):
        prepare_case(cases[0])
    cases[1]["expected"]["matched_sense"] = "sense_missing"
    with pytest.raises(ValueError, match="one of"):
        prepare_case(cases[1])
    cases[2]["task"] = "resolve_sense_relations"
    with pytest.raises(ValueError, match="unsupported grading task"):
        prepare_case(cases[2])


async def test_bad_dataset_fails_before_inference_or_output_creation(tmp_path):
    cases = example_cases()
    cases[-1]["expected"].pop("construction")
    transport = Transport(cases)
    model = DecisionModel(DecisionConfig(0.8), primary=transport)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="exactly"):
        await run_benchmark(
            cases,
            model,
            mode=DecisionMode.DECISION_ONLY,
            timeout=1,
            output=output,
        )
    assert not output.exists() and not transport.calls
