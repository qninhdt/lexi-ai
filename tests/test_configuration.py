"""Environment parsing belongs to CLI configuration, never to the Python API."""

import argparse
from pathlib import Path

import pytest

from lexi_ai import DecisionConfig, LLMConfig
from lexi_ai.cli import config as cli_config
from lexi_ai.config import database_schema_name

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("value", ["", "a" * 64, "bad;schema", "has space", 123])
def test_schema_names_reject_invalid_or_truncated_identifiers(value):
    with pytest.raises(ValueError, match="schema name"):
        database_schema_name(value)


def test_configs_do_not_expose_keys_in_repr():
    assert "private-key" not in repr(LLMConfig(api_key="private-key"))
    assert "private-key" not in repr(DecisionConfig(0.8, api_key="private-key"))


def test_shared_root_env_only_publishes_provider_variables():
    from dotenv import dotenv_values

    keys = set(dotenv_values(ROOT / ".env.example", interpolate=False))
    assert keys == set(cli_config.PROVIDER_VARIABLES)
    assert all(name.startswith(("LLM_", "DECISION_")) for name in keys)
    assert cli_config.DEFAULT_ENV_FILE == Path(".env")
    assert not (ROOT / "examples" / ".env.example").exists()


async def test_cli_env_is_explicit_and_does_not_mutate_environment(tmp_path, monkeypatch):
    import os

    for name in cli_config.PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "LLM_API_KEY=llm-file-key\nLLM_BASE_URL=https://llm.test/v1\nLLM_MODEL=file-model\n"
        "LLM_STRUCTURED_OUTPUTS=false\n"
        "LLM_TEMPERATURE=0\n"
        "LLM_REASONING_EFFORT=xhigh\n"
        "LLM_MAX_RETRIES=1\n"
        "DECISION_API_KEY=decision-file-key\nDECISION_BASE_URL=https://decision.test\n"
        "DECISION_MODEL=decision-model\nDECISION_FALLBACK_MODEL=fallback-model\n"
    )
    monkeypatch.setenv("LLM_MODEL", "explicit-env-model")
    before = dict(os.environ)
    args = argparse.Namespace(
        db_url=f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}",
        reference_path="selected.db",
        db_schema=None,
        threshold=0.9,
        env_file=path,
    )
    lexicon = cli_config.create_lexicon(args)
    try:
        assert lexicon.llm_config.model == "explicit-env-model"
        assert lexicon.llm_config.api_key == "llm-file-key"
        assert lexicon.llm_config.base_url == "https://llm.test/v1"
        assert lexicon.llm_config.structured_outputs is False
        assert lexicon.llm_config.temperature == 0
        assert lexicon.llm_config.reasoning_effort == "xhigh"
        assert lexicon.llm_config.max_retries == 1
        assert lexicon.decision_config.model == "decision-model"
        assert lexicon.decision_config.api_key == "decision-file-key"
        assert lexicon.decision_config.base_url == "https://decision.test"
        assert lexicon.decision_config.threshold == 0.9
        assert lexicon.decision_fallback_model == "fallback-model"
        assert dict(os.environ) == before
        assert not (tmp_path / "unused.db").exists()
    finally:
        await lexicon.close()


def test_cli_env_rejects_non_provider_settings(tmp_path):
    path = tmp_path / ".env"
    path.write_text("LLM_MODEL=selected\nDB_URL=not-allowed\n")
    with pytest.raises(ValueError, match="unsupported variables"):
        cli_config.create_lexicon(argparse.Namespace(env_file=path))


@pytest.mark.parametrize("value,expected", [("true", True), ("false", False), ("FALSE", False)])
def test_shared_env_parses_structured_outputs(value, expected):
    options = cli_config.provider_options(
        {
            "LLM_BASE_URL": "https://llm.test/v1",
            "LLM_MODEL": "selected",
            "LLM_STRUCTURED_OUTPUTS": value,
        },
        "LLM",
    )
    assert options["structured_outputs"] is expected


@pytest.mark.parametrize("value", [None, "", "0", "invalid"])
def test_shared_env_rejects_invalid_structured_outputs(value):
    with pytest.raises(ValueError, match="true or false"):
        cli_config.provider_options(
            {
                "LLM_BASE_URL": "https://llm.test/v1",
                "LLM_MODEL": "selected",
                "LLM_STRUCTURED_OUTPUTS": value,
            },
            "LLM",
        )


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), ("0", 0), ("0.7", 0.7)])
def test_shared_env_parses_temperature(value, expected):
    options = cli_config.provider_options(
        {
            "LLM_BASE_URL": "https://llm.test/v1",
            "LLM_MODEL": "selected",
            "LLM_TEMPERATURE": value,
        },
        "LLM",
    )
    assert options["temperature"] == expected


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), (" xhigh ", "xhigh")])
def test_shared_env_parses_reasoning_effort(value, expected):
    options = cli_config.provider_options(
        {
            "LLM_BASE_URL": "https://llm.test/v1",
            "LLM_MODEL": "selected",
            "LLM_REASONING_EFFORT": value,
        },
        "LLM",
    )
    assert options["reasoning_effort"] == expected


@pytest.mark.parametrize("value", [True, 1, []])
def test_shared_env_rejects_nonstring_reasoning_effort(value):
    with pytest.raises(ValueError, match="LLM_REASONING_EFFORT"):
        cli_config.provider_options(
            {
                "LLM_BASE_URL": "https://llm.test/v1",
                "LLM_MODEL": "selected",
                "LLM_REASONING_EFFORT": value,
            },
            "LLM",
        )


@pytest.mark.parametrize("value,expected", [(None, 2), ("", 2), ("0", 0), ("2", 2)])
def test_shared_env_parses_max_retries(value, expected):
    options = cli_config.provider_options(
        {"LLM_BASE_URL": "https://llm.test/v1", "LLM_MODEL": "selected", "LLM_MAX_RETRIES": value},
        "LLM",
    )
    assert options["max_retries"] == expected


@pytest.mark.parametrize("value", ["-1", "1.5", "false", True, 2])
def test_shared_env_rejects_invalid_max_retries(value):
    with pytest.raises(ValueError, match="LLM_MAX_RETRIES"):
        cli_config.provider_options(
            {
                "LLM_BASE_URL": "https://llm.test/v1",
                "LLM_MODEL": "selected",
                "LLM_MAX_RETRIES": value,
            },
            "LLM",
        )


@pytest.mark.parametrize("value", ["true", "nan", "inf", "-0.1", "2.1"])
def test_shared_env_rejects_invalid_temperature(value):
    with pytest.raises(ValueError, match="LLM_TEMPERATURE"):
        cli_config.provider_options(
            {
                "LLM_BASE_URL": "https://llm.test/v1",
                "LLM_MODEL": "selected",
                "LLM_TEMPERATURE": value,
            },
            "LLM",
        )


@pytest.mark.parametrize("decision_group", ["", "DECISION_API_KEY=\nDECISION_MODEL=\n"])
async def test_clis_need_only_llm_and_preserve_threshold(tmp_path, monkeypatch, decision_group):
    for name in cli_config.PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "LLM_API_KEY=explicit\nLLM_BASE_URL=https://llm.test/v1\nLLM_MODEL=selected\n"
        + decision_group
    )
    lexicon = cli_config.create_lexicon(
        argparse.Namespace(
            db_url=f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}",
            reference_path="selected.db",
            db_schema=None,
            threshold=0.9,
            env_file=path,
        )
    )
    try:
        assert lexicon.decision_config.api_key is None
        assert lexicon.decision_config.threshold == 0.9
        assert lexicon.llm_config.model == "selected"
        assert not (tmp_path / "unused.db").exists()
    finally:
        await lexicon.close()


def test_clis_require_llm_credentials(tmp_path, monkeypatch):
    for name in cli_config.PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text("LLM_MODEL=selected\nLLM_BASE_URL=https://llm.test/v1\n")
    values = cli_config.load_provider_values(path)
    assert cli_config.provider_options(values, "LLM")["api_key"] is None


def test_reasoning_models_can_reserve_output_tokens_beyond_sixteen_thousand():
    assert LLMConfig(max_completion_tokens=32768).max_completion_tokens == 32768
    with pytest.raises(ValueError, match="completion token ceiling"):
        LLMConfig(max_completion_tokens=32769)


@pytest.mark.parametrize(
    "variable,field,value,expected",
    [
        ("LLM_MAX_COMPLETION_TOKENS", "max_completion_tokens", "32768", 32768),
        ("LLM_MAX_COMPLETION_TOKENS", "max_completion_tokens", "", 4096),
        ("LLM_TIMEOUT", "timeout", "120", 120),
        ("LLM_TIMEOUT", "timeout", "", 30),
    ],
)
def test_cli_provider_request_budget(variable, field, value, expected):
    options = cli_config.provider_options(
        {"LLM_BASE_URL": "https://llm.test/v1", "LLM_MODEL": "selected", variable: value},
        "LLM",
    )
    assert getattr(LLMConfig(**options), field) == expected


@pytest.mark.parametrize(
    "variable,value",
    [
        ("LLM_MAX_COMPLETION_TOKENS", "32769"),
        ("LLM_MAX_COMPLETION_TOKENS", "1.5"),
        ("LLM_MAX_COMPLETION_TOKENS", True),
        ("LLM_TIMEOUT", "0"),
        ("LLM_TIMEOUT", "121"),
        ("LLM_TIMEOUT", "nan"),
    ],
)
def test_cli_rejects_invalid_request_budget(variable, value):
    with pytest.raises(ValueError, match=variable):
        cli_config.provider_options(
            {"LLM_BASE_URL": "https://llm.test/v1", "LLM_MODEL": "selected", variable: value},
            "LLM",
        )
