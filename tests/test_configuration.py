"""Environment parsing belongs to repository scripts, never to the installed library."""

import argparse
import ast
import importlib.util
from pathlib import Path

import pytest

from lexi_ai import DecisionConfig, LLMConfig

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("example_config", ROOT / "examples" / "_config.py")
example_config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example_config)


def test_library_has_no_environment_loaders():
    for path in (ROOT / "lexi_ai").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [item.name for item in node.names]
                )
                assert not any(name.split(".")[0] in {"os", "dotenv"} for name in modules), path


def test_configs_do_not_expose_keys_in_repr():
    assert "private-key" not in repr(LLMConfig(api_key="private-key"))
    assert "private-key" not in repr(DecisionConfig(0.8, api_key="private-key"))


def test_shared_root_env_only_publishes_provider_variables():
    from dotenv import dotenv_values

    keys = set(dotenv_values(ROOT / ".env.example", interpolate=False))
    assert keys == set(example_config.PROVIDER_VARIABLES)
    assert all(name.startswith(("LLM_", "DECISION_")) for name in keys)
    assert example_config.DEFAULT_ENV_FILE == ROOT / ".env"
    assert not (ROOT / "examples" / ".env.example").exists()


async def test_example_env_is_explicit_and_does_not_mutate_environment(tmp_path, monkeypatch):
    import os

    for name in example_config.PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "LLM_API_KEY=llm-file-key\nLLM_BASE_URL=https://llm.test/v1\nLLM_MODEL=file-model\n"
        "LLM_STRUCTURED_OUTPUTS=false\n"
        "LLM_TEMPERATURE=0\n"
        "DECISION_API_KEY=decision-file-key\nDECISION_BASE_URL=https://decision.test\n"
        "DECISION_MODEL=decision-model\nDECISION_FALLBACK_MODEL=fallback-model\n"
    )
    monkeypatch.setenv("LLM_MODEL", "explicit-env-model")
    before = dict(os.environ)
    args = argparse.Namespace(
        db_url=f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}",
        cambridge_path="selected.db",
        db_schema=None,
        threshold=0.9,
        env_file=path,
    )
    lexicon = example_config.create_lexicon(args)
    try:
        assert lexicon.llm_config.model == "explicit-env-model"
        assert lexicon.llm_config.api_key == "llm-file-key"
        assert lexicon.llm_config.base_url == "https://llm.test/v1"
        assert lexicon.llm_config.structured_outputs is False
        assert lexicon.llm_config.temperature == 0
        assert lexicon.decision_config.model == "decision-model"
        assert lexicon.decision_config.api_key == "decision-file-key"
        assert lexicon.decision_config.base_url == "https://decision.test"
        assert lexicon.decision_config.threshold == 0.9
        assert lexicon.decision_fallback_model == "fallback-model"
        assert dict(os.environ) == before
        assert not (tmp_path / "unused.db").exists()
    finally:
        await lexicon.close()


def test_example_env_rejects_non_provider_settings(tmp_path):
    path = tmp_path / ".env"
    path.write_text("LLM_MODEL=selected\nLEXI_DB_URL=not-allowed\n")
    with pytest.raises(ValueError, match="unsupported variables"):
        example_config.create_lexicon(argparse.Namespace(env_file=path))


@pytest.mark.parametrize("value,expected", [("true", True), ("false", False), ("FALSE", False)])
def test_shared_env_parses_structured_outputs(value, expected):
    options = example_config.provider_options(
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
        example_config.provider_options(
            {
                "LLM_BASE_URL": "https://llm.test/v1",
                "LLM_MODEL": "selected",
                "LLM_STRUCTURED_OUTPUTS": value,
            },
            "LLM",
        )


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), ("0", 0), ("0.7", 0.7)])
def test_shared_env_parses_temperature(value, expected):
    options = example_config.provider_options(
        {
            "LLM_BASE_URL": "https://llm.test/v1",
            "LLM_MODEL": "selected",
            "LLM_TEMPERATURE": value,
        },
        "LLM",
    )
    assert options["temperature"] == expected


@pytest.mark.parametrize("value", ["true", "nan", "inf", "-0.1", "2.1"])
def test_shared_env_rejects_invalid_temperature(value):
    with pytest.raises(ValueError, match="LLM_TEMPERATURE"):
        example_config.provider_options(
            {
                "LLM_BASE_URL": "https://llm.test/v1",
                "LLM_MODEL": "selected",
                "LLM_TEMPERATURE": value,
            },
            "LLM",
        )


@pytest.mark.parametrize("decision_group", ["", "DECISION_API_KEY=\nDECISION_MODEL=\n"])
async def test_examples_need_only_llm_and_preserve_threshold(tmp_path, monkeypatch, decision_group):
    for name in example_config.PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "LLM_API_KEY=explicit\nLLM_BASE_URL=https://llm.test/v1\nLLM_MODEL=selected\n"
        + decision_group
    )
    lexicon = example_config.create_lexicon(
        argparse.Namespace(
            db_url=f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}",
            cambridge_path="selected.db",
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


def test_examples_require_llm_credentials(tmp_path, monkeypatch):
    for name in example_config.PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text("LLM_MODEL=selected\nLLM_BASE_URL=https://llm.test/v1\n")
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        example_config.create_lexicon(argparse.Namespace(env_file=path))
