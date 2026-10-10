"""The maintainer script uses real SQLite and resumes interrupted native generation."""

import asyncio
import importlib.util
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from test_cli import provider as provider

from lexi_ai.datasets import checksum


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[1] / "scripts/generate-content.py"
    spec = importlib.util.spec_from_file_location("generate_content", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(source, tmp_path):
    words = tmp_path / "words.txt"
    words.write_text("\ufeffbank\nbank\n")
    return [
        "--word-list",
        str(words),
        "--output",
        str(tmp_path / "content.sqlite"),
        "--reference-path",
        str(source),
        "--count",
        "2",
    ]


def test_script_generates_and_resumes_with_progress_and_partial_failure(
    script, source, tmp_path, provider, capsys
):
    args = arguments(source, tmp_path)
    words = tmp_path / "words.txt"
    words.write_text(words.read_text() + "no-matching-reference\n")
    assert script.main(args) == 1
    captured = capsys.readouterr()
    assert "100%" in captured.err and "no exact search match" in captured.err
    calls = list(provider.words.calls)
    assert provider.questions.calls == 7
    assert script.main(args) == 1
    assert provider.words.calls == calls and provider.questions.calls == 7
    output = tmp_path / "content.sqlite"
    with closing(sqlite3.connect(output)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM words").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM questions").fetchone() == (11,)
        rows = connection.execute("SELECT question_type,payload FROM questions").fetchall()
        for kind, raw in rows:
            fixed = kind in {"DEFINITION_TO_WORD", "WORD_TO_DEFINITION", "WORD_TO_USAGE"}
            assert len(json.loads(raw)["distractors"]) == (5 if fixed else 3)
            assert sum(item[0] == kind for item in rows) == (1 if fixed else 2)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    with closing(sqlite3.connect(output)) as connection:
        connection.execute("DELETE FROM questions WHERE id=(SELECT MIN(id) FROM questions)")
        connection.commit()
    assert script.main(args) == 1
    assert provider.words.calls == calls and provider.questions.calls == 8
    with closing(sqlite3.connect(output)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM questions").fetchone() == (11,)
    assert Path(str(output) + ".sha256").read_text().strip() == checksum(output)


async def test_script_cancel_after_word_commit_resumes_without_word_inference(
    script, source, tmp_path, provider
):
    args = script.parser().parse_args(arguments(source, tmp_path))
    entered = asyncio.Event()
    original = provider.complete

    async def blocked(instruction, data, schema):
        if schema.__name__ in {"QuestionBatch", "AnchoredQuestionBatch"}:
            entered.set()
            await asyncio.Event().wait()
        return await original(instruction, data, schema)

    provider.complete = blocked
    task = asyncio.create_task(script.run(args))
    try:
        await asyncio.wait_for(entered.wait(), 5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    calls = list(provider.words.calls)
    with closing(sqlite3.connect(args.output)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM words").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM questions").fetchone() == (0,)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    provider.complete = original
    assert await script.run(args) == 0
    assert provider.words.calls == calls and provider.questions.calls == 7


async def test_script_parallel_words_and_seven_banks_without_batch_barrier(
    script, source, tmp_path, provider
):
    from test_prompting import prompt_context
    from test_word_generation import payload, stage_payload

    with sqlite3.connect(source) as connection:
        connection.execute("INSERT INTO word_alternatives VALUES(1,'banks','inflection')")
        connection.execute("INSERT INTO words VALUES(2,'harbor','harbor','word','done')")
        connection.execute("INSERT INTO entries VALUES(12,2,'noun',0,NULL,NULL,NULL)")
        connection.execute("INSERT INTO senses VALUES(102,12,'A place for boats','A2',NULL,0)")
    args = script.parser().parse_args(arguments(source, tmp_path))
    args.word_list.write_text("bank\nharbor\nbanks\n")
    both_words = asyncio.Event()
    questions_started = asyncio.Event()
    all_types_started = asyncio.Event()
    inventories, kinds = [], set()

    async def concurrent(instruction, data, schema):
        if schema.__name__ in {"InventoryOutput", "EnrichmentBatch"}:
            tag = "sense_request" if schema.__name__ == "EnrichmentBatch" else "word_request"
            context = prompt_context(data, tag)
            target = (
                context["word"]["lemma"]
                if schema.__name__ == "EnrichmentBatch"
                else context["target"]
            )
            if schema.__name__ == "InventoryOutput":
                inventories.append(target)
                if len(inventories) == 2:
                    both_words.set()
                await asyncio.wait_for(both_words.wait(), 5)
            elif schema.__name__ == "EnrichmentBatch" and target == "harbor":
                # Bank's questions must start before the other Word finishes.
                await asyncio.wait_for(questions_started.wait(), 5)
            output = payload(target)
            output["senses"][0]["examples"] = [f"The [{target}] opened."]
            if target == "bank":
                output["senses"][0]["forms"] = ["banks" + "|pl"]
            return stage_payload(output, data, schema)
        context = prompt_context(data)
        if context["word"] == "bank":
            kinds.add(context["question_type"])
            questions_started.set()
            if len(kinds) == 7:
                all_types_started.set()
            await asyncio.wait_for(all_types_started.wait(), 5)
        # Adapt the existing transport fixture's tagged target before native validation.
        response = await provider.questions.complete(instruction, data, schema.__bases__[0])
        return schema.model_validate(
            json.loads(response.model_dump_json().replace("bank", context["word"]))
        )

    provider.complete = concurrent
    assert await script.run(args) == 0
    assert sorted(inventories) == ["bank", "harbor"]
    assert len(kinds) == 7 and provider.questions.calls == 14
    with closing(sqlite3.connect(args.output)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM words").fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM questions").fetchone() == (22,)


def test_script_words_only_limit_and_resume(script, source, tmp_path, provider, capsys):
    args = arguments(source, tmp_path) + ["--words-only", "--limit", "1"]
    (tmp_path / "words.txt").write_text("bank\nno-matching-reference\n")
    assert script.main(args) == 0
    assert "100%" in capsys.readouterr().err
    calls = list(provider.words.calls)
    assert len(calls) == 2 and provider.questions.calls == 0
    assert script.main(args) == 0
    assert provider.words.calls == calls and provider.questions.calls == 0
    with closing(sqlite3.connect(tmp_path / "content.sqlite")) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM words WHERE generation_state='DONE'"
        ).fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM questions").fetchone() == (0,)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


async def test_script_caps_concurrent_words_at_thirty_two(
    script, source, tmp_path, provider, monkeypatch
):
    from test_prompting import prompt_context
    from test_word_generation import payload, stage_payload

    from lexi_ai import cli

    words = [f"item-{index}" for index in range(35)]
    with sqlite3.connect(source) as connection:
        for index, target in enumerate(words, 2):
            connection.execute(
                "INSERT INTO words VALUES(?,?,?,?,?)", (index, target, target, "word", "done")
            )
            connection.execute(
                "INSERT INTO entries VALUES(?,?,?,0,NULL,NULL,NULL)", (index + 20, index, "noun")
            )
            connection.execute(
                "INSERT INTO senses VALUES(?,?,?,'A2',NULL,0)", (index + 200, index + 20, "An item")
            )
    args = script.parser().parse_args(arguments(source, tmp_path) + ["--words-only"])
    args.word_list.write_text("\n".join(words))
    assert args.word_concurrency == 32 and args.max_concurrency == 128
    first_group = asyncio.Event()
    active, peak = 0, 0
    started = []
    original = cli.create_lexicon

    def instrument(arguments):
        lexicon = original(arguments)
        generate = lexicon.generate_word

        async def tracked(target):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            started.append(target)
            if len(started) == 32:
                first_group.set()
            try:
                await asyncio.wait_for(first_group.wait(), 5)
                return await generate(target)
            finally:
                active -= 1

        lexicon.generate_word = tracked
        return lexicon

    monkeypatch.setattr(cli, "create_lexicon", instrument)

    async def complete(instruction, data, schema):
        assert schema.__name__ in {"InventoryOutput", "EnrichmentBatch"}
        tag = "word_request" if schema.__name__ == "InventoryOutput" else "sense_request"
        context = prompt_context(data, tag)
        target = (
            context["target"] if schema.__name__ == "InventoryOutput" else context["word"]["lemma"]
        )
        output = payload(target)
        output["senses"][0]["examples"] = [f"The [{target}] arrived."]
        return stage_payload(output, data, schema)

    provider.complete = complete
    assert await script.run(args) == 0
    assert peak == 32 and active == 0 and started == words
    with closing(sqlite3.connect(args.output)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM words WHERE generation_state='DONE'"
        ).fetchone() == (35,)
        assert connection.execute("SELECT COUNT(*) FROM questions").fetchone() == (0,)


@pytest.mark.parametrize("option", ["--word-concurrency", "--limit"])
async def test_script_rejects_nonpositive_word_controls(script, source, tmp_path, option):
    args = script.parser().parse_args(arguments(source, tmp_path) + [option, "0"])
    with pytest.raises(ValueError, match="positive"):
        await script.run(args)
    assert not args.output.exists()
