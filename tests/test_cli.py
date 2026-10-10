"""Installed CLI contracts with real SQLite storage and native generation."""

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from test_datasets import dictionary as dictionary

from lexi_ai import Lexicon, QuestionType, cli, migrations
from lexi_ai import datasets as distribution
from lexi_ai.datasets import checksum
from lexi_ai.errors import InvalidResourceError


class Completion:
    """Only provider transport is substituted; search, generation and writes are native."""

    def __init__(self):
        from test_end_to_end import LLM
        from test_questions_from_word import LLM as QuestionLLM

        self.words = LLM()
        self.questions = QuestionLLM()

    async def complete(self, instruction, data, schema):
        from test_prompting import prompt_context

        if schema.__name__ in {"QuestionBatch", "AnchoredQuestionBatch"}:
            return await self.questions.complete(instruction, data, schema)
        result = await self.words.complete(instruction, data, schema)
        if schema.__name__ == "EnrichmentBatch":
            for item in result.senses:
                item.relations = {}
            count = prompt_context(data, "sense_request")["examples_per_sense"]
            for item in result.senses:
                item.examples = [f"The [bank] opened at {hour}." for hour in range(count)]
        return result


@pytest.fixture
def provider(monkeypatch):
    completion = Completion()
    monkeypatch.setattr(
        cli,
        "create_lexicon",
        lambda args: Lexicon(args.db_url, db_schema=args.db_schema, llm=completion),
    )
    return completion


def test_cli_word_and_question_generation(source, tmp_path, provider, capsys):
    output = tmp_path / "content.sqlite"
    words = tmp_path / "words.txt"
    words.write_text("\ufeffbank\nbank\n")
    assert (
        cli.main(
            [
                "generate",
                "word",
                "--word-list",
                str(words),
                "--output",
                str(output),
                "--reference-path",
                str(source),
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert len(first["words"]) == 1 and first["failures"] == []
    word = first["words"][0]
    assert word["lemma"] == "bank"
    calls = list(provider.words.calls)
    args = [
        "generate",
        "question",
        str(word["senses"][0]["id"]),
        "--type",
        "WORD_TO_DEFINITION",
        "--count",
        "2",
        "--output",
        str(output),
    ]
    assert cli.main(args) == 0
    questions = json.loads(capsys.readouterr().out)
    assert len(questions) == 2 and all(q["sense_id"] == word["senses"][0]["id"] for q in questions)
    assert provider.words.calls == calls and provider.questions.calls == 1
    assert cli.main(args) == 0
    capsys.readouterr()
    with closing(sqlite3.connect(output)) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert connection.execute("SELECT COUNT(*) FROM words").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM questions").fetchone() == (4,)
    assert Path(str(output) + ".sha256").read_text().strip() == checksum(output)
    assert cli.main(["word", "search", "bank", "--db", str(output)]) == 0
    assert len(json.loads(capsys.readouterr().out)["items"]) == 1


def test_batch_failure_retains_successes_and_nonzero_exit(source, tmp_path, provider, capsys):
    words = tmp_path / "words.txt"
    words.write_text("no-matching-reference\nbank\n")
    output = tmp_path / "content.sqlite"
    assert (
        cli.main(
            [
                "generate",
                "word",
                "--word-list",
                str(words),
                "--output",
                str(output),
                "--reference-path",
                str(source),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["words"][0]["lemma"] == "bank"
    assert result["failures"][0]["target"] == "no-matching-reference"
    assert "no exact search match" in captured.err
    assert Path(str(output) + ".sha256").exists()


def test_cli_search_uses_local_reference_when_reference_hits_are_hidden(
    source, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "content.sqlite"
    migrations.upgrade_to_head(f"sqlite+aiosqlite:///{output}")

    def no_download(*args):
        pytest.fail("an explicit reference file must not cause a download")

    monkeypatch.setattr(distribution, "download", no_download)
    assert (
        cli.main(["word", "search", "bank", "--db", str(output), "--reference-path", str(source)])
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"items": []}


@pytest.mark.parametrize(
    ("kind", "count", "distractors"),
    [
        ("DEFINITION_TO_WORD", 1, 5),
        ("WORD_TO_DEFINITION", 1, 5),
        ("WORD_TO_USAGE", 1, 5),
        ("CONTEXT_TO_WORD", 8, 3),
        ("CLOZE_TO_WORD", 8, 3),
        ("DIALOGUE_COMPLETION", 8, 3),
        ("MEANING_IN_CONTEXT", 8, 3),
    ],
)
async def test_cli_question_defaults_and_explicit_overrides(kind, count, distractors):
    requests = []

    class Questions:
        async def generate_questions(self, sense_id, kind, count, **kwargs):
            requests.append((sense_id, kind, count, kwargs["distractor_count"]))
            return []

    args = cli.parser().parse_args(
        ["generate", "question", "1", "--type", kind, "--output", "unused.sqlite"]
    )
    await cli.dispatch(Questions(), args)
    assert requests == [(1, kind, count, distractors)]
    args.count, args.distractor_count = 2, 7
    await cli.dispatch(Questions(), args)
    assert requests[-1] == (1, kind, 2, 7)


def test_cli_saved_content_theme_translation_and_local_grading(source, tmp_path, provider, capsys):
    output = tmp_path / "content.sqlite"
    generate = ["--output", str(output), "--reference-path", str(source)]
    saved = ["--db", str(output)]

    def command(*arguments):
        assert cli.main(list(arguments)) == 0
        return json.loads(capsys.readouterr().out)

    command("db", "init", *saved, "--reference-path", str(source))
    matches = command(
        "word", "search", "bank", "--include-reference", *saved, "--reference-path", str(source)
    )
    handle = matches["items"][0]["reference_id"]
    assert command("reference", "validate", handle, *saved, "--reference-path", str(source)) == {
        "valid": True
    }
    generated = command("generate", "word", "bank", *generate)
    word_id = str(generated["words"][0]["id"])
    word = command("word", "get", word_id, *saved)
    sense_id = str(word["senses"][0]["id"])
    assert command("sense", "get", sense_id, *saved)[0]["word_id"] == int(word_id)
    assert command("sense", "previews", sense_id, *saved)[0]["sense_id"] == int(sense_id)
    command(
        "generate",
        "question",
        sense_id,
        "--type",
        "WORD_TO_DEFINITION",
        "--count",
        "1",
        "--output",
        str(output),
    )
    questions = command("question", "list", sense_id, *saved)
    question = command("question", "get", str(questions[0]["id"]), *saved)
    assert command("question", "retrieve", sense_id, *saved)["sense_id"] == int(sense_id)
    grade = command(
        "question",
        "grade",
        str(question["id"]),
        "--format",
        "SINGLE_CHOICE",
        "--answer",
        question["correct"]["id"],
        *saved,
    )
    assert grade["task_fit"]
    assert command("question", "delete", str(question["id"]), *saved)

    created = command(
        "generate",
        "theme",
        "pirate",
        "--name",
        "Pirate",
        "--concept",
        "nautical voice",
        "--output",
        str(output),
    )
    assert command("theme", "get", "pirate", *saved) == created
    assert command("theme", "list", *saved) == [created]
    assert command("theme", "update", "pirate", "--name", "Captain", *saved)["name"] == "Captain"
    assert command("theme", "delete", "pirate", *saved)

    translation = command(
        "generate", "translation", "--text", "bank", "--language", "vi", "--output", str(output)
    )
    assert translation == "ngân hàng"
    rows = command("translation", "list", *saved)
    assert len(rows) == 1
    translation_id = str(rows[0]["id"])
    assert command("translation", "get", translation_id, *saved) == rows[0]
    assert command("translation", "delete", translation_id, *saved)
    command(
        "generate", "translation", "--text", "a bank", "--language", "vi", "--output", str(output)
    )
    assert command("translation", "purge", *saved) == 1
    assert command("translation", "list", *saved) == []
    assert command("db", "current", *saved) == migrations.inspect_head()


def test_question_generation_rejects_missing_sense(tmp_path, provider, capsys):
    output = tmp_path / "content.sqlite"
    assert (
        cli.main(
            ["generate", "question", "999", "--type", "WORD_TO_DEFINITION", "--output", str(output)]
        )
        == 1
    )
    assert "InvalidResourceError" in capsys.readouterr().err
    assert provider.words.calls == [] and provider.questions.calls == 0
    with closing(sqlite3.connect(output)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM words").fetchone() == (0,)


@pytest.mark.parametrize(
    "arguments",
    [
        ["generate", "word", "--output", "unused.sqlite"],
        ["generate", "word", "bank", "--word-list", "unused.txt", "--output", "unused.sqlite"],
        [
            "generate",
            "question",
            "1",
            "--type",
            "WORD_TO_DEFINITION",
            "--output",
            "unused.sqlite",
            "--count",
            "0",
        ],
        [
            "generate",
            "question",
            "1",
            "--type",
            "WORD_TO_DEFINITION",
            "--output",
            "unused.sqlite",
            "--distractor-count",
            "2",
        ],
        ["question", "retrieve", "--db", "unused.sqlite"],
        ["question", "retrieve", "1", "--request", "1:CONTEXT_TO_WORD:1", "--db", "unused.sqlite"],
        ["question", "list", "1", "2", "--after-id", "1", "--db", "unused.sqlite"],
        ["theme", "update", "pirate", "--db", "unused.sqlite"],
        ["import", "--kind", "reference", "--words-only", "--into", "postgresql://localhost/test"],
        ["db", "init", "--db", "unused.sqlite", "--db-schema", "lexi"],
    ],
)
def test_invalid_arguments_fail_before_storage(arguments, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_target_urls_and_explicit_allocation():
    assert cli.target_url("postgresql://user:pass@localhost/test").startswith("postgresql+asyncpg:")
    assert cli.target_url("content.sqlite").startswith("sqlite+aiosqlite:///")
    assert cli.allocation("12:CONTEXT_TO_WORD:3") == (12, QuestionType.CONTEXT_TO_WORD, 3)
    for target in ("mysql://localhost/test", "sqlite+aiosqlite:///:memory:"):
        with pytest.raises(ValueError):
            cli.target_url(target)


def test_installed_entry_points_have_no_database_or_provider_side_effects(tmp_path):
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("LLM_", "DECISION_"))
    }
    for prefix in (
        [sys.executable, "-m", "lexi_ai"],
        [str(Path(sys.executable).with_name("lexi"))],
    ):
        for suffix in (["--help"], ["--version"], ["db", "head"]):
            result = subprocess.run(
                [*prefix, *suffix],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            assert result.stdout
    assert not list(tmp_path.iterdir())


async def test_word_reuse_needs_no_inference_and_rejects_invalid_inputs(source, tmp_path):
    url = cli.target_url(str(tmp_path / "content.sqlite"))
    await asyncio.to_thread(migrations.upgrade_to_head, url)
    lexicon = Lexicon(url, llm=Completion())
    try:
        await lexicon.import_reference(source)
        await lexicon.start()
        first = await lexicon.generate_word("bank")
        lexicon.llm = None
        reused, usage = await lexicon.generate_word("bank", with_usage=True)
        assert reused == first and usage == []
        for target in ([], " "):
            with pytest.raises(ValueError):
                await lexicon.generate_word(target)
        lexicon.llm = Completion()
        with pytest.raises(InvalidResourceError):
            await lexicon.generate_questions(
                first.senses[0].id, QuestionType.WORD_TO_DEFINITION, 0, distractor_count=3
            )
    finally:
        await lexicon.close()


async def test_cancelled_dispatch_closes_owned_resources(tmp_path, monkeypatch):
    args = cli.parser().parse_args(["word", "get", "1", "--db", str(tmp_path / "db.sqlite")])
    lexicon = Lexicon(cli.target_url(str(tmp_path / "db.sqlite")))
    monkeypatch.setattr(cli, "Lexicon", lambda *a, **k: lexicon)

    async def cancel(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr(cli, "dispatch", cancel)
    with pytest.raises(asyncio.CancelledError):
        await cli.run(args)
    assert lexicon._closed


def test_cli_redacts_provider_failure(tmp_path, monkeypatch, capsys):
    async def fail(args):
        raise RuntimeError("https://private-key@provider.test")

    monkeypatch.setattr(cli, "run", fail)
    assert cli.main(["db", "head"]) == 1
    captured = capsys.readouterr()
    assert "private-key" not in captured.err and captured.out == ""


async def test_batch_question_retrieval_dispatch(monkeypatch, tmp_path):
    # Exercise API argument wiring separately from native storage integration above.
    from unittest.mock import AsyncMock

    lexicon = AsyncMock()
    args = cli.parser().parse_args(
        [
            "question",
            "retrieve",
            "--request",
            "1:CONTEXT_TO_WORD:2",
            "--db",
            str(tmp_path / "db.sqlite"),
        ]
    )
    await cli.dispatch(lexicon, args)
    lexicon.retrieve_questions.assert_awaited_once_with(
        [(1, QuestionType.CONTEXT_TO_WORD, 2)], theme=None
    )


async def test_cli_postgresql_generation_and_sqlite_import(dictionary, source, tmp_path, provider):
    url = dictionary.db.engine.url.render_as_string(hide_password=False)
    pg_args = cli.parser().parse_args(
        ["generate", "word", "bank", "--output", url, "--reference-path", str(source)]
    )
    pg_args.words = ["bank"]
    generated = await cli.run(pg_args)
    assert generated["words"][0].id == 1
    assert dictionary._cambridge.path is None
    await dictionary.start()
    assert (await dictionary.search("bank")).items[0].word_id == 1
    output = tmp_path / "content.sqlite"
    args = cli.parser().parse_args(
        ["generate", "word", "bank", "--output", str(output), "--reference-path", str(source)]
    )
    args.words = ["bank"]
    word = (await cli.run(args))["words"][0]
    args = cli.parser().parse_args(
        [
            "generate",
            "question",
            str(word.senses[0].id),
            "--type",
            "WORD_TO_DEFINITION",
            "--output",
            str(output),
            "--count",
            "1",
        ]
    )
    questions = await cli.run(args)
    assert len(questions) == 1 and questions[0].sense_id == word.senses[0].id
    before = len(provider.words.calls)
    args = cli.parser().parse_args(["import", str(output), "--into", url])
    assert await cli.run(args) > 0
    assert await cli.run(args) == 0
    assert len(provider.words.calls) == before
    assert len(await dictionary.list_questions(1)) == 1
    await dictionary.close()
    restarted = Lexicon(url)
    try:
        assert (await restarted.get_word(1)).lemma == "bank"
        assert await restarted.import_reference() == 0
    finally:
        await restarted.close()


async def test_words_run_in_parallel_with_real_storage(source, tmp_path):
    from test_prompting import prompt_context
    from test_word_generation import payload, stage_payload

    words = ["bank", "harbor", "dock"]
    with sqlite3.connect(source) as connection:
        for index, word in enumerate(words[1:], 2):
            connection.execute("INSERT INTO words VALUES(?,?,?,'word','done')", (index, word, word))
            connection.execute(
                "INSERT INTO entries VALUES(?,?,'noun',0,NULL,NULL,NULL)", (index, index)
            )
            connection.execute(
                "INSERT INTO senses VALUES(?,?,'A place for money','A1',NULL,0)", (index, index)
            )

    class Parallel:
        def __init__(self):
            self.active = self.peak = 0
            self.release = asyncio.Event()
            self.started = []

        async def complete(self, instruction, data, schema):
            tag = "sense_request" if schema.__name__ == "EnrichmentBatch" else "word_request"
            context = prompt_context(data, tag)
            target = (
                context["word"]["lemma"]
                if schema.__name__ == "EnrichmentBatch"
                else context["target"]
            )
            if schema.__name__ == "InventoryOutput":
                self.started.append(target)
                self.active += 1
                self.peak = max(self.peak, self.active)
                if self.active == 2:
                    self.release.set()
                try:
                    await asyncio.wait_for(self.release.wait(), 2)
                    await asyncio.sleep(0)
                finally:
                    self.active -= 1
            output = payload(target)
            output["senses"][0]["examples"] = [f"The [{target}] opened."]
            return stage_payload(output, data, schema)

    url = cli.target_url(str(tmp_path / "content.sqlite"))
    await asyncio.to_thread(migrations.upgrade_to_head, url)
    provider = Parallel()
    lexicon = Lexicon(url, reference_path=str(source), llm=provider, max_concurrency=2)
    try:
        await lexicon.import_reference(source)
        await lexicon.start()
        result = await asyncio.gather(*(lexicon.generate_word(word) for word in [*words, "BANK"]))
        assert {word.lemma for word in result} == set(words)
        assert provider.peak == 2 and provider.active == 0
        assert len(provider.started) == len(words)
    finally:
        await lexicon.close()


async def test_default_instance_request_limit_is_128_and_shared(tmp_path):
    class Transport:
        def __init__(self):
            self.active = self.peak = 0
            self.release = asyncio.Event()

        async def complete(self, instruction, data, schema):
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.active == 128:
                self.release.set()
            try:
                await asyncio.wait_for(self.release.wait(), 4)
                await asyncio.sleep(0)
                return schema(content="translated")
            finally:
                self.active -= 1

    url = cli.target_url(str(tmp_path / "content.sqlite"))
    await asyncio.to_thread(migrations.upgrade_to_head, url)
    provider = Transport()
    lexicon = Lexicon(url, llm=provider)
    try:
        assert lexicon.max_concurrency == 128
        await asyncio.gather(
            *(lexicon.translate_text(f"sentence {index}", "vi") for index in range(150))
        )
        assert provider.peak == 128 and provider.active == 0
    finally:
        await lexicon.close()


async def test_parallel_provider_cancellation_releases_all_slots(tmp_path):
    class Transport:
        def __init__(self):
            self.active = 0
            self.started = asyncio.Event()

        async def complete(self, *args, **kwargs):
            self.active += 1
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1

    url = cli.target_url(str(tmp_path / "content.sqlite"))
    await asyncio.to_thread(migrations.upgrade_to_head, url)
    provider = Transport()
    lexicon = Lexicon(url, llm=provider, max_concurrency=2)

    async def run_all():
        async with asyncio.TaskGroup() as group:
            for index in range(4):
                group.create_task(lexicon.translate_text(f"sentence {index}", "vi"))

    task = asyncio.create_task(run_all())
    try:
        await provider.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.active == 0 and lexicon._requests._value == 2
    finally:
        await lexicon.close()
