import pytest

from lexi_ai.db.session import Database
from lexi_ai.errors import InvalidOutputError, InvalidResourceError
from lexi_ai.schema import Base
from lexi_ai.text import content_hash
from lexi_ai.translation import storage as translations
from lexi_ai.translation.generate import translate_text


class LLM:
    def __init__(self):
        self.calls = 0

    async def complete(self, instruction, data, schema):
        self.calls += 1
        assert "expert multilingual translator" in instruction
        assert "<translation_context>" in data
        return schema(content="ngân hàng")


async def test_hash_hit_miss_and_management(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dict.db'}")
    llm = LLM()
    try:
        await db.create_schema(Base.metadata)
        assert await translate_text(db, llm, "bank", "vi") == "ngân hàng"
        assert await translate_text(db, None, "bank", "vi") == "ngân hàng"
        assert llm.calls == 1
        assert await translate_text(db, llm, " bank", "vi") == "ngân hàng"
        assert llm.calls == 2
        rows = await translations.list_rows(db)
        assert len(rows) == 2
        assert rows[0].input_hash != rows[1].input_hash
        assert (await translations.get(db, rows[0].id)).content == "ngân hàng"
        assert await translations.remove(db, rows[0].id)
        assert await translations.purge(db) == 1
    finally:
        await db.close()


async def test_failure_does_not_write(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dict.db'}")
    await db.create_schema(Base.metadata)

    class Failing:
        async def complete(self, instruction, data, schema):
            raise RuntimeError("provider failed")

    try:
        with pytest.raises(RuntimeError, match="provider failed"):
            await translate_text(db, Failing(), "bank", "vi")
        assert await translations.list_rows(db) == []
    finally:
        await db.close()


async def test_invalid_translation_is_not_cached_and_languages_are_distinct(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dict.db'}")

    class InvalidLLM:
        async def complete(self, instruction, data, schema):
            return {"content": ""}

    try:
        await db.create_schema(Base.metadata)
        with pytest.raises(InvalidOutputError):
            await translate_text(db, InvalidLLM(), "bank", "vi")
        assert await translations.list_rows(db) == []
        llm = LLM()
        assert await translate_text(db, llm, "bank", "vi") == "ngân hàng"
        assert await translate_text(db, llm, "bank", "fr") == "ngân hàng"
        assert llm.calls == 2
    finally:
        await db.close()


async def test_target_markup_is_unwrapped_before_translation_and_cache_lookup(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dict.db'}")
    plain = "  She brought the issue up.\n"
    marked = '  She <t inf="past">brought</t> the issue <t inf="base">up</t>.\n'

    class CaptureLLM(LLM):
        async def complete(self, instruction, data, schema):
            assert f"<text>{plain}</text>" in data
            assert "<t inf=" not in data
            return await super().complete(instruction, data, schema)

    llm = CaptureLLM()
    try:
        await db.create_schema(Base.metadata)
        assert await translate_text(db, llm, marked, "vi") == "ngân hàng"
        assert await translate_text(db, None, plain, "vi") == "ngân hàng"
        assert await translate_text(db, None, marked, "vi") == "ngân hàng"
        rows = await translations.list_rows(db)
        assert len(rows) == 1 and rows[0].input_hash == content_hash(plain)
        assert llm.calls == 1
    finally:
        await db.close()


@pytest.mark.parametrize(
    "content",
    [
        '<t inf="base">bank',
        '<t inf="unknown">bank</t>',
        '<t inf="base"><t inf="base">bank</t></t>',
        '<t inf="base"> </t>',
        "<t>bank</t>",
        "bank</t>",
        "",
        None,
    ],
)
async def test_invalid_target_markup_fails_before_cache_or_provider(tmp_path, content):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dict.db'}")
    llm = LLM()
    try:
        await db.create_schema(Base.metadata)
        # Even an existing plain-text hit cannot bypass markup validation.
        await translations.insert(db, content_hash("bank"), "vi", "ngân hàng")
        with pytest.raises(InvalidResourceError):
            await translate_text(db, llm, content, "vi")
        assert llm.calls == 0
        assert len(await translations.list_rows(db)) == 1
    finally:
        await db.close()
