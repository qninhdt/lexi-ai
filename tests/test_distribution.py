"""Check the wheel from a clean interpreter, not the editable checkout."""

import os
import subprocess
from pathlib import Path


def test_wheel_import_and_resources_from_isolated_install(tmp_path):
    root = Path(__file__).resolve().parent.parent
    wheel_dir = tmp_path / "dist"
    venv = tmp_path / "venv"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["uv", "venv", str(venv)], check=True, capture_output=True, text=True)
    python = venv / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(next(wheel_dir.glob("*.whl")))],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    code = """
import importlib.util
from pathlib import Path
from lexi_ai import DecisionConfig, Lexicon, LLMConfig
import lexi_ai
assert importlib.util.find_spec('lexi_ai_v2') is None
assert not hasattr(lexi_ai, 'LexiAI')
assert importlib.util.find_spec('lexi_ai.styles') is None
assert not hasattr(Lexicon, 'create_style')
assert not hasattr(lexi_ai, 'ContentCounts')
assert not hasattr(Lexicon, 'from_settings')
assert LLMConfig().base_url == 'https://api.openai.com/v1'
assert DecisionConfig(0.8).accepts(0.8)
assert Lexicon.__module__ == 'lexi_ai.api'
root = Path(lexi_ai.__file__).parent
assert (root / 'alembic.ini').is_file()
assert len(list((root / 'migrations' / 'versions').glob('*.py'))) == 2
assert (root / 'questions' / 'prompts' / 'generate_question.jinja').is_file()
assert not (root / 'prompts').exists()
for module in ('llm', 'decision', 'prompting'):
    assert importlib.util.find_spec('lexi_ai.' + module) is None
for module in ('questions', 'words', 'themes', 'translation', 'question_schemas', 'word_schemas'):
    assert importlib.util.find_spec('lexi_ai.inference.' + module) is None
from lexi_ai.questions.schemas import AnchoredQuestionBatch
from lexi_ai.words.schemas import WordOutput
from lexi_ai.translation.generate import translate_text
from lexi_ai.inference.prompting import render_prompt, render_decision
state, questions = render_decision('questions/prompts/decision/grade_single_word_1.json',
                                  question='question', answer='answer',
                                  question_type='cloze_to_word')
assert set(questions) == {'task_fit', 'spelling_error'}
system, user = render_prompt('questions/prompts/generate_question.jinja',
                            question_type='cloze_to_word',
                            theme=None, context={'word': 'bank'})
assert 'cloze' in system.lower() and 'bank' in user
system, user = render_prompt('questions/prompts/generate_question.jinja',
                            question_type='dialogue_completion',
                            theme=None, context={'word': 'bank'})
assert 'Target in Dialogue' in system
system, user = render_prompt('translation/prompts/translate.jinja',
                            target_language='vi', content='bank')
assert 'translator' in system and '<text>bank</text>' in user
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [str(python), "-c", code], cwd=tmp_path, env=env, check=True, capture_output=True, text=True
    )
