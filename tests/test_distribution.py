"""Build wheel from sdist and exercise installed resources outside the checkout."""

import os
import subprocess
import tarfile
from pathlib import Path


def test_sdist_wheel_and_installed_resources(tmp_path):
    root = Path(__file__).resolve().parent.parent
    wheel_dir = tmp_path / "dist"
    venv = tmp_path / "venv"
    subprocess.run(
        ["uv", "build", "--out-dir", str(wheel_dir)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    with tarfile.open(next(wheel_dir.glob("*.tar.gz"))) as archive:
        names = [Path(name).parts[1:] for name in archive.getnames()]
        assert all(
            parts[0]
            in {
                "lexi_ai",
                "pyproject.toml",
                "pyproject.toml.orig",
                "README.md",
                "LICENSE",
                "PKG-INFO",
            }
            for parts in names
            if parts
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
from pathlib import Path
from lexi_ai import Lexicon, migrations
import lexi_ai
assert Lexicon.__module__ == 'lexi_ai.api'
root = Path(lexi_ai.__file__).parent
assert (root / 'alembic.ini').is_file()
assert list(root.parent.glob('lexi_ai-*.dist-info/licenses/LICENSE'))
url = 'sqlite+aiosqlite:///' + str(Path.cwd() / 'installed.db')
migrations.upgrade_to_head(url)
assert migrations.inspect_current(url) == migrations.inspect_head()
from alembic import command
command.check(migrations.get_migration_config(url))
from lexi_ai.inference.prompting import render_prompt, render_decision
system, user = render_prompt('words/prompts/inventory.jinja', target='bank', references=[])
assert system.strip() and 'bank' in user
system, user = render_prompt('words/prompts/enrich_sense.jinja',
                            target='bank', word={}, sense={'definition': 'Money', 'pos': 'noun'},
                            examples_per_sense=1, references=[])
assert system.strip() and '<sense_request>' in user
state, questions = render_decision('questions/prompts/decision/grade_single_word_1.json',
                                  question='question', answer='answer',
                                  question_type='cloze_to_word')
assert set(questions) == {'task_fit', 'spelling_error'}
system, user = render_prompt('questions/prompts/generate_question.jinja',
                            question_type='cloze_to_word',
                            theme=None, context={'word': 'bank'})
assert system.strip() and 'bank' in user
system, user = render_prompt('questions/prompts/generate_question.jinja',
                            question_type='dialogue_completion',
                            theme=None, context={'word': 'bank'})
assert system.strip() and user.strip()
system, user = render_prompt('translation/prompts/translate.jinja',
                            target_language='vi', content='bank')
assert system.strip() and '<text>bank</text>' in user
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [str(python), "-c", code], cwd=tmp_path, env=env, check=True, capture_output=True, text=True
    )
