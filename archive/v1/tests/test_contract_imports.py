"""The public contract package must not initialize the runtime graph."""

import subprocess
import sys
from pathlib import Path


def test_contract_import_does_not_load_runtime_dependencies():
 project_root = Path(__file__).resolve().parents[1]
 code = """
import sys
import lexi_ai.questions.types

for module in (
 "sqlalchemy",
 "nltk",
 "lexi_ai.lexicon",
 "lexi_ai.dictionary",
 "lexi_ai.db",
):
 assert module not in sys.modules, module
"""

 subprocess.run(
  [sys.executable, "-c", code],
  cwd=project_root,
  check=True,
  capture_output=True,
  text=True,
 )
