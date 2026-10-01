# v2 example

`v2_selected_word.py` demonstrates source-neutral search → user-selected
Cambridge handle → one neutral Word generation. `v2_question_server.py` shows
one saved Question generated in the exact neutral/Style namespace, projected
without its answer or explanations, then graded by saved option ID. Run only after migrating a
**generated** database and supplying a separate read-only Cambridge SQLite
snapshot:

```bash
export LEXI_V2_DB_URL='sqlite+aiosqlite:////absolute/path/generated.db'
export LEXI_V2_CAMBRIDGE_DB_PATH='/absolute/path/cambridge.db'
export LEXI_V2_NEUTRAL_DEFINITIONS=2
export LEXI_V2_NEUTRAL_EXAMPLES=3
export LEXI_V2_STYLED_DEFINITIONS=2
export LEXI_V2_STYLED_EXAMPLES=3
export LEXI_V2_DECISION_THRESHOLD=0.8
export OPENAI_API_KEY='your-provider-key'
uv run python examples/v2_selected_word.py bank
uv run python examples/v2_question_server.py 123        # trusted server, neutral Sense ID
uv run python examples/v2_question_server.py 123 pirate # Style must already exist
uv run python examples/v2_full_flow.py bank             # select → question → grade → WSD → cache
```

For Jev decisions set `TYPESAFE_API_KEY`, optionally
`LEXI_V2_DECISION_FALLBACK_MODEL`. The generated-database Alembic baseline is
under `lexi_ai_v2/` and must **not** run on Cambridge.

Older numbered scripts under `examples/` were authored for the historical
`lexi_ai` API; they are retained as reference, **not** runnable v2 examples.
Do not forward full Question artifacts to clients: they contain answer keys.

`v2_full_flow.py bank pirate` selects content in an existing Style; it never
falls back to neutral when that namespace is absent. Create the Style first
with `await ai.create_style("pirate", "Pirate", "a nautical voice")`. A new
available handle generates neutral content first and then that Style's content.
