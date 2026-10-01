# Decision design and runtime contract

Decisions use the seven domain-owned JSON-e templates listed below. The shared
renderer only loads/renders JSON-e and constructs SDK `Noul`/`Choice` questions;
domain code owns the stages, gates, SQL selection and final results. There are no
inline rubrics, old grading enum, compatibility adapters or task-specific logic
in `inference/` or `db/`.

The original supplied annotated designs are preserved in
[single-word-design.md](single-word-design.md),
[definition-usage-design.md](definition-usage-design.md) and
[relation-design.md](relation-design.md). This page describes the implemented
runtime contract rather than replacing those source designs.

## Single-word answers

1. `questions/prompts/decision/grade_single_word_1.json` independently evaluates
   `task_fit` and `spelling_error`. Task fit uses the question-type-specific rule;
   correcting a spelling-only error for that judgment does not correct grammar,
   particles, inflection or derivation. Spelling is not conditional on task fit.
2. Only when `task_fit=true` and `spelling_error=false`, search the submitted answer,
   take the top-ranked Word, and load **all** of that Word's neutral Senses.
3. `grade_single_word_2.json` selects the meaning expressed in the saved question,
   or `no_candidate`. No matching Word/Sense never changes task correctness.

Result: `SingleWordGrade(task_fit: bool, spelling_error: bool, sense_id: int | None)`.
Saved option IDs and exact normalized saved single-word answers retain their
provider-free path. Unknown option IDs are invalid submissions. Answers exceeding
the lexical search limit can still receive diagnostics but no dictionary mapping.

## Definition answers

1. `grade_word_to_definition_1.json` receives all neutral Senses of the Question's
   owner Word. Select the most clearly intended meaning even if the submitted
   definition is inaccurate, incomplete or vague. Lack of identifiable intent
   yields `no_candidate`, not a failed accuracy judgment.
2. If identified, `grade_word_to_definition_2.json` evaluates that meaning alone:
   `accuracy=accurate|mixed|inaccurate`, `coverage=sufficient|partial|minimal`.

Result: `DefinitionGrade(sense_id, accuracy, coverage)`. All three are null when
no meaning is identifiable. The selected Sense need not be the Question's original
Sense. An incomplete published inventory is an explicit resource error.

## Usage answers

1. `grade_word_to_usage_1.json` evaluates whether the answer uses the target Word.
2. Only if `used=true`, `grade_word_to_usage_2.json` evaluates separate diagnostics
   against the Word and meaning saved in the Question, not later dictionary edits:

| Diagnostic | Values |
|---|---|
| `meaning` | `correct`, `approximate`, `wrong` |
| `form` | `correct`, `spelling_error`, `form_error` |
| `construction` | boolean |
| `collocation` | `natural`, `acceptable`, `unnatural` |
| `appropriacy` | `appropriate`, `marked`, `inappropriate` |

Result: `UsageGrade(used, meaning, form, construction, collocation, appropriacy)`.
When `used=false`, all five diagnostics are null. These are independent judgments,
not a combined correctness enum or learner mastery score.

## Sense Linking

`relations/prompts/resolve_sense_relations.json` receives source meaning, relation
rule, target lemma/gloss and the **complete** same-POS neutral candidate inventory.
Candidate keys are anonymous `candidate_1`, `candidate_2`, etc.; code maps the
selected key to an internal numeric Sense ID after inference.

A candidate must satisfy **both** target-gloss compatibility and the relation
rule with the source meaning. Wording overlap alone is insufficient.

| Relation | Direction from source to target |
|---|---|
| `synonym` | Essentially the same lexicalized concept |
| `antonym` | Opposing meanings on the relevant semantic dimension |
| `hypernym` | Source is a kind/type of target |
| `hyponym` | Target is a kind/type of source |
| `meronym` | Target is a part/member/substance of source |
| `holonym` | Source is a part/member/substance of target |

No matching candidate yields `unresolvable`; unavailable target Words remain
`pending`. Incomplete neutral evidence, invalid responses and provider failures
(including request-size limits) produce per-edge `error` results and leave work
pending. They are never converted into negative verdicts or silently truncated.
An independent edge can still succeed when another fails.

Public `SenseRelation.resolution_state` is `pending`, `resolved` or `unresolvable`.
The operation remains `resolve_relations()`. Invalidation helpers are
`install_relation_triggers` / `remove_relation_triggers`, with SQL names prefixed
`lexi_relation_`. This naming replaces the former WSD terminology without changing
the linking behavior or adding compatibility aliases.

Inference runs outside write transactions. Apply locks the edge first, then reads
fresh evidence in a separate READ COMMITTED statement and revalidates the source,
relation, target and entire candidate inventory. Changed evidence produces `noop`.
Domain triggers invalidate affected decisions, including prior negative decisions;
themed and no-op edits do not. Candidates are loaded once per target/POS per page.

## Transport and confidence

`DecisionConfig.threshold` is an inclusive boundary for Noul truth and Choice
confidence. A low-confidence Choice invokes at most one configured fallback with
the **same** state/questions. Both primary and fallback responses must select
valid keys and provide finite probabilities in `[0,1]`. Transport failures are
errors, not semantic no-match outcomes. Noul-only gates do not request discarded
Choices or trigger Choice fallback.

Threshold and provider key/base URL/model are explicit constructor parameters.
The runtime does not read `.env` or environment variables; references to older env
names in the preserved supplied designs are historical, not configuration APIs.
Fallback uses `LLMConfig` credentials/base URL and the separate fallback model ID.

Opt-in `with_usage=True` on `grade_answer`, `resolve_relations` or `DecisionModel.decide`
returns `(normal_result, list[TokenUsage])`. Primary and fallback usage are retained,
grouped by actual response model ID, including all staged requests. Unknown counts
remain `None`; the TypeSafe SDK currently exposes input/output but no cache breakdown.
Relation errors/noops do not discard already reported inference costs. Transport
failures without metadata and hidden SDK retries cannot be reliably billed from
these counters alone; consumer pricing remains outside the library.

User/source strings are context data, never evaluated again as JSON-e templates.
Only static parsed templates are cached; no decision or dictionary snapshot cache
is introduced. Numeric lexicon IDs remain unchanged.

No calibration dataset has been supplied. Semantic calibration, provider capacity
and live-provider quality remain unverified; fake-provider tests verify the staged
contracts and database tests verify freshness, concurrency and query work.

## Question generation

`questions/schemas.py` defines exactly two output shapes: `AnchoredQuestionBatch`
for Definition to Word, Word to Definition and Context to Word, and `QuestionBatch`
for the other types. The first has `correct_explanation` but no `correct` field;
code attaches the dictionary answer. The second includes model-generated `correct`.
The structured schema controls output fields—there are no prompt instructions
telling the model not to generate `correct`, nor extra output-selection machinery.
