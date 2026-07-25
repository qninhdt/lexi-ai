# CEFR Sense Classification Experiment

**Status:** brainstorm approved; implementation not started  
**Date:** 2026-07-24  
**Goal:** fine-tune and evaluate a local sense-level CEFR classifier, demonstrate DL-for-NLP and practical MLOps, then integrate the accepted model into Lexi-AI.

## 1. Problem

Predict the CEFR level of an English word sense:

```text
Input:  target + POS + definition + context examples
Output: A1 | A2 | B1 | B2 | C1 | C2
```

CEFR is predicted at **sense level**, not lemma level. A lemma may contain basic and advanced meanings; the local source has 2,254 lemmas spanning multiple CEFR levels.

Primary research question:

> Do semantic sense features (definition, POS, context) predict CEFR better than word identity and frequency for lemmas unseen during training?

This is stronger than a random example split because it tests semantic generalization rather than memorization.

## 2. Decisions

1. **Phase 0 precedes data implementation:** lock task, hypotheses, split, metrics, and baselines first.
2. **Primary data uses core examples only:** `examples.is_extra = 0`.
3. **Primary evaluation uses unseen-lemma split.** Known-lemma evaluation remains secondary.
4. **Store canonical normalized tables, not pre-rendered prompt variants.** Training materializes feature views from the same rows and split.
5. **Production input is full sense input.** Paper-compatible `target + context` is a baseline.
6. **MLOps stack:** Git + DVC + W&B; Hugging Face Hub only for accepted public releases.
7. Do not add MLflow alongside W&B. Defer Airflow, Kubeflow, Feast, Evidently, and Kubernetes.

## 3. Source Audit

Source database: root file `./data` (SQLite, local-only).

```text
SHA-256: 147a99e1d723671a6488100d7742783c45fb1e254a515d834709442cb963b3a4
```

| Item | Count |
|---|---:|
| CEFR-labeled senses | 16,817 |
| Distinct lemmas | 10,201 |
| All contextual examples | 93,661 |
| Core examples (`is_extra=0`) | 42,151 |
| Extra examples (`is_extra=1`) | 51,510 |
| Senses covered by core examples | 16,153 |
| Lemmas spanning multiple CEFR levels | 2,254 |

CEFR sense distribution:

| Level | Senses | All examples | Core examples |
|---|---:|---:|---:|
| A1 | 855 | 7,585 | 3,670 |
| A2 | 1,710 | 11,916 | 5,083 |
| B1 | 3,265 | 21,620 | 9,159 |
| B2 | 4,698 | 28,128 | 11,501 |
| C1 | 2,497 | 10,858 | 5,232 |
| C2 | 3,792 | 13,554 | 7,506 |

### Extra-example defect

Extra examples are unsafe as sense-level supervision:

- 8,146 duplicate groups by `(lemma, context)`.
- 5,001 groups assign different CEFR labels to the same `(lemma, context)`.

Core examples contain only eight duplicate groups and two conflicting `(lemma, context)` groups. Therefore, extra examples are excluded from the primary dataset. They may become an auxiliary corpus only after reliable sense reassignment.

### Target alignment

A preliminary substring check finds a headword or known inflection in 89.39% of core examples. This is an upper bound because substring matching can create false positives. Coverage also decreases from 97.08% at A1 to 81.92% at C2; filtering unmatched rows may bias the label distribution.

The real aligner must use Unicode-aware boundaries, longest matching, and the entry's inflection list. Every context records one of:

```text
exact | inflection | multiple | missing
```

Unmatched examples remain in the canonical dataset. Marker-dependent feature views may select the aligned subset and must report resulting class drift.

## 4. Experiment Contract

### Prediction unit

The label belongs to a sense. Contexts are observations of that sense. Primary metrics are computed after aggregating context logits to one prediction per sense.

### Feature views

All views use the same canonical rows and split:

| View | Input | Purpose |
|---|---|---|
| `majority` | None | Sanity baseline |
| `lookup` | Known lemma/sense | Measure memorization ceiling |
| `frequency-only` | External lemma frequency | Test frequency hypothesis |
| `target-only` | Lemma | Measure word-identity memorization |
| `contextual` | Target + one context | Reproduce paper-style setup |
| `sense-only` | Target + POS + definition | Dictionary-sense classifier |
| `full` | Target + POS + definition + 1–3 contexts | Production candidate |

### Metrics

Primary:

- Sense-level Macro-F1 on unseen lemmas.

Secondary:

- Quadratic Weighted Kappa.
- MAE after mapping `A1=0` through `C2=5`.
- Adjacent accuracy (absolute level error ≤ 1).
- Per-class precision, recall, and F1.
- Confusion matrix.
- Calibration/error by POS, entry type, and target-alignment status.
- Bootstrap confidence intervals grouped by lemma.

### Success condition

The full model should outperform the strongest non-semantic baseline (`frequency-only` or `target-only`) on the unseen-lemma test set. Prefer a bootstrap confidence interval for the metric difference rather than an arbitrary fixed point threshold.

## 5. Canonical Data Model

### `senses.parquet`

One row per labeled sense:

```text
sense_uid
source_sense_id
lemma
lemma_key
headword
pos
entry_type
definition
guideword
grammar
domain
cefr_label
cefr_index
label_source
source_db_sha256
```

`sense_uid` is a content-derived stable hash, not only a SQLite auto-increment ID.

### `contexts.parquet`

One row per core example:

```text
context_uid
sense_uid
raw_text
normalized_text_hash
example_order
collocation
target_surface
target_start
target_end
alignment_status
alignment_method
```

Raw text is preserved for model input. Normalized text exists only for keys, deduplication, and leakage checks.

### `splits.parquet`

```text
sense_uid
split_regime
split
split_version
seed
```

### `quarantine.parquet`

```text
record_uid
reason
details
```

Quarantine reasons include conflicting labels, invalid CEFR values, empty required fields, duplicate same-sense examples, and ambiguous source mappings. Known conflicts are not resolved heuristically.

### Manifests and reports

```text
source-manifest.json
dataset-manifest.json
data-quality.json
dataset-card.md
```

The manifests record source hash, Git revision, schema version, code/config hashes, row counts, class distributions, exclusion counts, and artifact hashes.

## 6. Data Pipeline

```text
fingerprint source
       ↓
extract senses + core contexts
       ↓
normalize stable keys and POS
       ↓
validate schema and relationships
       ↓
align target surfaces
       ↓
deduplicate and quarantine conflicts
       ↓
create grouped splits
       ↓
materialize feature views
       ↓
push private DVC artifacts
```

Suggested DVC stages:

```text
fingerprint → extract → validate → align → split → materialize
```

A stage reruns only when its dependencies or parameters change.

## 7. Splits and Leakage Controls

### Primary: unseen lemma

- Group key: project-normalized `lemma_key`.
- Initial ratio: 70/15/15.
- Grouped stratification approximates the CEFR distribution across splits.
- No lemma may cross train, validation, and test.

### Secondary: known lemma, unseen sense/context

- Group by `sense_uid`; all contexts of one sense remain in one split.
- A lemma may occur across splits through different senses.
- Measures the easier product scenario for vocabulary already represented during training.

### Sentence contamination

The same source sentence may annotate multiple target words. After splitting:

1. Report identical sentence hashes crossing splits.
2. Build a strict test subset excluding any sentence observed in train or validation.
3. Report standard and strict results separately.

### Context weighting

When training one row per context:

```text
sample_weight = 1 / number_of_contexts_for_sense
```

This prevents senses with many examples from dominating the objective.

## 8. MLOps Architecture

### Git

Tracks code, schemas, configs, manifests, lock files, reports, and model cards. It never tracks the Cambridge source DB, secrets, or private dataset artifacts.

### DVC

Tracks private Parquet artifacts and pipeline lineage. The Cambridge source stays local; Git stores only its fingerprint and metadata. Derived Cambridge text remains private until redistribution rights are verified.

Remote decision:

- **Cloudflare R2 (S3-compatible)** is the private DVC remote.
- Rationale: works from local, Kaggle, and Colab; provides production-like object-storage practice without coupling experiment tracking to the storage vendor.
- Credentials remain local/secret-managed and must never enter Git, DVC metadata, notebooks, or W&B configs.

### W&B

Each training run records:

```text
git_sha
source_db_sha
dvc_dataset_hash
split_id
config_hash
seed
base_model_revision
hyperparameters
metrics
confusion matrices
checkpoints
error-analysis tables
```

W&B tracks runs and candidate models. DVC remains the dataset source of truth; do not version the same full dataset independently in W&B.

### Hugging Face Hub

Only accepted models are published with tokenizer, inference example, model card, data provenance, limitations, and metrics for both split regimes.

### CI

GitHub Actions runs lightweight checks against synthetic fixtures:

- Data transform unit tests.
- Tabular schema validation.
- Split leakage tests.
- DVC definition validation.
- Lint/type/test commands.

Full extraction and GPU training are not CI jobs because the source data is private and GPU work is expensive.

### Reproducibility identity

Every result must be recoverable from:

```text
git SHA
+ source DB SHA
+ DVC dataset hash
+ split version
+ config hash
+ random seed
+ base model revision
```

## 9. Execution Flow

### Local CPU phase

```text
source DB → dvc repro → validate reports → dvc push → git push
```

### Kaggle/Colab GPU phase

```text
git clone → install pinned environment → dvc pull → training CLI → W&B run → candidate checkpoint
```

Notebooks are launchers and EDA surfaces, not the source of truth for transformations or training logic.

## 10. Remaining Phases

### Phase 2 — Baseline Experiments

Phase 2 establishes how much CEFR signal comes from class priors, frequency, lemma surface, sparse text features, and frozen semantic representations before any end-to-end transformer fine-tuning.

#### Baseline ladder

| ID | Features | Estimator | Purpose |
|---|---|---|---|
| B0 | None | Global majority | Sanity floor and metric check |
| B1 | Lemma Zipf frequency + OOV flag | Ridge and multinomial logistic regression | Test the frequency hypothesis |
| B2 | Character n-grams, length, token count, POS, entry type; optional frequency | Linear classifier | Strongest non-semantic lexical baseline |
| B3a | Target + POS + definition TF-IDF | LinearSVC | Definition-only sparse semantic signal |
| B3b | Target + context TF-IDF | LinearSVC | Paper-compatible context-only signal |
| B3c | Target + POS + definition + contexts TF-IDF | LinearSVC | Full sparse semantic baseline |
| B4 | Frozen MiniLM full-sense embedding | Logistic regression or SVC | Strong frozen sentence-encoder baseline |
| B5 | Frozen BERT target-token contextual embedding | SVC | ME6-inspired paper baseline |
| B6 | Empirical lemma-level label distribution | Lookup with prior fallback | Known-lemma product diagnostic |

B0–B5 run on unseen-lemma and known-lemma regimes where applicable. B6 runs only on known-lemma evaluation because every primary-test lemma is intentionally unseen.

#### Feature construction

Frequency features must record the external frequency resource, package/data version, OOV behavior, and MWE aggregation rule. Frequency-only and frequency-plus-surface runs remain separate so their effects are identifiable.

Frozen MiniLM serializes one sense as:

```text
Word: {lemma}
Part of speech: {pos}
Definition: {definition}
Examples:
- {example_1}
- {example_2}
- {example_3}
```

The encoder is not updated. Embeddings are computed once, keyed by `sense_uid`, and cached by DVC.

The ME6-inspired baseline marks a reliable target span, extracts frozen BERT hidden states for its subword tokens, averages those vectors, and trains an SVC. Context decision scores are averaged by `sense_uid` before evaluation. Until the implementation is verified against the original ME6 code/method, results must be labeled **ME6-inspired**, not an exact reproduction.

Because target alignment coverage decreases at advanced levels, B5 reports coverage and CEFR distribution before and after filtering. Its score is not compared to full-dataset systems without that qualification.

#### Training protocol

- Freeze `dataset_id`, split IDs, and feature schemas before runs begin.
- Fit vectorizers, scalers, priors, and estimators on train only.
- Select hyperparameters using validation only.
- Use a small auditable grid (for example `C ∈ {0.1, 1, 10}` and class weighting on/off), not broad HPO.
- Keep the final test sealed after selecting one configuration per baseline family; lock validation results and configs for Phase 4.
- Use sense-balanced context weights: `1 / context_count_for_sense`.
- Aggregate context scores into one sense prediction before primary metrics.
- Keep standard and strict decontaminated test results separate.

Every baseline reports results for all unseen lemmas, strict decontaminated examples, multi-level lemmas, single-level lemmas, CEFR class, POS, entry type, and target-alignment status where applicable. The multi-level-lemma subset is the critical test of whether semantic features add value beyond lexical frequency.

#### Phase 2 MLOps

DVC/R2 owns deterministic feature caches and result tables:

```text
features/frequency-v1.parquet
features/minilm-sense-v1.parquet
features/bert-context-v1.parquet
reports/baseline-results-v1.parquet
reports/baseline-errors-v1.parquet
```

Embedding manifests include encoder/tokenizer revisions, pooling strategy, maximum length, dataset hash, and feature schema version.

W&B uses one project with grouped baseline runs:

```text
project: lexi-cefr
group: phase-2-baselines
job_type: baseline
```

Each run records baseline ID, feature view, lineage tuple, hyperparameters, coverage, runtime, memory, validation metrics, confusion matrices, and validation error-analysis tables. DVC remains the dataset/feature source of truth; W&B stores fitted baseline models and validation artifacts, not the private Cambridge-derived dataset. Final-test metrics are added only in Phase 4.

#### Phase 2 deliverables

```text
baseline-results.parquet
baseline-summary.json
baseline-comparison.md
error-analysis.parquet
best-nonsemantic-baseline/
best-frozen-baseline/
```

#### Exit and promotion gate

Phase 2 completes only when:

- Every run references the same frozen dataset and split versions.
- Train-only preprocessing and validation-only model selection are verified.
- Unseen-lemma validation, strict-validation, and multi-level subset results are available; final-test predictions remain sealed.
- The strongest non-semantic and frozen semantic baselines are identified.
- Frequency contribution and target-alignment coverage are quantified.
- Feature caches have reproducible hashes and exist in the private R2 remote.
- W&B runs contain complete lineage and comparison artifacts.

Phase 4 promotion requires the locked Phase 3 finalist to outperform the stronger of the best non-semantic and frozen semantic baselines on the unseen-lemma strict test. Prefer a grouped bootstrap confidence interval for the metric difference rather than declaring success from a small point-score gain.

If TF-IDF beats frozen encoders, investigate Cambridge-specific wording cues. If frequency nearly matches semantic systems, center Phase 3 on multi-level and unseen lemmas. If frozen embeddings are already strong, fine-tuning still needs to justify itself through statistically credible quality, calibration, or deployment gains.

### Phase 3 — Model-Agnostic Fine-Tuning

Phase 3 fine-tunes an encoder classifier without coupling the experiment framework to a fixed backbone. Backbone IDs and revisions are configuration values; data, objectives, metrics, tracking, and promotion logic remain unchanged when a newer encoder is substituted.

#### Model contract

```yaml
model:
  id: "<huggingface-encoder-id>"
  revision: "<immutable-commit>"
  max_length: 256
  trust_remote_code: false

objective:
  type: cross_entropy
```

The trainer resolves supported encoders through Hugging Face `AutoTokenizer` and `AutoModel`/`AutoModelForSequenceClassification`. Model-specific behavior belongs in a small adapter only when the generic contract cannot represent the required pooling or head. Decoder LLMs and parameter-efficient tuning are out of scope initially; full encoder fine-tuning is feasible for approximately 16.8K sense samples.

#### Training unit and input

The primary production dataset has one sample per sense:

```text
Word: {lemma}
Part of speech: {pos}
Definition: {definition}
Examples:
1. {core_example_1}
2. {core_example_2}
3. {core_example_3}
```

Examples are selected deterministically by source order, with at most three core examples. Definition text has truncation priority over examples. Missing examples do not remove an otherwise valid labeled sense. No CEFR-bearing metadata, source IDs, frequency values, extra examples, or synthetic augmentation enter the main model input.

Use normal text and tokenizer separators rather than adding custom vocabulary tokens. Before training, profile token lengths for every shortlisted tokenizer and choose the smallest standard limit that preserves at least 99% of complete definitions, normally bounded at 512. The selected limit and truncation rate by CEFR level are versioned artifacts.

#### Objectives

Train objectives in increasing complexity:

1. **Six-class cross-entropy** — required reference objective.
2. **Class-balanced cross-entropy** — weights computed from train only; no default oversampling.
3. **CORAL-style ordinal regression** — five cumulative thresholds for ordered A1–C2 predictions, implemented with a rank-consistent shared score/ordered-threshold head.

Plain MSE regression, focal loss, broad multi-objective losses, and label smoothing are deferred until evidence identifies a specific failure. Primary selection remains validation Macro-F1; QWK, MAE, and adjacent accuracy break close ties. Do not collapse metrics into an arbitrary weighted score.

#### Training defaults

```yaml
optimizer: adamw
learning_rate: [1e-5, 2e-5, 3e-5]
weight_decay: 0.01
warmup_ratio: 0.1
max_epochs: 6
early_stopping_patience: 2
effective_batch_size: 32
gradient_clip_norm: 1.0
precision: bf16_if_supported_else_fp16
```

Use dynamic padding and save the best validation checkpoint. Standard attention is the comparison default; FlashAttention, unpadding optimizations, and quantization belong to Phase 5. Final configurations run at least three seeds and report mean, standard deviation, hardware, CUDA/PyTorch/Transformers versions, precision, and attention implementation. Exact bitwise equality across different cloud GPUs is not required, but lineage and metric distributions must be reproducible.

#### Experiment funnel

1. **Sanity:** overfit 64–128 examples; run a one-epoch subset smoke test; verify save/load parity.
2. **Backbone screen:** compare a small number of current encoder candidates using one seed, full input, standard CE, and identical settings.
3. **Objective screen:** on the leading backbone, compare CE, balanced CE, and CORAL with one seed.
4. **Seed confirmation:** run the top two configurations with three seeds.
5. **Limited input ablation:** compare sense-only versus full input on the winning configuration; run a paper-compatible contextual variant only if Phase 2 evidence justifies it.

Do not evaluate a full Cartesian product of backbones, objectives, inputs, and hyperparameters. The target budget is approximately 10–15 GPU hours, with a hard review stop at 20 hours.

#### Sealed-test policy

All Phase 2 and Phase 3 model/configuration selection uses train and validation only. The unseen-lemma final test and strict decontaminated subset remain sealed until Phase 4. Phase 2 baselines and Phase 3 finalists are locked before final predictions are generated. This prevents human iteration on final-test outcomes.

#### Phase 3 MLOps

W&B groups runs under:

```text
project: lexi-cefr
group: phase-3-finetuning
```

Every run logs dataset/split IDs, Git and DVC hashes, exact model/tokenizer revisions, objective, input view, length/truncation report, hyperparameters, random seed, hardware/software environment, curves, validation metrics, runtime, throughput, peak VRAM, logits, and checkpoints.

DVC/R2 remains the source of truth for datasets, split manifests, and static tokenization reports. W&B owns training runs, resumable/best checkpoints, validation predictions, and candidate model artifacts. Checkpoint aliases are assigned only after multi-seed confirmation.

#### Phase 3 deliverables

```text
backbone-screen.json
objective-screen.json
seed-stability.json
input-ablation.json
locked-finalists.json
validation-predictions.parquet
candidate-checkpoints/
```

#### Exit gate

Phase 3 completes when:

- A model-agnostic training path works for at least two compatible encoder backbones.
- Tiny-set overfit, subset smoke, and checkpoint reload checks pass.
- One backbone/objective/input configuration is selected using validation only.
- Finalists have multi-seed stability results and complete lineage.
- Tokenization/truncation behavior is audited by CEFR level.
- Locked baseline and fine-tuned finalists are registered for Phase 4.
- No final-test labels, predictions, or error examples have been inspected.

### Phase 4 — Evaluation

Run ablations, strict decontaminated testing, grouped bootstrap confidence intervals, error analysis, and calibration. Do not report only one aggregate score.

### Phase 5 — Optimization and registry

Measure model size, CPU/GPU latency, throughput, and memory. Quantize/package only after model quality is accepted. Promote candidates through W&B registry and publish the final release to Hugging Face.

### Phase 6 — Lexi-AI integration

Add local inference behind a project-owned protocol, preserve existing behavior as fallback during rollout, and compare quality, latency, and avoided LLM calls.

### Phase 7 — Monitoring

Only after real usage exists: monitor prediction distribution, confidence, missing-input rates, drift, and disagreement with Cambridge labels or audited samples.

## 11. Phase 1 Completion Criteria

- Deterministic rebuild produces identical artifact hashes.
- Every final label is one of A1–C2.
- No required definition is empty.
- Conflicts are quarantined and documented.
- No lemma or sense leakage under its declared split regime.
- Sentence overlap is measured; strict test set exists.
- Split/class/alignment distributions are reported.
- Source, code, config, and artifact lineage is complete.
- Cambridge source and derived text remain private.
- Dataset card documents provenance, exclusions, limitations, and license uncertainty.

## 12. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Model memorizes lemma/frequency | Inflated result | Unseen-lemma split + target/frequency baselines |
| Extra examples are mislabeled at sense level | Label noise | Exclude `is_extra=1` from primary data |
| Target alignment filtering removes more advanced samples | Class bias | Keep canonical rows; report alignment by level |
| Context count differs by sense | Loss weighting bias | Sense-balanced sample weights |
| Same sentence appears across target words/splits | Context leakage | Overlap report + strict test subset |
| Cambridge redistribution restrictions | Cannot publish dataset | Private DVC remote; publish code/manifests only |
| MLOps stack becomes the project | Delivery slowdown | DVC + W&B only; defer orchestration/monitoring tools |

## 13. Next Steps

1. Create a detailed implementation plan for Phase 0 and Phase 1.
2. Configure Cloudflare R2 as the private DVC remote using secret-managed credentials.
3. Implement and validate the source fingerprint, canonical extraction, and quarantine reports.
4. Freeze `dataset-v1` and both split manifests before baseline training.
5. Execute the Phase 2 baseline ladder before selecting the Phase 3 fine-tuning architecture.

## Unresolved Questions

- Can derived Cambridge definitions/examples be redistributed? Until verified, treat them as private.
