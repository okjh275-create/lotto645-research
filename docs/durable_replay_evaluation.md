# Durable Replay Evaluation

## Purpose

`durable-replay-evaluation` evaluates durable replay inputs and writes the evaluation result to stdout as deterministic JSON.

The command supports two explicit input modes:

- explicit source mode
- selector mode

The modes are mutually exclusive. This document describes the current verified contract only. It does not define automatic artifact discovery, automatic model selection, manifest policy, latest-selection policy, or a production wrapper.

## Invocation

The repository-level invocation form is currently unresolved.

Do not assume that `python -m lrp` is the verified production entrypoint for this command.

Until the repository-level entrypoint is proven, use the following form only as a syntax template:

```text
<ROOT_CLI_ENTRYPOINT> durable-replay-evaluation
  --history <HISTORY_PATH>
  --window-name <WINDOW_NAME>
  --start-round <START_ROUND>
  --end-round <END_ROUND>
  --artifact-root <ARTIFACT_ROOT>
  --candidate-selector "ROUND_NO|MODEL_NAME[|REGIME_ID[|STRATEGY_NAME[|ARTIFACT_KEY]]]"
```

This template is not claimed runnable. Placeholder values must be replaced only with verified repository inputs.

## Shared Arguments

Both input modes use the same shared arguments:

- `--history`: explicit replay history input path
- `--window-name`: explicit evaluation window name
- `--start-round`: explicit first evaluation round
- `--end-round`: explicit last evaluation round

These values are forwarded to the downstream execution request. Their lower-layer semantic validation remains owned by the execution stack.

## Explicit Source Mode

Explicit source mode uses one or both of:

- `--candidate`
- `--baseline`

Each source descriptor is parsed into a `DurableReplayExecutionSource`.

The existing explicit source mode is preserved and executes through `DurableReplayExecutionRequest` and `DurableReplayExecutionService`.

Candidate-only and baseline-only explicit source shapes remain supported by the current CLI contract.

## Selector Mode

Selector mode uses:

- `--artifact-root`
- one or more `--candidate-selector` values and/or
- one or more `--baseline-selector` values

Each selector descriptor is parsed into a `DurableReplayArtifactSelector`.

Selector mode constructs a `DurableReplayCompositionRequest` and executes through `DurableReplayCompositionService`.

The composition layer delegates canonical artifact path projection to the discovery layer and operational replay execution to the execution layer.

## Selector Descriptor

The selector descriptor syntax is:

```text
ROUND_NO|MODEL_NAME[|REGIME_ID[|STRATEGY_NAME[|ARTIFACT_KEY]]]
```

Fields:

- `ROUND_NO`: decimal round number
- `MODEL_NAME`: explicit caller-supplied model identity
- `REGIME_ID`: optional provenance field
- `STRATEGY_NAME`: optional provenance field

The CLI does not infer model identity, candidate/baseline classification, regime identity, or strategy identity.


### Artifact key

`ARTIFACT_KEY` is an optional physical storage identity. It does not replace
`MODEL_NAME`, and it is not inferred from the model name.

Existing two- to four-field selector descriptors retain their prior meaning.
A fifth field supplies the artifact key:

```text
ROUND_NO|MODEL_NAME|REGIME_ID|STRATEGY_NAME|ARTIFACT_KEY
```

The artifact key uses the same explicit validation policy as keyed prediction
output: ASCII letters, digits, `.`, `_`, and `-`; maximum length 128; no path
separators, absolute path forms, empty value, `.` or `..`.

## Artifact Root

`--artifact-root` is the parent directory of `prediction-evaluation-sources/`.

For a selector round, the discovery layer derives the canonical artifact path:

```text
<artifact_root>/prediction-evaluation-sources/round_<NNNN>/evaluation_source.json

When `artifact_key` is explicit, the canonical keyed path is:

```text
<artifact_root>/prediction-evaluation-sources/round_<NNNN>/<artifact_key>/evaluation_source.json
```

When `artifact_key` is omitted, the legacy round-only path remains unchanged. Keyed lookup does not fall back to the legacy path, and legacy lookup does not scan keyed children.
```

`evaluation_source.json` existence is not checked by the CLI, composition layer, or artifact-discovery path projection layer. Artifact reading and validation belong to lower layers.

## Ordering

CLI order is preserved.

- candidate selector order is preserved
- baseline selector order is preserved
- duplicate selectors are preserved

The resulting candidate and baseline source tuples are forwarded in their preserved order.

## Mode Exclusivity

Explicit source mode and selector mode cannot be mixed in one invocation.

Do not combine:

- `--candidate` or `--baseline`

with:

- `--artifact-root`
- `--candidate-selector`
- `--baseline-selector`

There is no partial cross-mode merge and no filesystem-based mode inference.

## Output

Both modes return a `TopKReplayEvaluationResult`.

The CLI renders the result through the existing single serialization path and writes deterministic JSON to stdout.

There is no mode-specific output schema.

## Failure Behavior

Failure ownership remains layered:

- CLI descriptor syntax failures are argparse-owned failures
- mode-conflict failures are argparse-owned failures
- composition failures are propagated unchanged
- explicit execution failures are propagated unchanged

The CLI does not add an exception-normalization layer.

## Examples

The following repository-level selector-mode command shape is now verified.
The placeholder values remain placeholders and must be replaced with real paths,
rounds, model labels, and artifact keys:

```text
python -m lrp durable-replay-evaluation \
  --history <HISTORY_PATH> \
  --window-name <WINDOW_NAME> \
  --start-round <START_ROUND> \
  --end-round <END_ROUND> \
  --artifact-root <ARTIFACT_ROOT> \
  --candidate-selector "<ROUND>|<MODEL>|||<CANDIDATE_ARTIFACT_KEY>" \
  --baseline-selector "<ROUND>|<MODEL>|||<BASELINE_ARTIFACT_KEY>"
```



### Selector mode syntax template

The following is a syntax-only template. It is not a verified executable command because the repository-level entrypoint, real durable replay artifact, and real selector model identity have not yet been proven for a production invocation.

```text
<ROOT_CLI_ENTRYPOINT> durable-replay-evaluation
  --history <HISTORY_PATH>
  --window-name <WINDOW_NAME>
  --start-round <START_ROUND>
  --end-round <END_ROUND>
  --artifact-root <ARTIFACT_ROOT>
  --candidate-selector "ROUND_NO|MODEL_NAME[|REGIME_ID[|STRATEGY_NAME[|ARTIFACT_KEY]]]"
```

`--candidate-selector` and `--baseline-selector` are repeatable. Add only verified selector values and preserve the desired CLI order.

### Explicit source mode syntax template

The existing source descriptor parser uses an explicit artifact path, round number, model name, and optional provenance fields. Use the current CLI help and contract tests as the authoritative syntax source before constructing an operational invocation.

No fabricated history path, artifact path, round, model name, regime id, or strategy name is presented here as a real repository example.

## Non-Goals

This documentation does not introduce or promise:

- directory scanning
- latest artifact selection
- automatic candidate/baseline classification
- automatic model inference
- artifact persistence
- durable schema changes
- replay adaptation changes
- replay evaluation changes
- a new root command
- a production wrapper
- filesystem writes
- product-code changes

The current operational contract remains explicit-input and deterministic.

## Durable replay result artifact persistence

`durable-replay-evaluation` keeps its deterministic JSON stdout behavior by default.
Result persistence is opt-in through `--output`.

Example:

```powershell
python -m lrp durable-replay-evaluation `
  --history "artifacts/validation/project_r_release_readiness/r04_real_model_evaluation_e2e/history.json" `
  --window-name "round_1231" `
  --start-round 1231 `
  --end-round 1231 `
  --artifact-root "artifacts/validation/project_as_same_round_artifact_identity/as07_distinct_same_round_selector_e2e" `
  --candidate-selector "1231|candidate-model|||candidate-a" `
  --baseline-selector "1231|baseline-model|||baseline-a" `
  --output "artifacts/replay-results"
```

When `--output` is omitted, no result artifact is written and the existing stdout-only behavior is preserved.

When `--output` is supplied, the CLI persists the same JSON-compatible payload that is emitted to stdout by reusing the existing operation artifact writer.

Result artifact identity:

```text
<output_root>/
  durable-replay-evaluations/
    round_<END_ROUND>/
      evaluation_result.json
      manifest.json
  operation_log.jsonl
```

The storage partition uses `end_round`. One replay invocation produces at most one result artifact.
The manifest uses the existing operation artifact contract, including SHA256 and byte-count metadata.
The replay evaluation algorithm, source discovery, actual-draw projection, selector semantics, and `artifact_key` behavior are unchanged.

## Durable replay result artifact read-back

Durable replay result artifacts can be consumed through the dedicated read-only
`DurableReplayResultArtifactConsumer`.

The consumer identity is explicit:

```text
artifact_root + end_round
```

The consumer resolves exactly:

```text
<artifact_root>/
  durable-replay-evaluations/
    round_<END_ROUND>/
      evaluation_result.json
      manifest.json
```

Before reading `evaluation_result.json`, the consumer calls the existing
`verify_manifest` runtime verifier and requires a `PASS` verification status.
The consumer does not duplicate runtime SHA256 verification ownership.

On successful verification, the result JSON must contain a top-level object.
The payload is returned as a read-only mapping. The consumer does not mutate
the result artifact, manifest, or operation log.

The consumer does not perform latest-result discovery, sibling-round scanning,
cross-round search, candidate/baseline source selection, result rewriting,
production champion mutation, or replay algorithm changes.

Missing paths, manifest verification failures, invalid JSON, non-object payloads,
and wrong-round requests fail closed.

## Durable replay result artifact inspection

Persisted durable replay evaluation results can be inspected through the read-only
`DurableReplayResultArtifactInspectionService`.

The service accepts the existing
`DurableReplayResultArtifactConsumerRequest` identity:

- `artifact_root`
- `end_round`

It reuses `DurableReplayResultArtifactConsumer`, so manifest verification remains
mandatory before the result payload is accepted.

The inspection result is the frozen
`DurableReplayResultArtifactInspection` dataclass with exactly these fields:

- `status`
- `round_count`
- `candidate_model_name`
- `baseline_model_name`
- `evaluation`

The `evaluation` subtree is preserved as a read-only mapping. The inspection layer
does not derive new metrics, reconstruct source replay inputs, infer history windows,
or mutate the result artifact.

There is no auto-discovery, latest-result selection, cross-round scanning, source
artifact reconstruction, CLI command, or production lifecycle integration in this
capability. Missing or invalid artifacts remain fail-closed through the lower-layer
consumer contract.

## Durable replay result comparison summary

Persisted durable replay evaluation results can be summarized through the deterministic, read-only DurableReplayResultComparisonSummaryService.

The service accepts an existing DurableReplayResultArtifactInspection and returns the frozen DurableReplayResultComparisonSummary model.

The summary preserves the inspection identity/context fields:

- status
- round_count
- candidate_model_name
- baseline_model_name

It projects only the existing comparison deltas already present in evaluation.top3, evaluation.top5, and evaluation.top10:

- baseline_delta_mean_best_hits
- baseline_delta_3plus_rate
- baseline_delta_4plus_rate

This produces nine exact top-k delta fields in total: three metrics for each of top3, top5, and top10.

The existing evaluation.window mapping is copied into a detached read-only window projection.

The comparison summary does not create a winner or loser label, does not make a promotion recommendation, does not change ranking weights, and does not add discovery, selector, CLI, filesystem, database, or production lifecycle behavior.

Missing top-k blocks, missing frozen delta fields, non-mapping structures, or invalid numeric values fail closed rather than receiving synthetic defaults.

## Durable replay result comparison assessment

The deterministic, read-only DurableReplayResultComparisonAssessmentService consumes an existing DurableReplayResultComparisonSummary.

It returns the frozen DurableReplayResultComparisonAssessment model and preserves status, round_count, candidate_model_name, baseline_model_name, and window context.

Each of the nine existing AW comparison delta metrics is classified strictly by sign:

- positive value => candidate_advantage
- exact zero => neutral
- negative value => baseline_advantage

No epsilon band or custom threshold is applied. The three aggregate counts candidate_advantage_count, neutral_count, and baseline_advantage_count must sum to 9.

Validation is fail-closed: bool values are rejected as numeric evidence, NaN is rejected, non-numeric delta evidence is rejected, and window must be a mapping.

The assessment result is immutable and its window projection is detached and read-only.

This capability does not create a winner label, promotion recommendation, champion action, selector/discovery behavior, CLI command, database persistence, or production lifecycle mutation.
