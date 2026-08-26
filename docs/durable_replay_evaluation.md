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

## Durable replay result promotion eligibility

The read-only DurableReplayResultPromotionEligibilityService consumes an existing DurableReplayResultComparisonAssessment.

It returns the frozen DurableReplayResultPromotionEligibility model and preserves status, round_count, candidate_model_name, baseline_model_name, the three AX aggregate counts, and the window context.

The recommendation policy uses AX aggregate counts only:

- eligible: candidate_advantage_count > baseline_advantage_count and candidate_advantage_count >= 2
- ineligible: baseline_advantage_count > candidate_advantage_count
- insufficient_evidence: all remaining valid cases

The aggregate counts must be non-negative integers, bool is rejected, and candidate_advantage_count + neutral_count + baseline_advantage_count must sum to 9.

The window must be a mapping. The returned window is detached and read-only.

This layer does not reinterpret raw delta magnitudes, does not declare a winner, and does not publish or roll back a champion.

No production lifecycle mutation, CLI command, selector/discovery behavior, database persistence, or existing promotion.py modification is performed by this capability.

## Durable replay promotion action plan

The read-only DurableReplayResultPromotionActionPlanService consumes an existing DurableReplayResultPromotionEligibility.

It returns the frozen DurableReplayResultPromotionActionPlan model and preserves status, round_count, candidate_model_name, baseline_model_name, recommendation, and window context.

The action mapping is exact and uses AY recommendation only:

- eligible -> prepare_publish
- insufficient_evidence -> hold
- ineligible -> block

prepare_publish is a planning state only. It does not publish a champion.

The window must be mapping-compatible. The returned window is detached and read-only.

This capability does not access raw deltas or AX aggregate counts, does not execute publication or rollback, and does not mutate the champion registry.

No production lifecycle invocation, CLI command, artifact persistence, latest-result discovery, or cross-round search is introduced by this capability.

## Durable replay promotion publication request

The read-only DurableReplayPromotionPublicationRequestService binds a DurableReplayResultPromotionActionPlan to explicitly supplied publication identity.

The service accepts:

- action_plan
- source_decision
- registry_root

A request is allowed only when action equals prepare_publish.

The output DurableReplayPromotionPublicationRequest preserves status, round_count, candidate_model_name, baseline_model_name, recommendation, action, window, source_decision, and registry_root.

source_decision and registry_root are never auto-discovered. Empty or invalid publication identity values fail closed.

The window must be mapping-compatible and is returned as a detached read-only projection.

Direct composability with ProductionChampionRegistryPublisher.publish is verified because the request exposes the exact required publisher inputs: source_decision and registry_root, with matching str | Path annotations.

This capability does not publish, does not write the champion registry, does not invoke production lifecycle orchestration, and does not add a CLI command.

Actual production mutation remains owned by the existing production publication layer.

## Durable replay promotion publication execution

`DurableReplayPromotionPublicationExecutionService` is the reusable execution boundary after `DurableReplayPromotionPublicationRequest`.

The service consumes a `DurableReplayPromotionPublicationRequest` whose action is `prepare_publish` and delegates the exact explicit publication identity to `ProductionChampionRegistryPublisher.publish`:

- `source_decision` is forwarded from the request unchanged.
- `registry_root` is forwarded from the request unchanged.
- the existing `ProductionChampionPublicationResult` is returned directly and unchanged.

The execution adapter does not discover publication identity, recompute promotion policy, inspect raw deltas or AX aggregate counts, perform rollback, or add CLI behavior. Registry and filesystem mutation remain owned by `ProductionChampionRegistryPublisher`.

Malformed request identity is rejected before the publisher is invoked. Publisher exceptions are propagated rather than converted into success. The adapter does not duplicate registry write logic.

The real E2E validation uses an isolated temporary registry root. Publication is performed by the real publisher only inside that temporary location; the production registry remains untouched.

### Durable replay publication lifecycle adaptation

`DurableReplayPublicationLifecycleAdaptationService` is the reusable boundary between the typed durable-replay publication request/execution flow and the existing production lifecycle stage-result contract.

The service accepts a `DurableReplayPromotionPublicationRequest` and delegates publication execution exactly once to `DurableReplayPromotionPublicationExecutionService`. The BB execution service remains the execution owner, while `ProductionChampionRegistryPublisher` remains the registry-mutation owner.

The BC adapter returns the existing `ProductionLifecycleStageResult` rather than introducing another result model. The stage result uses `name = "publication"` and `status = "PASS"`.

Its `detail` contains exactly the five existing publication-result fields: `source_path`, `source_sha256`, `published_path`, `published_at_kst`, and `selected_model`.

The BC adapter does not call `ProductionChampionRegistryPublisher.publish` directly, does not call `run_publication_stage`, and does not own `publish-champion` CLI behavior. It does not discover publication identity, recompute eligibility or promotion policy, perform rollback, or duplicate registry-write logic.

Real E2E validation used an existing repository champion-decision artifact as read-only source input and published only to an isolated temporary registry. The resulting `ProductionLifecycleStageResult` preserved the five publication-result fields and source SHA-256 identity, while the production registry remained untouched.

### Durable replay publication lifecycle entrypoint

`DurableReplayPublicationLifecycleEntrypoint` is the additive typed operational
entrypoint for the durable-replay publication flow. It accepts an explicit
`DurableReplayPromotionPublicationRequest` and delegates that exact request,
unchanged, to `DurableReplayPublicationLifecycleAdaptationService`.

The entrypoint returns the existing `ProductionLifecycleStageResult` produced
by the BC adaptation layer directly and unchanged. On a successful publication
stage, the existing lifecycle result semantics remain `name = "publication"`
and `status = "PASS"` with the same five publication detail fields:
`source_path`, `source_sha256`, `published_path`, `published_at_kst`, and
`selected_model`.

This entrypoint is additive. It does not consume `argparse.Namespace`, does not
replace or call `run_publication_stage`, does not own `publish-champion` CLI
behavior, and does not call `ProductionChampionRegistryPublisher.publish`
directly. Publication mutation remains owned by
`ProductionChampionRegistryPublisher` through the existing BB execution and BC
adaptation layers.

The entrypoint performs only boundary validation before delegation: the input
must be a `DurableReplayPromotionPublicationRequest`, the action must remain
`prepare_publish`, and the BC result must be a
`ProductionLifecycleStageResult`. BC exceptions propagate unchanged. The
entrypoint does not discover publication identity, infer `source_decision` or
`registry_root`, recompute eligibility or promotion policy, perform rollback,
or duplicate lifecycle result construction or registry-write logic.

Real E2E verification uses an existing valid champion-decision fixture and an
explicit temporary registry. Publication mutation is confined to that temporary
registry, source SHA256 is verified, and the temporary root is removed after
the test; the production registry remains untouched.

### Durable replay publication invocation transport

`DurableReplayPublicationInvocationTransport` is the explicit presentation-safe carrier for a complete durable replay publication request. It preserves the exact nine request fields: `status`, `round_count`, `candidate_model_name`, `baseline_model_name`, `recommendation`, `action`, `window`, `source_decision`, and `registry_root`.

`DurableReplayPublicationInvocationTransportCodec` owns only deterministic transport adaptation:

- `encode(request)` converts a `DurableReplayPromotionPublicationRequest` into the immutable transport model.
- `to_mapping(transport)` returns a detached mapping suitable for a later presentation layer.
- `from_mapping(mapping)` requires the exact nine-field schema and fails closed on missing, unknown, malformed, or unsupported values.
- `decode(transport)` restores a BA-equivalent `DurableReplayPromotionPublicationRequest`.

Path values are preserved as caller-supplied text via `str(Path(...))` semantics. The transport does not normalize, resolve, discover, expand, or default `source_decision` or `registry_root`. The `window` is detached, top-level read-only after transport/decode, and recursively restricted to JSON-compatible content.

The transport is mapping-composable for future presentation wiring, but it does not own file I/O, stdin/stdout behavior, CLI parsing, `DurableReplayPublicationLifecycleEntrypoint` invocation, lifecycle orchestration, `ProductionChampionRegistryPublisher.publish`, rollback, eligibility recomputation, or promotion-policy recomputation.

The verified execution composition remains:

`DurableReplayPromotionPublicationRequest`
→ invocation transport encode/mapping/decode
→ `DurableReplayPublicationLifecycleEntrypoint`
→ lifecycle adaptation
→ publication execution
→ `ProductionChampionRegistryPublisher`

Real E2E verification uses only an isolated temporary registry. The production registry remains untouched by transport validation.

## Durable replay publication invocation JSON presentation

`DurableReplayPublicationInvocationJsonCodec` is the JSON-text presentation boundary above the closed BE invocation transport.

- The authoritative transport remains `DurableReplayPublicationInvocationTransport`.
- The JSON codec delegates structural and scalar validation to `DurableReplayPublicationInvocationTransportCodec`.
- Encoding uses the exact BE nine-field mapping and produces deterministic canonical JSON with sorted keys, compact separators, `ensure_ascii=False`, and non-finite numeric rejection.
- Decoding requires a JSON object root, rejects malformed JSON, rejects duplicate object keys recursively, and rejects `NaN`, `Infinity`, and `-Infinity`.
- Unicode values are preserved losslessly.
- The JSON layer does not normalize or reinterpret `source_decision`, `registry_root`, or `window`.
- The JSON layer does not own file I/O, stdin/stdout, CLI parsing, BD lifecycle invocation, `run_publication_stage`, or `ProductionChampionRegistryPublisher.publish`.
- Missing fields, unknown fields, scalar type validation, and window validation remain BE transport responsibilities.
- The BF validation E2E uses real repository decision evidence only as read-only fixture input. No temporary registry is created, and the production registry remains untouched.

### Durable replay publication invocation JSON file carrier

`DurableReplayPublicationInvocationJsonFileCarrier` is the physical-file presentation boundary for the durable replay publication invocation transport.

Ownership is intentionally narrow:

- `DurableReplayPublicationInvocationJsonFileCarrier` owns only the explicit physical file carrier and file-envelope validation.
- `DurableReplayPublicationInvocationJsonCodec` remains the owner of deterministic canonical JSON encoding and decoding.
- `DurableReplayPublicationInvocationTransportCodec` remains the owner of the exact nine-field transport structure and field-type validation.
- The carrier writes with exclusive create semantics, so an existing target fails closed instead of being overwritten.
- The carrier writes UTF-8 without a BOM and appends exactly one terminal LF to the canonical JSON payload.
- The carrier accepts an input file with no terminal newline, one LF, or one CRLF; a bare CR, a BOM, invalid UTF-8, or multiple trailing newlines fail closed.
- Parent directories are not created by this capability, and path text is not normalized, resolved, expanded, discovered, or defaulted.
- The carrier does not parse or reconstruct JSON fields itself; it delegates JSON syntax/canonicalization to `DurableReplayPublicationInvocationJsonCodec`.
- The carrier does not call `DurableReplayPublicationLifecycleEntrypoint`, `ProductionChampionRegistryPublisher.publish`, `run_publication_stage`, or any CLI surface.
- The carrier does not discover `source_decision` or `registry_root`, recompute eligibility or promotion policy, perform rollback, or mutate the production registry.
- Real end-to-end validation uses a temporary carrier file only; the production registry remains untouched.

### Durable replay invocation JSON file inspection CLI

The durable replay publication invocation stack exposes a read-only operator-facing inspection command in `lrp.cli.durable_replay_publication_invocation_json_file`.

The command accepts one explicit `--input` path and delegates physical carrier loading to `DurableReplayPublicationInvocationJsonFileCarrier`. The CLI does not perform its own filesystem I/O, path discovery, path normalization, environment expansion, default-path selection, or carrier mutation.

After BG returns a `DurableReplayPublicationInvocationTransport`, the CLI delegates presentation encoding to `DurableReplayPublicationInvocationJsonCodec` and writes exactly one canonical BF JSON payload plus one terminal newline to stdout. The transport remains the exact nine-field BE transport shape.

The CLI is intentionally read-only. It does not construct a transport from per-field flags, reconstruct a `DurableReplayPromotionPublicationRequest`, invoke `DurableReplayPublicationLifecycleEntrypoint`, call `run_publication_stage`, invoke `ProductionChampionRegistryPublisher`, mutate the production registry, recompute eligibility or promotion policy, perform rollback, or read transport input from stdin.

Failure ownership is preserved. Argument-shape failures belong to argparse; physical carrier failures remain owned by BG; JSON presentation failures remain owned by BF; BE retains structural and type validation. The CLI does not swallow or replace these failures.

This CLI remains independent from the existing `production-lifecycle`, `publish-champion`, and durable replay evaluation CLI surfaces. It is an additive inspection boundary, not a lifecycle replacement or a second publication path.
