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
  --candidate-selector "ROUND_NO|MODEL_NAME[|REGIME_ID[|STRATEGY_NAME]]"
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
ROUND_NO|MODEL_NAME[|REGIME_ID[|STRATEGY_NAME]]
```

Fields:

- `ROUND_NO`: decimal round number
- `MODEL_NAME`: explicit caller-supplied model identity
- `REGIME_ID`: optional provenance field
- `STRATEGY_NAME`: optional provenance field

The CLI does not infer model identity, candidate/baseline classification, regime identity, or strategy identity.

## Artifact Root

`--artifact-root` is the parent directory of `prediction-evaluation-sources/`.

For a selector round, the discovery layer derives the canonical artifact path:

```text
<artifact_root>/prediction-evaluation-sources/round_<NNNN>/evaluation_source.json
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

### Selector mode syntax template

The following is a syntax-only template. It is not a verified executable command because the repository-level entrypoint, real durable replay artifact, and real selector model identity have not yet been proven for a production invocation.

```text
<ROOT_CLI_ENTRYPOINT> durable-replay-evaluation
  --history <HISTORY_PATH>
  --window-name <WINDOW_NAME>
  --start-round <START_ROUND>
  --end-round <END_ROUND>
  --artifact-root <ARTIFACT_ROOT>
  --candidate-selector "ROUND_NO|MODEL_NAME[|REGIME_ID[|STRATEGY_NAME]]"
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
