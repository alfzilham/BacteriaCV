# Security Audit Report

## Status

**INCOMPLETE - source-only review completed; audit validators were not run.** Run status is `incomplete` with reason `audit_schema_validators_not_run_due_to_no_code_execution_constraint`. The user requested no code execution. I therefore did not run project code, tests, builds, scripts, or the skill's validation programs. The final records and coverage ledger have not been checked by `validate-findings.cjs` or `validate-coverage-ledger.cjs`.

The parent-side files `findings.json`, `coverage-ledger.json`, and `run-metadata.json` were successfully parsed as JSON with PowerShell. The ASCII-range scan with `rg` returned no matches. These checks do not replace the skill validators. `git check-ignore -v` confirmed that the report directory is ignored by the `docs/report/` rule.

## Run information

- Profile: `standard`
- Scope: full repository source at `.`; current worktree contents were reviewed read-only
- Source ref: `c110ae5b283e035007da8f5689211051a225ea71`; worktree was dirty
- Target path in retained metadata: `.` relative repository root, to comply with the repository rule prohibiting absolute paths in committed documents
- Execution policy: source-only; no target-controlled process or network activity
- Prior audit: no compatible prior `coverage-ledger.json` or `findings.json` was present when this run started
- Agent invocations: 34 total; 4 reconnaissance, 14 hunters, 8 coverage critics, 4 candidate verifiers, 4 final record verifiers
- Strict agent budget: none was set
- Selected companion areas: mobile/local IPC, image resource use, data lifecycle, supply chain and release
- Coverage critics: final critic returned no missing units or reassignments after the last wave
- Graph context: both `graphify-out/graph.json` and the dated snapshot contain 497 nodes and 625 links (602 extracted, 23 inferred), built from `0cbfcc0de0d798f0a6b31551b6e9fde21d573afd`; both are stale relative to the reviewed HEAD and were used as navigation, not source evidence
- Documentation inventory: current `docs/` contains `docs/report/SPIKE-REPORT.md` and `docs/review/CODE_REVIEW.md`; Git reports `docs/CODEX.md`, `docs/SPIKE-REPORT.md`, and the task reports deleted in the dirty worktree. `docs/CODEX.md` was skipped per the user's instruction. The available spike report and review stub were read.
- Skills inventory: `.agents/skills/` was enumerated. The security-audit workflow and relevant mobile, resource, lifecycle, and supply-chain companion guidance were used.

This is not a complete validated audit. The source review covered the runtime picker/classifier path, Android manifests and entry activity, app data lifecycle, CI/dependency/release configuration, spike scripts, and developer hook/tooling. No local behavior, merged release manifest, hosted runner policy, device effect, or external cache behavior was measured.

## Security posture

No confirmed security vulnerabilities were established from repository source. Two source-grounded CI hypotheses remain `needs_validation` because decisive GitHub policy and action behavior are outside the repository. Other reviewed concerns were rejected as security findings where no lower-trust principal or meaningful cross-boundary impact was established; related resilience observations are recorded as hardening notes.

## Needs validation

| Fingerprint | Boundary | Current result |
|---|---|---|
| `pull-request-ci-token-authority` | Contributor-controlled pull request code to CI runner credentials and repository authority | `needs_validation`; effective token permissions, fork policy, and checkout credential behavior are not in source |
| `pr-ci-cache-poisoning` | Pull request cache state to a later trusted master workflow | `needs_validation`; action cache behavior and cross-event cache scope are not in source |

### Pull request CI authority

Source trace: `.github/workflows/ci.yml:6` enables `pull_request`; the test job checks out source at line 36 and runs tests at line 48. The analyze and Android build jobs also check out and execute commands at lines 18-30 and 54-73. The workflow does not declare token permissions. A contributor-controlled change could cross into repository authority only if external policy grants a write-capable token and checkout credentials are accessible to later commands. Neither condition is established here.

Safe resolution: using read-only owner access, inspect repository and organization Actions permission settings, fork pull-request policy, branch protections, and existing run summaries for effective permissions and checkout credential persistence. Do not create a test pull request, probe, or attempt a write.

### Pull request CI cache scope

Source trace: `.github/workflows/ci.yml:6` enables pull request runs; the Flutter setup action has `cache: true` in the analyze, test, and Android build jobs at lines 20-24, 38-42, and 61-65. The workflow does not define cache paths, keys, save/restore behavior, or cross-event isolation. Source does not show that a pull request can alter state later restored by a trusted master run.

Safe resolution: inspect the resolved `subosito/flutter-action` source and existing workflow/cache metadata using read-only access. Establish the cache paths, keys, save/restore behavior, and whether pull request state is available to a push-to-master run. Do not create a test pull request or modify a cache.

Neither item has a severity because no unauthorized result has been demonstrated.

## Confirmed findings

None.

## Hardening notes

- `lib/pipeline/classifier.dart:180-195` checks encoded file size before decoding, but source does not show a decoded-pixel limit before resize. The selected image is user supplied and source establishes no impact beyond the user's app process, so this was not reported as a security vulnerability.
- `lib/ui/upload_screen.dart:58-77` reads and decodes a selected image for preview. Confirm the resolved picker size/path contract before treating the reported file size as a memory bound.
- `android/app/build.gradle.kts:32-36` selects the debug signing configuration for release. Repository source does not establish actual signing-key ownership, a lower-trust promotion path, or a shipped artifact. Confirm the intended release signing identity before publishing.
- `.github/workflows/ci.yml` references actions by major-version tags. Pinning to immutable revisions is supply-chain hardening; no unauthorized execution or release path was established from source.
- `spike/scripts/export_test_reference.py:57-80` joins dataset row paths to a configured local root without a visible containment check. The dataset and environment are selected by the local operator; no lower-trust input or cross-principal read/write effect was established.
- `tools/tests/test_text_encoding_audit_sabotage.py:76-86` writes a backup to a predictable temporary path with ordinary file opening, then restores and removes that path. No lower-trust principal, protected target, or privilege boundary was established. Exclusive creation and symlink rejection would reduce local file-corruption risk.

## Positive source patterns

- Runtime classification uses a bundled model and on-device ONNX Runtime; no runtime network client, account, telemetry, or server path was found in the reviewed app source.
- The main Android manifest exposes only the launcher activity and contains no custom deep-link or data handler. The activity has no custom method channel or intent processing in the reviewed code.
- The classifier rejects encoded files over 20 MiB before its classification read.
- The inspected runtime path keeps preview and results in app memory and has no source-visible history database, export, logging, analytics, or remote sink.
- The inspected CI workflow has no source-visible secret reference, artifact publication, deployment, or release upload step.

## Coverage summary

The ledger has 11 units: 9 source-reviewed units are `covered` and 2 remain `candidate` pending the external facts described above. No units remain `planned`, `in_progress`, `blocked`, `deferred`, or `out_of_scope`. The final coverage critic accepted the coverage map with no additional units or reassignments. This statement is based on source-review and critic results; the coverage validator was not run, so it is not a schema-validated coverage claim.

No target code, build, test, fixture, or script was executed. The environment did not establish the required empty allowlisted environment, disabled external network, read-only target/toolchain, scratch-only target writes, explicit resource limits, or race-safe artifact promotion. The audit therefore cannot make runtime or deployed-behavior claims.
