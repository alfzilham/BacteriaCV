# Architecture and Security Boundaries

## Product and principals

BacteriaCVMobile is an Android Flutter application that lets its local user select one microscopy image, preview it, and run on-device classification. Source shows no user accounts, server, telemetry, persistence, or multi-user resource model. The local user is the only established principal. File-picker content is the main lower-trust input; the Android launcher is an additional component entry surface.

## Stack and active paths

The runtime path is Flutter `main()` -> app UI -> `FilePicker` -> in-memory preview -> `BacteriaClassifier` -> resize and bundled ONNX Runtime session -> result UI. Android launches a plain `FlutterActivity`; the inspected main manifest has only MAIN/LAUNCHER. Build-time paths include Gradle, Flutter dependencies, a GitHub Actions CI workflow, Python spike scripts, and a tracked pre-commit hook/encoding audit. The spike scripts use local environment and dataset paths; the hook runs repository tools in a developer shell when enabled. No comparable security baseline was source-grounded.

## Boundaries and controls

- File picker to preview: one local file is selected; the UI checks reported size, reads and decodes the path for preview, then retains path and preview in widget state. Classifier independently checks current file length before reading and decoding. The app source does not establish the picker plugin's URI grant, copy, or missing-size behavior.
- Classifier: local image bytes are resized and passed to a bundled model. No remote inference or network client was found in the reviewed runtime paths. Image decoder behavior and memory ceilings are not established by source alone.
- Android component: exported launcher activity has no source-visible custom deep-link or payload handler. Main manifest requests media read permissions; debug/profile manifests add INTERNET. Final merged release manifest was not built or inspected.
- Build and release: workflow runs CI checks/build; release Gradle configuration refers to debug signing. Hosted runner policy, repository protections, actual signing key, shipped artifact, and Play promotion controls are outside repository evidence.
- Developer tooling: spike scripts read local data/configuration and generate artifacts; the pre-commit hook runs tracked shell/Python tools under local Git hook configuration. These are separate from app runtime and CI trust boundaries.
- App data: the inspected runtime path keeps preview/classification state in memory; no persistent history, database, export, logging, or remote sink was found.

## Validation constraints

This run is source-only. The audit request retained the no-code-execution constraint, and this environment does not establish an empty allowlisted environment, disabled external networking, read-only target/toolchain mounts, scratch-only target writes, explicit resource limits, or race-safe Windows artifact promotion. No target-controlled build, test, fixture, or script was run. These controls are required before any local reproduction.

## Prior evidence and companion selection

No prior compatible security-audit ledger or findings file was present under the selected output root. The graph report is stale relative to the reviewed HEAD and is used only as navigation. Selected companions: `DESKTOP-MOBILE-AND-LOCAL-IPC.md` for picker, Android component, and local-file boundaries; `RESOURCE-EXHAUSTION-AND-AVAILABILITY.md` for image parsing and preview memory; `DATA-ISOLATION-AND-LIFECYCLE.md` for image/result retention; and `SUPPLY-CHAIN-AND-RELEASE.md` for dependency, CI, signing, and artifact handoffs. Browser, cloud, HTTP/auth, protocol, AI, and native memory-safety companions are excluded because reconnaissance found no corresponding source-visible surface; Android image decode internals remain outside source-only proof.
