# Needs Validation

These are source-grounded boundary hypotheses, not confirmed vulnerabilities. No severity is assigned. Do not probe a live workflow, create a test pull request, modify a cache, or attempt a write as part of resolving them.

## pull-request-ci-token-authority

- **Trace:** `.github/workflows/ci.yml:6` pull request trigger -> line 36 test-job checkout -> line 48 test execution. Analyze and Android build jobs also execute after checkout at lines 18-30 and 54-73.
- **Boundary hypothesis:** contributor-controlled pull request code may be able to use repository credentials available to the CI job.
- **Verified source facts:** the workflow has no explicit `permissions` block; it checks out source and runs commands in three jobs.
- **Blockers:** effective repository/organization `GITHUB_TOKEN` permissions; fork pull-request secrets and approval policy; branch protection; resolved `actions/checkout@v4` credential persistence. These facts are outside repository source.
- **Safe owner-observed check:** inspect repository and organization Actions settings and existing pull request run summaries for actual permissions, fork policy, branch protection, and whether checkout credentials persist for later commands. Use read-only access only.

## pr-ci-cache-poisoning

- **Trace:** `.github/workflows/ci.yml:6` pull request trigger -> Flutter setup action at lines 20-24, with equivalent cache settings at lines 38-42 and 61-65.
- **Boundary hypothesis:** a pull request may influence cached toolchain state that a later trusted master workflow restores.
- **Verified source facts:** all three jobs set `cache: true`; the workflow source does not define cache paths, keys, lifecycle, or event separation.
- **Blockers:** resolved `subosito/flutter-action` version and its cache implementation; GitHub Actions cache access and namespace behavior across pull request and push-to-master runs.
- **Safe local check:** inspect the resolved action source without running the workflow to document cache paths, keys, and save/restore behavior.
- **Safe owner-observed check:** read existing workflow-run/cache metadata and applicable cache scope/access policy to establish whether pull request state can be restored by a trusted master run. Do not create a pull request or change a cache.
