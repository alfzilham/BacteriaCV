# AGENTS.md: Instructions for Agents in this Repository

> **STATUS 12 Oktober 2026.** Repo ini dianggap **selesai** oleh owner.
> Fokus pengerjaan pindah ke `BacteriaCVMobile`, yang dikerjakan dengan satu
> sesi Codex CLI. Lihat `docs/CODEX.md`.
>
> Dokumen ini tetap berlaku sebagai rujukan keputusan, dan untuk pekerjaan
> web di masa depan. Isinya jangan dihapus.

## 1. Roles
| Agent | Tool | Role | Authority |
|------|------|-------|------------|
| Codex CLI | OpenAI Codex CLI | Author and auditor in one session | Writes and modifies code according to SPEC, ARCHITECTURE, DESIGN, then verifies its own work against the rules in section 4 |
| Antigravity CLI | Antigravity IDE 1.107.0, chat mode --mode agent | Interface author | Writes and modifies app/static/index.html and docs/DESIGN.md according to DESIGN.md and after owner approval |

Codex CLI works alone here. It writes the code and then verifies that same
code, so it is not an independent auditor. Section 4 exists because of that,
and every rule in it is mandatory.

The industrial brutalist interface redesign was carried out by Antigravity CLI
version 1.107.0.

The Codex CLI version number is not recorded because `codex --version` was
unavailable on the machine where the later reviews ran.

## 2. Sources of truth
Priority order when a conflict arises:
1. SPEC.md
2. ARCHITECTURE.md
3. DESIGN.md
4. PRD.md and CONTEXT.md (draft status until validated)
5. AGENTS.md
6. Existing code

An agent must not change a decision recorded in SPEC.md without explicit approval from the project owner.

## 3. Rules for the author
1. Do one task per change. Never combine a refactor with a new feature.
2. Follow the interface in DESIGN.md. If the interface must change, write up the proposal and wait for approval.
3. Never change the lookup table [L] or the class weighting rules without approval.
4. Never use test data for any decision during training.
5. Never store credentials, tokens, or absolute paths in code.
6. Every finished task must come with tests according to DESIGN.md Section 5.
7. When done, write the change summary and the file list for the project owner.
8. The summary must be complete enough that the owner can judge the task without guessing its scope.
9. Never declare a task finished before the verification in section 4 has actually been run.
10. Route every question to the document that owns the answer before acting on it.

    | Question about | Authoritative document |
    |----------------|-------------------------|
    | Interface, colour, contrast, typography, layout, components, UI behaviour | `docs/DESIGN.md` |
    | Model architecture, inference path, data contract | `docs/ARCHITECTURE.md` |
    | Agent rules, finding format, severity vocabulary | `AGENTS.md` section 4 |
    | Data, dataset, provenance | `README.md` and `data/` |

11. Before reporting a defect, check whether a written decision already exists in one of the documents
    above, and whether a test already locks it in. If either one exists, it is not a new defect.

## 4. Rules for the auditor (the same session)
1. Verification happens before a task is considered finished. It is not optional.
2. Check conformance with SPEC.md and ARCHITECTURE.md.
3. Check quality using the report grading rubric: conceptual understanding, novelty and relevance, technical depth, and systematicity.
4. Recompute the numbers from their source. Never copy a number from the output that produced it.
5. Confirm a number through a second path before calling it correct. A green test only counts once you
   have seen it fail on the code that was supposed to fail it.
6. Label every claim that comes from your own report and was not verified through a second path as
   "not independently verified". Do not state it as settled fact.
7. Check specifically:
   - data leakage between train, validation, and test;
   - use of class weights computed from data other than train data;
   - the lookup table being applied only at the training stage;
   - macro F1-score being computed and reported.
8. Never delete or weaken a test to make a task pass. Never invent a cause when the cause is unknown.
9. Finding format: `[severity] file:line - description - proposed fix`.
   Severity: `critical`, `major`, `minor`.
10. Report defects found in your own work. Staying silent repeats the mistake with nobody to catch it.
11. Critical findings block task completion until fixed. Minor findings do not block, but must be recorded.
12. There are only two verdicts: PASS or FAIL. PASS is given when there are no critical or major findings.
13. After all findings are closed, compile one final report in the format of section 8.

## 5. Workflow
```
Task from the owner
      |
      v
Codex CLI: implementation + tests
      |
      v
Codex CLI: verification per section 4, recompute the numbers
      |
      v
Codex CLI: findings list, propose fixes, request approval from the owner
      |
      v
Owner: approve or reject each finding
      |
      v
Codex CLI: apply the approved fixes
      |
      v
Codex CLI: re-verify if there are critical or major findings
      |
      v
Codex CLI: final report with verdict
      |
      v
Task is done when the verdict is PASS
```

If the verdict is FAIL, the flow returns to the verification stage. There is no limit on the number of rounds, but
each round must produce a real change or an explanation of why that finding was closed.

## 5.1 Pull request workflow

Numbered as a subsection of section 5 on purpose. Inserting it as its own
numbered top-level section would have forced the sections after it to be
renumbered, and the existing section numbers are referenced elsewhere.

`main` is protected. A direct push to `main` is rejected by the server and is
forbidden by policy as well. Every change reaches `main` through a pull request
that the project owner merges.

### The current flow
```
Work on a branch, never on main
      |
      v
Commit locally with a message that says what changed
      |
      v
Push the branch
      |
      v
Open a pull request targeting main
      |
      v
Wait for the three CI status checks
      |
      v
Owner reviews and merges
```

### Forbidden
- Pushing to `main`. The server rejects it, and policy forbids it too.
- Merging your own pull request. The owner merges.
- Force pushing to any branch.
- Deleting a branch that is not fully merged.

### Required gates
A pull request cannot be merged until all three status checks are green:

- `CI / Tes pada Python 3.11`
- `CI / Tes pada Python 3.12`
- `CI / Tes pada Python 3.13`

Plus at least one approving review.

### One finding, one branch, one pull request
Do not bundle unrelated changes. Two problems means two branches and two pull
requests.

### When the pull request is created
After the verdict is PASS, not before. The pull request is created after the
verification, so the review on GitHub checks something already verified rather
than searching for problems from the start.

### Secrets
A token never appears in a file, a commit, or a pull request comment. To call the
GitHub API, use the `gh` CLI, which is already logged in. When CI needs a
credential, use GitHub secrets; never write it into a workflow.

## 6. General limits
- No agent uploads DIBaS data to an external service.
- No agent runs full training without confirmation, because of the consumer GPU load.
- Every change is recorded in git history with a commit message that references the task.
- Raw DIBaS data is not committed. Only `data/index.csv` and
  `data/raw/zips_manifest.csv` are committed as provenance evidence.

## 7. Commit Message Format
`[type] short summary` with types: feat, fix, test, docs, refactor.

## 8. Final Report Format
The final report contains five sections in a fixed order:
1. **Result summary** - what was verified and how it was verified.
2. **Findings** - remaining findings in `[severity] file:line - description - proposal` format.
   If there are none, write "No findings".
3. **Status of each previous finding** - for each finding from the previous round, state its
   status: closed, rejected with a reason, or still open.
4. **Specification conformance** - assessment against SPEC.md and the rules in this AGENTS.md.
5. **Verdict** - PASS or FAIL, with a brief reason.
