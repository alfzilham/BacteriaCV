# AGENT.md: Instructions for Agents in this Repository

## 1. Roles
| Agent | Tool | Role | Authority |
|------|------|-------|------------|
| opencode | opencode | Code author | Writes and modifies code according to SPEC, ARCHITECTURE, DESIGN |
| Codex CLI | OpenAI Codex CLI | Read-only auditor | Verifies claims by recomputing the numbers from the source; reports findings with a PASS or FAIL verdict; does not modify production code |
| Antigravity CLI | Antigravity IDE 1.107.0, chat mode --mode agent | Interface author | Writes and modifies app/static/index.html and docs/DESIGN.md according to DESIGN.md and after owner approval |

The auditor and the author must not be the same for a single task.

The industrial brutalist interface redesign was carried out by Antigravity CLI version 1.107.0, not by opencode.

The Codex CLI version number is not recorded because the `codex --version` command was unavailable on the machine where the Task F audit ran. Its role remains read-only and unchanged.

## 2. Sources of truth
Priority order when a conflict arises:
1. SPEC.md
2. ARCHITECTURE.md
3. DESIGN.md
4. PRD.md and CONTEXT.md (draft status until validated)
5. AGENT.md
6. Existing code

An agent must not change a decision recorded in SPEC.md without explicit approval from the project owner.

## 3. Rules for opencode (author)
1. Do one task per change. Never combine a refactor with a new feature.
2. Follow the interface in DESIGN.md. If the interface must change, write up the proposal and wait for approval.
3. Never change the lookup table [L] or the class weighting rules without approval.
4. Never use test data for any decision during training.
5. Never store credentials, tokens, or absolute paths in code.
6. Every finished task must come with tests according to DESIGN.md Section 5.
7. When done, hand over the change summary, file list, and audit prompt to the project owner.
8. The audit prompt contains context, references, file list, and inspection focus. The prompt must be
   complete enough that the auditor does not need to guess the task scope.
9. Never declare a task finished before the auditor's final report states PASS.

## 4. Rules for Codex CLI (auditor)
1. The audit must happen before a task is considered finished.
2. Check conformance with SPEC.md and ARCHITECTURE.md.
3. Check quality using the report grading rubric: conceptual understanding, novelty and relevance, technical depth, and systematicity.
4. Check specifically:
   - data leakage between train, validation, and test;
   - use of class weights computed from data other than train data;
   - the lookup table being applied only at the training stage;
   - macro F1-score being computed and reported.
5. Finding format: `[severity] file:line - description - proposed fix`.
   Severity: `critical`, `major`, `minor`.
6. The auditor does not modify production code. Production code is changed only by the author, because if
   the auditor modifies the file they just audited, their own mistakes can be covered up and the audit
   result is no longer independent.
7. The auditor may modify test files after the project owner explicitly approves the proposal. Test files
   are not objects whose soundness the audit evaluates.
8. After the audit is done, the auditor immediately compiles the finding list and requests verification from
   the project owner. The auditor does not wait to be asked again.
9. Each verification request states per finding: severity, file:line, impact or risk, and a concrete
   proposed fix. The owner must be able to decide without reading the code.
10. Critical findings block task completion until fixed.
11. There are only two verdicts: PASS or FAIL. PASS is given when there are no critical
    or major findings. Minor findings do not block, but must still be recorded.
12. After all approved findings are fixed, the auditor compiles one final report stating
    the status of each finding and the final verdict. That report is what the owner forwards
    to the author.

## 5. Workflow
```
Task from the owner
      |
      v
opencode: implementation + tests
      |
      v
opencode: hand over summary, file list, and audit prompt
      |
      v
Codex CLI: audit, produce findings
      |
      v
Codex CLI: request verification of findings from the owner
      |
      v
Owner: approve or reject each finding
      |
      v
Codex CLI: apply approved test fixes
      |
      v
Codex CLI: re-audit if there are critical or major findings
      |
      v
Codex CLI: final report with verdict
      |
      v
Owner: forward the final report to opencode
      |
      v
Task is done when the verdict is PASS
```

If the verdict is FAIL, the flow returns to the audit stage. There is no limit on the number of rounds, but
each round must produce a real change or an explanation of why that finding was closed.

## 6. General limits
- No agent uploads DIBaS data to an external service.
- No agent runs full training without confirmation, because of the consumer GPU load.
- Every change is recorded in git history with a commit message that references the task.
- Raw DIBaS data is not committed. Only `data/index.csv` and
  `data/raw/zips_manifest.csv` are committed as provenance evidence.

## 7. Commit Message Format
`[type] short summary` with types: feat, fix, test, docs, refactor.

## 8. Auditor Final Report Format
The final report contains five sections in a fixed order:
1. **Result summary** — what was verified and how it was verified.
2. **Findings** — remaining findings in `[severity] file:line - description - proposal` format.
   If there are none, write "No findings".
3. **Status of each previous finding** — for each finding from the previous round, state its
   status: closed, rejected with a reason, or still open.
4. **Specification conformance** — assessment against SPEC.md and the rules in this AGENT.md.
5. **Verdict** — PASS or FAIL, with a brief reason.