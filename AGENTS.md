# AGENTS.md

## 0) Source of truth
- If `PROJECT_SPEC.md` exists, read it first and treat it as authoritative (requirements, constraints, acceptance criteria).
- If no spec exists, ask for one (or draft a short spec and confirm before large changes).
- Don’t duplicate long docs here—link to existing README/design docs instead.

## 1) Decision priority order (use this when tradeoffs exist)
When choices conflict, prioritize in this order:
1. **Spec / acceptance criteria** (what must be true)
2. **Safety & security** (no data loss, no secrets, safe defaults)
3. **Preserve existing behavior** (backward compatibility, public APIs, existing user workflows)
4. **Verification** (tests/build/run, measurable proof)
5. **Maintainability** (clarity, structure, docs)
6. **UX polish / performance tuning** (only after correctness + stability)

If you must violate a lower-priority item, call it out plainly and choose the safest option.

## 2) Public repository expectations
Assume the work may be shared publicly (e.g., GitHub) and used by others:
- Write readable, maintainable code; avoid project-local hacks.
- Avoid machine-specific paths, usernames, and environment assumptions.
- Never include secrets, tokens, private keys, personal data, or proprietary content in code, tests, logs, docs, or screenshots.
- Prefer clear error messages and docs so a new user can run the project.
- Keep examples/sample data sanitized and small.

## 3) Plan → implement → prove (default workflow)
Before coding:
- Write a short plan (milestones + risks + what “done” means).
- **Inventory first when changing existing code**: identify current features/flows, entrypoints, configs, and baseline checks.
- Identify unknowns early and propose a quick spike when feasibility is uncertain.

While coding:
- Implement incrementally in small steps.
- After each meaningful step: run relevant checks (tests/lint/typecheck/build/run) and fix failures before continuing.
- Add/extend tests for bug fixes and non-trivial features.

Completion proof:
- Provide objective evidence: commands run + results, and point to key files changed.
- If you could not run checks, say exactly why and provide copy/paste commands for the user to run.

## 4) Compare-and-add rule (avoid feature regressions)
When improving UI/UX, refactoring, or “modernizing”:
- **Do not remove existing functionality** unless explicitly instructed.
- Create a quick mapping: *existing feature → where it lives now* (screen/command/setting/API).
- Prefer relocating and improving discoverability over deleting or “simplifying away” features.
- If you believe something should be removed, present it as a proposal with rationale and a safe fallback, but do not delete by default.

## 5) Refactor mode (when asked to refactor/reorganize)
When the task is refactoring, reorganizing, or cleaning up an existing codebase:
- Start with a repo survey: entrypoints, current architecture, dependency layout, and test status.
- Propose a refactor plan with explicit goals, constraints, and a safe sequence.
- Preserve external behavior and public APIs unless explicitly instructed otherwise.
- Make changes in small, reviewable steps. Avoid mixing refactors with feature additions unless requested.
- Maintain a migration path (compat imports/re-exports when feasible), update docs/tests, and verify entrypoints still work.
- End with a concise summary of what moved/changed and how to run the updated project.

## 6) “Fix this repo” mode (when given many files / messy code)
- Reproduce → localize → prove (regression test) → fix.
- Prefer root-cause fixes over patches.
- If multiple fixes are possible, choose the safest and most maintainable.
- Keep a running summary (in the final response) of what changed and why.

## 7) Working agreements
- Prefer correctness and clarity over cleverness.
- Keep changes small and reviewable; avoid unrelated diffs.
- If a tradeoff is required, state it plainly and choose the safest default.
- Only modify files needed for the task; avoid editing generated files, lockfiles, or vendored code unless required.

## 8) Risk tiers & destructive operations (extra caution)
Treat these as high-risk changes and handle with additional safeguards:
- Database migrations / schema changes
- Auth/token/session handling
- Data format migrations (storage schema, exported formats)
- Service worker / caching behavior (PWAs)
- Deleting user data or overwriting input files

For high-risk changes:
- Provide a rollback strategy (or reversible approach).
- Add verification steps (queries/commands) to prove correctness.
- Prefer additive migrations and backward-compatible changes.

## 9) Dependencies & packaging
- Don’t add new dependencies unless there is a clear benefit; justify any new dependency briefly.
- Prefer widely used, well-maintained libraries.
- Keep dependency changes isolated (one commit if possible) and update install/run instructions.
- If the project uses a lockfile/constraints, keep it consistent with existing tooling.

## 10) Safety & scope control
- Never commit or print secrets. Use environment variables for configuration; document required env vars.
- Treat untrusted inputs as hostile (files, archives, registry hives, documents, etc.):
  - validate inputs, avoid executing embedded content, and fail safely
  - use safe temp files and avoid overwriting originals unless explicitly required
- Treat large binary inputs as read-only unless explicitly required to modify them.
- Avoid network dependence in tests; mock external services.
- If a task involves destructive actions, add a confirmation step and provide a rollback strategy.

## 11) Commands (prefer copy/paste exactness)
If the repo already defines commands (README/Makefile/package scripts/pyproject), use those.
Otherwise, propose and document a minimal set (and add lightweight wrappers only if asked):

- Setup: `<fill in>`
- Run: `<fill in>`
- Tests: `<fill in>`
- Lint/format: `<fill in>`
- Typecheck (if used): `<fill in>`

## 12) Definition of done (checklist)
Before calling work “done”:
- Required functionality meets the spec / acceptance criteria.
- Tests pass (and new tests added for fixes/features where appropriate).
- Lint/format/typecheck (if used) are clean or unchanged from baseline.
- README/user docs updated if behavior, setup, or UI changed.
- No accidental large binaries/secrets added; `.gitignore` updated if needed.

## 13) Quality bar
- Public APIs should be documented (docstrings or equivalent) and have stable names.
- Error handling should be explicit, with actionable messages.
- Prefer deterministic tests (seed randomness; avoid real network/time where practical).
- Prefer cross-platform behavior when feasible; if platform-specific, document it clearly.
- Avoid premature abstraction; refactor once the shape of the code is proven by tests and usage.
- For performance-sensitive paths: measure first, optimize second; avoid O(N^2) traps on large inputs.
- If you introduce a failure (tests/build/run), fix it immediately and re-verify (don’t ask for permission to repair obvious breakage).

## 14) UI / UX (for GUI or web front-ends)
- Aim for a professional, modern look with a clean layout; avoid trendy clutter.
- Use a clear information hierarchy: primary actions obvious, secondary actions de-emphasized.
- Prefer simple, consistent patterns (standard navigation, spacing, typography; minimal color).
- Never block the UI thread for long work; use background tasks + progress indicators.
- Provide guardrails: confirmations for destructive actions; undo/redo where feasible (or safe export workflow).
- Keep UI code separated from core logic (thin UI layer over testable core).
- Include a short “User Guide” section in README (navigate, search, edit, export).
- For PWAs:
  - Be careful with service worker caching; avoid caching API responses by accident.
  - Ensure update flow works (version bumps where needed), and provide clear refresh guidance if behavior changes.
  - Use accessible defaults: reasonable contrast, large tap targets, keyboard focus states.

## 15) When unsure / blocked
- Inspect existing conventions first.
- Call out assumptions that could affect behavior and ask or choose the safest default.
- If progress stalls: stop, summarize what you tried, and propose 2–3 next options.
