---
name: review-rapidsmpf
description: Use this skill to review GitHub pull requests for rapidsmpf
---

Use this skill when the user invokes `/review-rapidsmpf` with:

- a rapidsmpf GitHub PR link
- a currently checked-out rapidsmpf PR
- specified rapidsmpf code changes or a diff

rapidsmpf GitHub repository is located at: <https://github.com/rapidsai/rapidsmpf>

# Review RapidsMPF Pull Request

1. **Fetch PR metadata and diff**

   ```bash
   gh pr view <PR_NUMBER> --repo rapidsai/rapidsmpf --json title,body,files,additions,deletions,baseRefName,headRefName
   gh pr diff <PR_NUMBER> --repo rapidsai/rapidsmpf
   ```

   Hint: Check if `GH_TOKEN` (or GitHub CLI auth) is already configured
   in the environment (e.g. via your secret manager) so `gh` can
   authenticate and bypass rate limits; do not run `gh auth token` from
   within the agent. If `gh` auth is unavailable, fall back to GitHub's
   raw diff/patch URLs, `git fetch` of the PR ref, the unauthenticated
   GitHub REST API with `curl`, or any other available method.

2. **Fetch review comments already posted** for context on what's
   already been suggested and need not be repeated.

3. **Read the review guidelines** — these are the authoritative
   checklists for rapidsmpf. All rules in these files apply during
   review:

   - **C++/CUDA**: `cpp/REVIEW_GUIDELINES.md`
   - **Python / Cython**: `python/REVIEW_GUIDELINES.md`

4. **Analyze the changes** against the checklists, reading relevant
   source files for context. Public headers under
   `cpp/include/rapidsmpf/` (especially `error.hpp`,
   `memory/cuda_memcpy_async.hpp`, `progress_thread.hpp`,
   `memory/buffer.hpp`, `memory/buffer_resource.hpp`,
   `communicator/communicator.hpp`, `shuffler/shuffler.hpp`,
   `coll/*.hpp`) are the source of truth for APIs and ownership
   contracts.

5. **Produce a structured review** using the output format at the
   bottom.

6. **Dump the structured review** to
   `.agents/reviews/<PR_NUMBER>/review.md`.

---

## Reviewer Discipline

The detailed rules live in `cpp/REVIEW_GUIDELINES.md` and
`python/REVIEW_GUIDELINES.md`. A few meta-rules that apply to *any*
review of this repo:

- **No bypass of CI**: never suggest `--no-verify`, disabling hooks,
  or skipping checks.
- **No drive-by refactors**: the diff should change only what's
  necessary; flag unrelated reformatting or scope creep.
- **No API invention**: public APIs should align with existing
  RapidsMPF patterns and documented contracts — don't propose new
  surface area that doesn't match prevailing conventions.

## Reference Material

| Topic | Path |
|-------|------|
| C++ review guidelines | `cpp/REVIEW_GUIDELINES.md` |
| Python / Cython review guidelines | `python/REVIEW_GUIDELINES.md` |
| C++ public headers (source of truth for APIs) | `cpp/include/rapidsmpf/` |
| Python package | `python/rapidsmpf/` |
| Pre-commit config (lint / format / hooks) | `.pre-commit-config.yaml` |
| CI scripts | `ci/` |
| Local CI reproduction | `.agents/skills/reproduce-ci-locally/SKILL.md` |
| Contributing guide | `CONTRIBUTING.md` |

Online docs:

- Documentation: <https://docs.rapids.ai/api/rapidsmpf/nightly/>
- C++ API: <https://docs.rapids.ai/api/librapidsmpf/nightly/>
- GitHub Issues: <https://github.com/rapidsai/rapidsmpf/issues>

---

## Output Format

Structure your review as follows:

```markdown
## PR Review: <PR title>

**PR:** <link>
**Summary:** <1-2 sentence summary of what the PR does>

### Findings

#### Critical
- **[file:line]** Description of issue that must be fixed before merge.

#### Suggestions
- **[file:line]** Description of improvement to consider.

#### Nits
- **[file:line]** Minor style or formatting issue. Keep these minimal,
  don't suggest adding comments around every line of code or obvious
  logic.

#### Highlights
- Highlight well-written code, good test coverage, or clever
  solutions.

### Verdict
One of: **Approve**, **Request Changes**, or **Comment**
With a brief justification.
```

If there are no findings in a category, omit that category.
