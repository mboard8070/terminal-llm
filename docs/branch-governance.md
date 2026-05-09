# Branch Governance

This project uses protected branches to keep integration and release work controlled.

## Protected Branches

| Branch pattern | Direct push | Merge access | Force push | Purpose |
| --- | --- | --- | --- | --- |
| `main` | No one | Maintainers | Disabled | Protected release line. Changes land through merge requests with a passing pipeline. |
| `trace-iteration` | Maintainers | Maintainers | Disabled | Active integration branch for trace, routing, and workflow changes. |
| `release/*` | No one | Maintainers | Disabled | Stabilization branches for release candidates and patch releases. |
| `hotfix/*` | No one | Maintainers | Disabled | Emergency fixes that still require merge-request review and CI. |

## Branch Naming

Use these branch prefixes for new work:

- `feature/<short-name>` for new product behavior.
- `fix/<short-name>` for bug fixes.
- `chore/<short-name>` for maintenance and dependency work.
- `docs/<short-name>` for documentation-only changes.
- `experiment/<short-name>` for disposable spikes.
- `release/<version>` for release stabilization.
- `hotfix/<issue-or-version>` for urgent production fixes.

## Merge Expectations

- Merge requests target `trace-iteration` for active feature work unless the change is release-only.
- Merge requests target `main` only when promoting reviewed, green integration work.
- `release/*` and `hotfix/*` branches are MR-only.
- Protected branches require the project approval rule and a successful pipeline before merge.
- All discussions must be resolved before merge.
