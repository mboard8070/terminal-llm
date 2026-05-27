# CI/CD Variables

This project currently has no GitLab CI/CD variables configured at the project level. The `mboard8070` namespace is a personal namespace, so there are no group-level variables inherited by this project.

## Audit Result

Checked on 2026-05-09:

| Scope | Result |
| --- | --- |
| Project CI/CD variables | None configured |
| Group CI/CD variables | Not applicable; personal namespace |
| Protected branches | `main`, `trace-iteration`, `release/*`, `hotfix/*` |
| Protected tags | `v*` |
| Production environment | Protected; Maintainer deploy access; one Maintainer approval |

## Required Settings For Future Secrets

When a CI/CD variable is added, use the narrowest safe scope:

| Variable type | Masked | Protected | Environment scope |
| --- | --- | --- | --- |
| Production secret | Yes | Yes | `production` |
| Staging secret | Yes | No, unless needed only from protected refs | `staging` |
| Development secret | Yes | No, unless needed only from protected refs | `development` |
| Shared non-secret value | No | No | `*` or the specific environment |

Production variables must only be available to protected refs. Production deployment is manual, targets the protected `production` environment, and is limited to protected release tags through `CI_COMMIT_REF_PROTECTED`.

## Operator Checklist

- Do not store local `.env` files or provider keys in the repository.
- Prefer environment-scoped variables over global project variables.
- Mark all API keys, tokens, passwords, and signing credentials as masked.
- Mark production credentials as protected and scope them to `production`.
- Rotate any variable immediately if it appears in job logs, artifacts, issues, merge requests, or committed files.
