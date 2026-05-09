# GitLab Environments

Maude records deployments through GitLab CI environment jobs. The current jobs are placeholders so deployment history exists before the final infrastructure commands are wired in.

| Environment | Trigger | Purpose |
| --- | --- | --- |
| `development` | Automatic on `trace-iteration`; manual from merge request pipelines | Records integration deployments for active workflow and routing work. |
| `staging` | Manual on `main` | Records release-candidate verification before production. |
| `production` | Manual on tag pipelines | Records production release deployments after package artifacts are built. |

## Deployment Placeholders

The deployment jobs intentionally echo the target ref and commit SHA instead of changing infrastructure. Replacing a placeholder with a real deploy command should keep:

- The `environment` block so GitLab continues to record deployment history.
- The `resource_group` so only one deployment per environment runs at a time.
- The existing branch or tag rules unless the release process changes.

## Production Protection

The `production` environment is protected in GitLab:

- Deploy access is limited to Maintainers.
- Production deployments require one Maintainer approval.
- The `deploy:production` job remains manual and only appears in tag pipelines.

## Environment Variables

No GitLab CI/CD variables are currently configured for this project. When variables are added:

- Production secrets must be marked protected and scoped to `production`.
- Lower-environment values must be scoped to `development`, `staging`, or `*` only when they are safe for non-production jobs.
- Shared non-secret values should use clear names and avoid granting production access by default.
- Production deploy jobs should depend only on protected branches or protected tags so protected variables cannot be exposed from untrusted refs.
