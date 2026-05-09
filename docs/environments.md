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

## Follow-Up

Step 4 will protect the `production` environment, require production deployment approval, and separate production-scoped variables from lower-environment variables.
