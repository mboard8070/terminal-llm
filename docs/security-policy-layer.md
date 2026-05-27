# Security Policy Layer

Maude uses GitLab CI security scanners plus merge-request gates to keep security-sensitive changes reviewable before they reach protected branches.

## Active Controls

| Control | Status |
| --- | --- |
| Secret detection | Enabled in `.gitlab-ci.yml` |
| SAST | Enabled in `.gitlab-ci.yml` |
| Dependency scanning | Enabled in `.gitlab-ci.yml` |
| Required successful pipeline before merge | Enabled for protected branch workflow |
| Required merge request approval | Enabled |
| Author self-approval | Disabled |
| Committer approval | Disabled |
| Security-sensitive CODEOWNERS | Defined in `CODEOWNERS` |
| Protected production environment | Enabled |

## Security Approval Policy

Native GitLab scan-result security approval policies are not currently available through this project API. The API path checked on 2026-05-09 returned `404 Not Found`.

Until native scan-result policies are available, security-sensitive merge requests must use the regular merge-blocking approval rule and explicitly verify:

- Secret detection, SAST, and dependency scanning jobs passed.
- Any high or critical finding is fixed before merge, or a written exception is recorded in the merge request.
- CODEOWNERS for touched security-sensitive paths reviewed the change.
- The change does not weaken protected branches, protected tags, protected environments, production approvals, or CI/CD variable protections.

If GitLab scan-result policies become available later, add a merge-blocking policy that requires Maintainer approval for newly detected high or critical SAST, secret-detection, dependency-scanning, or container-scanning findings.

## Container Scanning

Container scanning is not enabled because this repository does not currently contain a `Dockerfile`, `Containerfile`, or compose file. If Maude adds container images:

- Add GitLab container scanning to `.gitlab-ci.yml`.
- Build images in CI before the scan job.
- Block release or production deployment for high or critical image findings unless an exception is documented.
- Enable GitLab container registry cleanup for old images.

## License And Dependency Governance

GitLab dependency scanning is enabled. License governance is handled by review until native license scanning or a dedicated license inventory job is added.

Before adding or upgrading dependencies:

- Prefer dependencies with active maintenance and clear licenses.
- Avoid copyleft or source-available licenses unless the distribution impact is reviewed.
- Review Python dependencies in `requirements.txt` and `maude-client/pyproject.toml`.
- Review phone dependencies in `maude-phone/package.json` and `maude-phone/package-lock.json`.
- Treat dependency additions that touch authentication, browser automation, filesystem access, command execution, or provider APIs as security-sensitive changes.
