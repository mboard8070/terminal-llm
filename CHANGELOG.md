# Changelog

All notable Maude changes should be summarized here before a release tag is created.

This project uses protected `v*` tags for releases. Tag pipelines publish package artifacts to GitLab Generic Packages and attach those package links to the GitLab release.

## Release Cadence

- Use `vMAJOR.MINOR.PATCH` tags for production releases.
- Cut a release from `main` after the `trace-iteration` branch has been reviewed, merged, and verified.
- Prefer small monthly minor releases when there is user-facing work ready.
- Use patch releases for urgent fixes that should not wait for the next planned minor release.
- Create release-candidate branches as `release/<version>` when stabilization needs more than one merge request.

## Release Note Process

Before creating a protected `v*` tag:

- Move relevant entries from `Unreleased` into a versioned section.
- Include user-facing changes, operational changes, security fixes, and migration notes.
- Confirm the tag pipeline will build `maude-client` and `maude-phone` artifacts.
- Confirm production deployment remains manual and requires approval.

## Unreleased

### Added

- GitLab environment deployment placeholders for development, staging, and production.
- Protected production environment with Maintainer deployment approval.
- CI/CD variable governance and protected release tag policy.
- Release artifact links for tag-created GitLab releases.

### Changed

- Production deployment jobs run only for protected tag refs.

## v0.1.0 - 2026-05-08

### Added

- Initial GitLab CI/CD workflow with Python tests, phone build, secret detection, SAST, dependency scanning, package jobs, and release job.
- GitLab workflow files including CODEOWNERS, issue templates, merge request template, and incident template.
- Protected branch rules for `main`.
- Initial GitLab release record.
