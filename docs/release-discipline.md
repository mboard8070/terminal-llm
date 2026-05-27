# Release Discipline

Maude releases are GitLab tag releases backed by CI package artifacts.

## Tag Cadence

- Production releases use protected `vMAJOR.MINOR.PATCH` tags.
- Minor releases should be cut when reviewed user-facing or operator-facing work has accumulated, usually no more than monthly while active development continues.
- Patch releases are reserved for urgent fixes, security updates, or production deployment corrections.
- Release stabilization work should happen on `release/<version>` when it needs multiple commits or a dedicated review cycle.

## Release Flow

1. Merge reviewed integration work from `trace-iteration` to `main`.
2. Update `CHANGELOG.md`, moving shipped items from `Unreleased` into the target version section.
3. Create a protected `v*` tag from the reviewed `main` commit.
4. Let the tag pipeline build package artifacts and create the GitLab release.
5. Review the generated GitLab release links for:
   - `maude-client` wheel
   - `maude-phone` dist archive
6. Run the manual production deployment job only after the release package links are present and deployment approval is granted.

## Release Artifacts

Tag pipelines upload artifacts to GitLab Generic Packages:

- `maude-client/<tag>/<wheel file>`
- `maude-phone/<tag>/maude-phone-dist-<tag>.tar.gz`

The `release` job consumes package metadata from the package jobs and attaches both package URLs to the GitLab release.

## Rollback Notes

Rollback should prefer a new patch release over retagging an existing version. Existing release tags are protected and should remain immutable.
