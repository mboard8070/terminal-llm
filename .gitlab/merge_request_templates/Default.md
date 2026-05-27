## Summary

What changed?

## Change Type

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor
- [ ] CI/CD or release
- [ ] Security or hardening
- [ ] Model routing or provider behavior
- [ ] Documentation

## Verification

- [ ] `venv/bin/python -m pytest`
- [ ] `ruff check .`
- [ ] `npm run build` in `maude-phone`, if touched
- [ ] Manual behavior checked, if applicable
- [ ] CI pipeline is green

## Risk

What could break?

## Review Gates

- [ ] The MR has at least one approval.
- [ ] All discussions are resolved.
- [ ] CODEOWNERS were reviewed for touched paths.
- [ ] No secrets, tokens, local env files, or generated private data are committed.
- [ ] Release notes or wiki docs are updated, if operator behavior changes.

## Model And Deployment Impact

- [ ] No model route, alias, provider, or default selection changed.
- [ ] No production deployment behavior changed.
- [ ] Feature flags are updated, if rollout behavior changed.
- [ ] Model Registry entry is updated, if model-routing behavior changed.

## Notes

Related issues, deployment notes, or follow-up work.
