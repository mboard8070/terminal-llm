# Coding Best Practices

## Architecture & Design

- **Keep it simple.** Write the least amount of code that solves the problem. Every line is a liability.
- **Single Responsibility.** Each function, class, or module should do one thing well. If you can't describe what it does in one sentence, split it.
- **Composition over inheritance.** Prefer small, composable pieces over deep class hierarchies.
- **Fail early, fail loud.** Validate inputs at boundaries (user input, API responses, file I/O). Don't silently swallow errors.
- **Don't abstract prematurely.** Three concrete implementations before you extract a pattern. Duplication is cheaper than the wrong abstraction.

## Code Quality

- **Naming matters more than comments.** A function called `find_available_port()` doesn't need a comment. A function called `fap()` needs to be renamed.
- **Functions should be short.** If a function doesn't fit on one screen (~30 lines), it's doing too much.
- **Minimize state.** Prefer pure functions. When you need state, keep it as local as possible.
- **Return early.** Guard clauses at the top eliminate nesting. Don't wrap entire function bodies in if/else.
- **Consistent formatting.** Use a formatter (ruff, black, prettier). Never argue about style — automate it.

## Error Handling

- **Handle errors at the right level.** Catch exceptions where you can actually do something about them, not everywhere.
- **Be specific.** Catch `FileNotFoundError`, not `Exception`. Bare excepts hide bugs.
- **Provide context.** Error messages should say what happened, what was expected, and ideally how to fix it.
- **Don't use exceptions for control flow.** If something is expected to happen regularly, it's not exceptional.

## Security

- **Never trust user input.** Sanitize, validate, escape. Always.
- **No secrets in code.** Use environment variables or secret managers. Never commit API keys, passwords, or tokens.
- **Parameterize queries.** Never concatenate strings into SQL or shell commands.
- **Principle of least privilege.** Request only the permissions you need. Run with the minimum access required.
- **Keep dependencies updated.** Outdated packages are the most common attack vector.

## Testing

- **Test behavior, not implementation.** Tests should verify what code does, not how it does it. Implementation changes shouldn't break tests.
- **One assertion per test.** Each test should verify one thing. If it fails, you know exactly what broke.
- **Test the edges.** Empty inputs, null values, boundary conditions, error paths. Happy path tests alone are insufficient.
- **Integration tests for boundaries.** Mock internal code sparingly. Test real database queries, real API calls, real file I/O where it matters.

## Performance

- **Measure before optimizing.** Profile first. The bottleneck is rarely where you think it is.
- **Algorithmic complexity first.** No amount of micro-optimization fixes an O(n^2) loop over a large dataset.
- **Cache expensive operations.** But invalidate correctly — stale caches cause subtle bugs.
- **Batch I/O.** One query returning 100 rows beats 100 queries returning 1 row.

## Version Control

- **Commit often, push regularly.** Small, focused commits are easier to review, revert, and bisect.
- **Write meaningful commit messages.** "Fix bug" is useless. "Fix race condition in websocket reconnect that dropped messages under load" is useful.
- **One change per commit.** Don't mix refactoring with feature work. Don't fix a bug in the same commit as a new feature.
- **Branch from main, merge back quickly.** Long-lived branches diverge and create painful merges.

## Python-Specific

- **Use type hints.** They catch bugs before runtime and serve as documentation.
- **Use `pathlib` over `os.path`.** Cleaner, more readable, cross-platform.
- **Use f-strings.** Not `.format()`, not `%`. F-strings are faster and more readable.
- **Use context managers.** `with open(...)` guarantees cleanup. Never manually close files.
- **Use dataclasses or named tuples.** Not raw dicts for structured data. You get type safety, defaults, and readability.
- **Use virtual environments.** Always. Never install into system Python.
- **Use `uv` or `pip-tools` for dependency management.** Pin your dependencies. Reproducible builds matter.

## Code Review Checklist

Before submitting code, verify:

1. Does it work? Have you actually run it?
2. Is there a simpler way to do this?
3. Are edge cases handled?
4. Are errors handled gracefully with useful messages?
5. Is it secure? No injection risks, no exposed secrets?
6. Will the next person understand this without asking you?
7. Are there tests for the new behavior?
8. Does it break any existing tests?
