# API Design Best Practices

## URL Structure

- **Use nouns, not verbs.** `/users/123` not `/getUser?id=123`. The HTTP method is the verb.
- **Plural resource names.** `/users`, `/orders`, `/products`. Consistent pluralization avoids confusion.
- **Nested resources for relationships.** `/users/123/orders` — orders belonging to user 123. Limit nesting to 2 levels max.
- **Lowercase, hyphen-separated.** `/user-profiles` not `/userProfiles` or `/user_profiles`. URLs are case-sensitive and hyphens are URL-safe.
- **No trailing slashes.** Pick one convention and enforce it. `/users` not `/users/`. Redirect the other.
- **Version in the URL.** `/v1/users`. Header-based versioning is technically cleaner but practically harder for clients. URL versioning is explicit and debuggable.

## HTTP Methods

- **GET** — Read. Never mutates state. Must be idempotent. Cacheable.
- **POST** — Create. Not idempotent (calling twice creates two resources). Returns 201 + Location header.
- **PUT** — Full replace. Idempotent. Client sends the complete resource. If it doesn't exist, create it.
- **PATCH** — Partial update. Send only the fields that changed. Not necessarily idempotent.
- **DELETE** — Remove. Idempotent (deleting twice returns 204 or 404, but no error). Returns 204 No Content.
- **Don't tunnel everything through POST.** If you're POST-ing to `/users/123/delete`, you've reinvented RPC badly. Use DELETE.

## Status Codes

Use the right code. Clients depend on them for control flow.

### Success
- **200 OK** — General success. GET returns data, PUT/PATCH returns updated resource.
- **201 Created** — Resource created. Always include `Location` header pointing to the new resource.
- **204 No Content** — Success with no body. DELETE, or PUT/PATCH when you don't return the resource.

### Client Errors
- **400 Bad Request** — Malformed request, validation failure, missing required field.
- **401 Unauthorized** — Not authenticated. "Who are you?" Include `WWW-Authenticate` header.
- **403 Forbidden** — Authenticated but not authorized. "I know who you are, but you can't do this."
- **404 Not Found** — Resource doesn't exist. Also use when you don't want to reveal that the resource exists to unauthorized users.
- **409 Conflict** — State conflict. Duplicate email, version mismatch, resource already exists.
- **422 Unprocessable Entity** — Request is well-formed but semantically invalid. Good for validation errors.
- **429 Too Many Requests** — Rate limited. Include `Retry-After` header.

### Server Errors
- **500 Internal Server Error** — Something broke. Log it, alert on it, don't expose internals.
- **502 Bad Gateway** — Upstream service failed.
- **503 Service Unavailable** — Temporarily down. Include `Retry-After` header.

## Request & Response Design

### Requests
- **Accept both JSON body and query params appropriately.** Filters, pagination, sorting in query params. Resource data in request body.
- **Validate early, fail fast.** Check required fields, types, and constraints before touching the database.
- **Use consistent field naming.** `snake_case` for JSON (Python/Ruby convention) or `camelCase` (JS convention). Pick one, never mix.

### Responses
- **Envelope pattern (optional but useful).** `{"data": {...}, "meta": {...}}` gives room for pagination info, request IDs, etc.
- **Always return the created/updated resource.** After POST or PATCH, return the full object so clients don't need a follow-up GET.
- **Consistent error format.** Every error should follow the same structure:
  ```json
  {
    "error": {
      "code": "VALIDATION_ERROR",
      "message": "Email is required",
      "field": "email"
    }
  }
  ```
- **Use ISO 8601 for dates.** `2026-03-21T14:30:00Z`. Never invent a date format. Include timezone.
- **Null vs. absent.** Be intentional. `"phone": null` means "phone exists but has no value." Omitting `phone` means "phone wasn't requested/relevant." Document which you use.

## Pagination

- **Offset-based:** `?offset=20&limit=10`. Simple, allows jumping to any page. Breaks if data changes between pages (items shift).
- **Cursor-based:** `?cursor=abc123&limit=10`. Stable under concurrent writes. Better for infinite scroll and real-time feeds. Clients can't jump to arbitrary pages.
- **Always return pagination metadata:**
  ```json
  {
    "data": [...],
    "meta": {
      "total": 243,
      "limit": 10,
      "offset": 20,
      "next": "/users?offset=30&limit=10"
    }
  }
  ```
- **Default limit, enforce max.** Default to 20-50, cap at 100. Never let a client request all records.

## Filtering, Sorting, Search

- **Filter with query params.** `?status=active&role=admin`. Keep it flat.
- **Sort with a `sort` param.** `?sort=created_at` (ascending), `?sort=-created_at` (descending, prefix with minus).
- **Search with `q` param.** `?q=john` for full-text search. Separate from field-specific filters.
- **Don't over-engineer.** Start with basic equality filters. Add range filters (`?price_min=10&price_max=50`) only when needed.

## Authentication & Authorization

- **Use Bearer tokens.** `Authorization: Bearer <token>`. Standard, well-supported, stateless.
- **JWTs for stateless auth.** Include user ID, roles, expiry. Verify signature server-side. Keep payloads small.
- **Short-lived access tokens, long-lived refresh tokens.** Access: 15-60 minutes. Refresh: days/weeks, stored securely, rotated on use.
- **API keys for service-to-service.** Simpler than OAuth when both sides are internal. Still use HTTPS.
- **Rate limit by API key/user.** Return `429` with `Retry-After`. Include rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

## Error Handling

- **Be helpful, not verbose.** Error messages should help developers fix the problem. "Field 'email' must be a valid email address" not "Validation error."
- **Never expose internals.** No stack traces, no SQL errors, no file paths in production error responses.
- **Use error codes for programmatic handling.** Machines parse codes, humans read messages. `"code": "DUPLICATE_EMAIL"` is actionable; `"message": "..."` is informational.
- **Validate everything at the boundary.** Type check, length check, format check, range check. Don't rely on the database to catch bad data.

## Versioning

- **Version from day one.** Adding versioning later is painful. Start with `/v1/`.
- **Breaking changes get a new version.** Removing a field, changing a type, altering behavior = new version.
- **Non-breaking changes are fine in-place.** Adding a new optional field, adding a new endpoint = no version bump needed.
- **Support at most 2 versions.** Deprecate the old one with a timeline. Maintaining 5 versions is unsustainable.
- **Communicate deprecation.** `Deprecation` and `Sunset` headers. Give clients 6+ months to migrate.

## Performance

- **Support `fields` parameter.** `?fields=id,name,email` — let clients request only what they need. Reduces payload and server work.
- **Use ETags for caching.** Return `ETag` header, accept `If-None-Match`. Return `304 Not Modified` when nothing changed.
- **Compress responses.** Support `gzip`/`br` encoding. Most frameworks handle this automatically.
- **Batch endpoints for common operations.** If clients always fetch a user + their orders + their profile, consider a combined endpoint instead of forcing 3 round-trips.

## Documentation

- **OpenAPI/Swagger spec.** Machine-readable, generates client SDKs and interactive docs. Maintain it as source of truth.
- **Show real examples.** Every endpoint needs a request example and response example with realistic data, not `"string"` and `0`.
- **Document error responses.** Every endpoint should list its possible error codes and what triggers them.
- **Include authentication instructions.** How to get a token, how to send it, what happens when it expires.

## API Design Checklist

Before shipping an endpoint, verify:

1. Does the URL follow REST conventions (nouns, plural, lowercase)?
2. Are you using the correct HTTP method?
3. Are you returning the right status code for every scenario?
4. Is the error format consistent with the rest of the API?
5. Is input validated before hitting the database?
6. Is pagination implemented for list endpoints?
7. Is the endpoint authenticated and authorized appropriately?
8. Is it documented with examples?
