# Cybersecurity Best Practices

## Authentication

- **Hash passwords with bcrypt, scrypt, or Argon2.** Never MD5 or SHA-256 for passwords. These are fast hashes — attackers can brute-force billions per second. Bcrypt/Argon2 are deliberately slow.
- **Salt every hash.** A unique random salt per password prevents rainbow table attacks. Bcrypt and Argon2 handle this automatically.
- **Enforce minimum password complexity.** 12+ characters minimum. Check against breached password lists (Have I Been Pwned API). Don't require arbitrary complexity rules (uppercase + number + symbol) — length matters more.
- **Multi-factor authentication everywhere.** TOTP (authenticator app) or hardware keys (FIDO2/WebAuthn). SMS-based 2FA is better than nothing but vulnerable to SIM swapping.
- **Rate limit login attempts.** Lock after 5-10 failed attempts, escalating cooldown. Log every failed attempt with IP and timestamp.
- **Session management:**
  - Generate cryptographically random session IDs (minimum 128 bits)
  - Set `HttpOnly`, `Secure`, and `SameSite=Strict` on session cookies
  - Expire sessions after inactivity (15-30 minutes for sensitive apps)
  - Invalidate all sessions on password change
  - Rotate session ID after login (prevents session fixation)

## Input Validation & Injection

### SQL Injection
- **Use parameterized queries. Always.** Never concatenate user input into SQL strings. This is the #1 most exploited vulnerability and the easiest to prevent.
- **Bad:** `f"SELECT * FROM users WHERE id = {user_input}"`
- **Good:** `cursor.execute("SELECT * FROM users WHERE id = ?", (user_input,))`
- **ORMs help but aren't bulletproof.** Raw query methods in ORMs can still be vulnerable. Audit any raw SQL.

### Command Injection
- **Never pass user input to shell commands.** Use `subprocess.run(["cmd", arg])` (list form), not `os.system(f"cmd {user_input}")` or `subprocess.run(f"cmd {user_input}", shell=True)`.
- **If you must use shell:** Whitelist allowed characters, reject everything else. But seriously, don't use shell.
- **Avoid `eval()`, `exec()`, `Function()` with user input.** In any language. Ever.

### XSS (Cross-Site Scripting)
- **Escape all user-generated content on output.** HTML-encode `<`, `>`, `&`, `"`, `'` before rendering in pages.
- **Use Content Security Policy (CSP) headers.** `Content-Security-Policy: default-src 'self'` blocks inline scripts and external script loading.
- **Sanitize HTML if you must allow it.** Use a library like DOMPurify (JS) or bleach (Python). Never regex your way through HTML sanitization.
- **Use `textContent` not `innerHTML`** when inserting user data into the DOM.

### Path Traversal
- **Validate file paths server-side.** Reject `../`, `..\\`, null bytes, and URL-encoded variants.
- **Use `os.path.realpath()` and verify the resolved path is within the allowed directory.**
- **Never use user input directly in file paths** without normalization and containment checks.

## Secrets Management

- **No secrets in code. No secrets in git history.** If a secret was ever committed, rotate it immediately — removing it from code doesn't remove it from history.
- **Use environment variables for local dev.** `.env` files in `.gitignore`. Never commit `.env`.
- **Use a secrets manager in production.** AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager, or even encrypted environment variables in your CI/CD. Not plaintext config files on the server.
- **Rotate secrets regularly.** API keys, database passwords, tokens. Automate rotation where possible.
- **Least privilege for secrets.** A service that reads from the database doesn't need the admin password. Create scoped credentials.
- **Scan for leaked secrets.** Use tools like `gitleaks`, `trufflehog`, or GitHub's secret scanning. Run them in CI.

## Network Security

- **HTTPS everywhere.** No exceptions. Use Let's Encrypt for free certs. Set `Strict-Transport-Security` (HSTS) header with a long max-age.
- **TLS 1.2 minimum.** Disable TLS 1.0, 1.1, and all SSL versions. Prefer TLS 1.3.
- **Firewall defaults to deny.** Only open ports you actively use. Close everything else. Audit open ports regularly with `ss -tlnp` or `nmap`.
- **Don't expose internal services.** Databases, caches (Redis), message queues should never be accessible from the internet. Bind to `127.0.0.1` or use private networks.
- **Use a reverse proxy.** Nginx, Caddy, or Traefik in front of your application server. Handles TLS termination, rate limiting, and request filtering.
- **DNS security.** Enable DNSSEC if possible. Use CAA records to restrict which CAs can issue certificates for your domain.

## Server Hardening

- **Update everything.** OS, packages, dependencies, firmware. Unpatched systems are the most common attack vector. Automate updates where safe.
- **Disable root SSH login.** Use key-based authentication only. Disable password auth in `sshd_config`. Change the default SSH port (security through obscurity helps against bots, not targeted attacks).
- **Principle of least privilege.** Run services as non-root users. Use `setcap` instead of running as root when a binary needs specific capabilities.
- **Remove unnecessary software.** Every installed package is attack surface. Minimal base images for containers. No dev tools on production servers.
- **File permissions matter.** Config files: `640` or `600`. Scripts: `750`. Private keys: `600`. Web-accessible directories: never writable by the web server user.
- **Enable audit logging.** `auditd` on Linux. Log all sudo usage, file access to sensitive directories, and authentication events.

## Container Security

- **Don't run as root in containers.** Use `USER nonroot` in your Dockerfile. Running as root inside a container means root on the host if the container is escaped.
- **Use minimal base images.** `alpine`, `distroless`, or `slim` variants. Smaller image = fewer vulnerabilities.
- **Pin image versions.** `python:3.12-slim` not `python:latest`. Scan images with Trivy, Grype, or Snyk.
- **Read-only filesystems where possible.** `docker run --read-only`. Mount specific writable volumes only where needed.
- **Don't store secrets in images.** Use Docker secrets, environment variables at runtime, or mounted secret volumes. Never `COPY .env` or `ARG API_KEY` in a Dockerfile.
- **Limit container capabilities.** `--cap-drop=ALL --cap-add=NET_BIND_SERVICE` — only grant what's needed.
- **Network isolation.** Use Docker networks to segment containers. Frontend container shouldn't be able to reach the database directly.

## Dependency Security

- **Pin your dependencies.** Lock files (`uv.lock`, `package-lock.json`, `Pipfile.lock`) ensure reproducible builds and prevent supply chain attacks via version drift.
- **Audit regularly.** `npm audit`, `pip-audit`, `safety check`. Run in CI, fail the build on critical vulnerabilities.
- **Minimize dependencies.** Every package is code you didn't write and can't fully audit. Do you really need a library for `left-pad`?
- **Verify package integrity.** Use checksums and signatures. Enable npm's `--ignore-scripts` for untrusted packages.
- **Monitor for compromised packages.** Subscribe to security advisories for your ecosystem. Socket.dev, Snyk, GitHub Dependabot.

## Data Protection

- **Encrypt data at rest.** Full-disk encryption (LUKS, FileVault, BitLocker). Database-level encryption for sensitive fields.
- **Encrypt data in transit.** TLS for all network communication. Internal service-to-service communication included.
- **Minimize data collection.** Don't store what you don't need. Every piece of stored PII is a liability. GDPR, CCPA, and similar regulations require justification for data retention.
- **Anonymize where possible.** Use hashed or tokenized identifiers instead of PII for analytics and logging.
- **Backup encryption.** Backups are a copy of your most sensitive data. Encrypt them. Store encryption keys separately from the backups.
- **Secure deletion.** When data must be removed, overwrite it — don't just delete the pointer. `shred` for files, proper purge for databases.

## Logging & Monitoring

- **Log security events.** Authentication attempts (success and failure), privilege escalation, data access, configuration changes, API errors.
- **Never log secrets.** Passwords, tokens, API keys, credit card numbers. Sanitize logs before writing. Mask sensitive fields.
- **Centralize logs.** Ship to a dedicated logging service (ELK, Loki, CloudWatch). Logs on the compromised server get deleted by attackers.
- **Alert on anomalies.** Failed login spikes, unusual API patterns, new admin accounts, off-hours access. Automated alerts, not daily log reviews.
- **Retain logs appropriately.** 90 days minimum for security investigation. Compliance may require longer. Secure log storage with tamper detection.
- **Include context in logs.** Timestamp (UTC), source IP, user ID, action, result, request ID. A log entry should tell a complete story.

## Incident Response

- **Have a plan before you need one.** Document: who to contact, how to isolate systems, where backups are, how to rotate all credentials.
- **Contain first, investigate second.** Isolate the affected system, revoke compromised credentials, then figure out what happened.
- **Preserve evidence.** Don't wipe the compromised system before forensics. Snapshot it. Copy logs. Document timeline.
- **Rotate everything that might be compromised.** Tokens, API keys, passwords, certificates. When in doubt, rotate.
- **Post-mortem without blame.** What happened, why, what was the impact, what will prevent it next time. Share findings. Blameless culture finds more bugs.

## Security Checklist

Before deploying, verify:

1. Are all passwords hashed with bcrypt/Argon2 (not MD5/SHA)?
2. Is every database query parameterized?
3. Is user input escaped/sanitized before rendering?
4. Are secrets in environment variables or a secrets manager (not in code)?
5. Is HTTPS enforced with valid certificates?
6. Are unnecessary ports closed?
7. Are dependencies pinned and audited for vulnerabilities?
8. Is logging in place for auth events and errors (with secrets redacted)?
9. Are containers running as non-root with minimal capabilities?
10. Is there an incident response plan documented and accessible?
