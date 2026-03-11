"""
GitHub tool implementations — repos, issues, PRs, branches, commits,
workflow runs, releases, and search via gh CLI.
"""

import subprocess
import json

from tool_registry import register_tool


def _run_gh(*args: str, timeout: int = 30) -> tuple[int, str]:
    """Run a gh CLI command and return (returncode, output)."""
    try:
        result = subprocess.run(
            ["gh", *args],
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return result.returncode, output
    except FileNotFoundError:
        return 1, "Error: gh CLI not found. Install from https://cli.github.com/"
    except subprocess.TimeoutExpired:
        return 1, "Error: gh command timed out."


# ── Pull Requests ──────────────────────────────────────────────

def tool_github_list_prs(repo: str = "", state: str = "open", limit: int = 10) -> str:
    """List pull requests for a repository."""
    args = ["pr", "list", "--state", state, "--limit", str(limit), "--json",
            "number,title,state,author,headRefName,baseRefName,createdAt,mergeable"]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error listing PRs: {output}"
    try:
        prs = json.loads(output)
    except json.JSONDecodeError:
        return f"Error parsing PR list: {output}"
    if not prs:
        return f"No {state} pull requests found."
    lines = []
    for pr in prs:
        author = pr.get("author", {}).get("login", "unknown")
        mergeable = pr.get("mergeable", "UNKNOWN")
        lines.append(
            f"#{pr['number']}  {pr['title']}\n"
            f"   {pr['headRefName']} → {pr['baseRefName']}  by {author}  "
            f"state={pr['state']}  mergeable={mergeable}"
        )
    return "\n\n".join(lines)


def tool_github_view_pr(pr_number: int, repo: str = "") -> str:
    """View details of a specific pull request."""
    args = ["pr", "view", str(pr_number), "--json",
            "number,title,state,body,author,headRefName,baseRefName,"
            "mergeable,reviewDecision,statusCheckRollup,additions,deletions,changedFiles"]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error viewing PR #{pr_number}: {output}"
    try:
        pr = json.loads(output)
    except json.JSONDecodeError:
        return f"Error parsing PR: {output}"
    author = pr.get("author", {}).get("login", "unknown")
    checks = pr.get("statusCheckRollup", []) or []
    check_summary = ""
    if checks:
        passed = sum(1 for c in checks if c.get("conclusion") == "SUCCESS")
        check_summary = f"  checks: {passed}/{len(checks)} passed"
    return (
        f"PR #{pr['number']}: {pr['title']}\n"
        f"  {pr['headRefName']} → {pr['baseRefName']}  by {author}\n"
        f"  state: {pr['state']}  mergeable: {pr.get('mergeable', 'UNKNOWN')}\n"
        f"  review: {pr.get('reviewDecision') or 'NONE'}{check_summary}\n"
        f"  +{pr.get('additions', 0)} -{pr.get('deletions', 0)} in {pr.get('changedFiles', 0)} files\n"
        f"\n{pr.get('body') or '(no description)'}"
    )


def tool_github_create_pr(title: str, body: str = "", base: str = "", head: str = "",
                           repo: str = "", draft: bool = False) -> str:
    """Create a pull request."""
    args = ["pr", "create", "--title", title]
    if body:
        args.extend(["--body", body])
    if base:
        args.extend(["--base", base])
    if head:
        args.extend(["--head", head])
    if draft:
        args.append("--draft")
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error creating PR: {output}"
    return f"Pull request created: {output}"


def tool_github_merge_pr(pr_number: int, repo: str = "", method: str = "merge",
                         delete_branch: bool = True) -> str:
    """Merge a pull request."""
    if method not in ("merge", "squash", "rebase"):
        return f"Error: Invalid merge method '{method}'. Use merge, squash, or rebase."
    args = ["pr", "merge", str(pr_number), f"--{method}"]
    if delete_branch:
        args.append("--delete-branch")
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error merging PR #{pr_number}: {output}"
    return f"Successfully merged PR #{pr_number} via {method}.\n{output}"


def tool_github_close_pr(pr_number: int, repo: str = "", comment: str = "") -> str:
    """Close a pull request without merging."""
    if comment:
        _run_gh("pr", "comment", str(pr_number), "--body", comment,
                *(["--repo", repo] if repo else []))
    args = ["pr", "close", str(pr_number)]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error closing PR #{pr_number}: {output}"
    return f"PR #{pr_number} closed.\n{output}"


def tool_github_pr_diff(pr_number: int, repo: str = "") -> str:
    """View the diff of a pull request."""
    args = ["pr", "diff", str(pr_number)]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args, timeout=60)
    if rc != 0:
        return f"Error getting PR diff: {output}"
    # Truncate very large diffs
    if len(output) > 8000:
        output = output[:8000] + "\n... (diff truncated, showing first 8000 chars)"
    return output


def tool_github_pr_comments(pr_number: int, repo: str = "") -> str:
    """List comments on a pull request."""
    args = ["pr", "view", str(pr_number), "--json", "comments"]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error getting PR comments: {output}"
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return f"Error parsing comments: {output}"
    comments = data.get("comments", [])
    if not comments:
        return f"No comments on PR #{pr_number}."
    lines = []
    for c in comments:
        author = c.get("author", {}).get("login", "unknown")
        created = c.get("createdAt", "")[:10]
        body = c.get("body", "")
        if len(body) > 200:
            body = body[:200] + "..."
        lines.append(f"  {author} ({created}): {body}")
    return f"Comments on PR #{pr_number}:\n" + "\n".join(lines)


def tool_github_comment_pr(pr_number: int, body: str, repo: str = "") -> str:
    """Add a comment to a pull request."""
    args = ["pr", "comment", str(pr_number), "--body", body]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error commenting on PR: {output}"
    return f"Comment added to PR #{pr_number}."


# ── Issues ─────────────────────────────────────────────────────

def tool_github_list_issues(repo: str = "", state: str = "open", labels: str = "",
                             limit: int = 10) -> str:
    """List issues for a repository."""
    args = ["issue", "list", "--state", state, "--limit", str(limit),
            "--json", "number,title,state,author,labels,createdAt,assignees"]
    if labels:
        args.extend(["--label", labels])
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error listing issues: {output}"
    try:
        issues = json.loads(output)
    except json.JSONDecodeError:
        return f"Error parsing issues: {output}"
    if not issues:
        return f"No {state} issues found."
    lines = []
    for iss in issues:
        author = iss.get("author", {}).get("login", "unknown")
        label_names = [l.get("name", "") for l in iss.get("labels", [])]
        label_str = f"  [{', '.join(label_names)}]" if label_names else ""
        assignees = [a.get("login", "") for a in iss.get("assignees", [])]
        assignee_str = f"  -> {', '.join(assignees)}" if assignees else ""
        lines.append(f"#{iss['number']}  {iss['title']}{label_str}{assignee_str}  by {author}")
    return "\n".join(lines)


def tool_github_view_issue(issue_number: int, repo: str = "") -> str:
    """View details of a specific issue."""
    args = ["issue", "view", str(issue_number), "--json",
            "number,title,state,body,author,labels,assignees,comments,createdAt,closedAt"]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error viewing issue #{issue_number}: {output}"
    try:
        iss = json.loads(output)
    except json.JSONDecodeError:
        return f"Error parsing issue: {output}"
    author = iss.get("author", {}).get("login", "unknown")
    labels = [l.get("name", "") for l in iss.get("labels", [])]
    assignees = [a.get("login", "") for a in iss.get("assignees", [])]
    comments = iss.get("comments", [])
    body = iss.get("body") or "(no description)"
    if len(body) > 3000:
        body = body[:3000] + "\n... (truncated)"
    result = (
        f"Issue #{iss['number']}: {iss['title']}\n"
        f"  state: {iss['state']}  by {author}\n"
    )
    if labels:
        result += f"  labels: {', '.join(labels)}\n"
    if assignees:
        result += f"  assignees: {', '.join(assignees)}\n"
    result += f"  comments: {len(comments)}\n\n{body}"
    return result


def tool_github_create_issue(title: str, body: str = "", labels: str = "",
                              assignee: str = "", repo: str = "") -> str:
    """Create a new issue."""
    args = ["issue", "create", "--title", title]
    if body:
        args.extend(["--body", body])
    if labels:
        args.extend(["--label", labels])
    if assignee:
        args.extend(["--assignee", assignee])
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error creating issue: {output}"
    return f"Issue created: {output}"


def tool_github_close_issue(issue_number: int, comment: str = "", repo: str = "") -> str:
    """Close an issue."""
    if comment:
        _run_gh("issue", "comment", str(issue_number), "--body", comment,
                *(["--repo", repo] if repo else []))
    args = ["issue", "close", str(issue_number)]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error closing issue #{issue_number}: {output}"
    return f"Issue #{issue_number} closed.\n{output}"


def tool_github_comment_issue(issue_number: int, body: str, repo: str = "") -> str:
    """Add a comment to an issue."""
    args = ["issue", "comment", str(issue_number), "--body", body]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error commenting on issue: {output}"
    return f"Comment added to issue #{issue_number}."


# ── Repos ──────────────────────────────────────────────────────

def tool_github_list_repos(owner: str = "", limit: int = 10) -> str:
    """List repositories for a user/org or the authenticated user."""
    args = ["repo", "list"]
    if owner:
        args.append(owner)
    args.extend(["--limit", str(limit),
                 "--json", "name,description,visibility,updatedAt,primaryLanguage,stargazerCount"])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error listing repos: {output}"
    try:
        repos = json.loads(output)
    except json.JSONDecodeError:
        return f"Error parsing repos: {output}"
    if not repos:
        return "No repositories found."
    lines = []
    for r in repos:
        stars = r.get("stargazerCount", 0)
        pl = r.get("primaryLanguage") or {}
        lang = pl.get("name", "") if isinstance(pl, dict) else ""
        vis = r.get("visibility", "").lower()
        desc = r.get("description", "") or ""
        if len(desc) > 60:
            desc = desc[:60] + "..."
        parts = [f"{r['name']}"]
        if vis == "private":
            parts.append("[private]")
        if lang:
            parts.append(lang)
        if stars:
            parts.append(f"★{stars}")
        if desc:
            parts.append(f"— {desc}")
        lines.append("  ".join(parts))
    return "\n".join(lines)


def tool_github_view_repo(repo: str = "") -> str:
    """View detailed info about a repository."""
    args = ["repo", "view", "--json",
            "name,owner,description,url,visibility,defaultBranchRef,"
            "stargazerCount,forkCount,languages,createdAt,updatedAt,"
            "hasIssuesEnabled,hasWikiEnabled,isArchived,licenseInfo"]
    if repo:
        args.insert(2, repo)
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error viewing repo: {output}"
    try:
        r = json.loads(output)
    except json.JSONDecodeError:
        return f"Error parsing repo: {output}"
    owner = r.get("owner", {}).get("login", "unknown")
    branch = r.get("defaultBranchRef", {}).get("name", "unknown") if r.get("defaultBranchRef") else "unknown"
    langs = r.get("languages", [])
    lang_str = ", ".join(l.get("node", {}).get("name", "") for l in (langs.get("edges", []) if isinstance(langs, dict) else []))
    license_name = ""
    if r.get("licenseInfo"):
        license_name = r["licenseInfo"].get("name", "")
    result = (
        f"{owner}/{r.get('name', '?')}\n"
        f"  {r.get('description') or '(no description)'}\n"
        f"  url: {r.get('url', '')}\n"
        f"  visibility: {r.get('visibility', '').lower()}  branch: {branch}\n"
        f"  ★ {r.get('stargazerCount', 0)}  forks: {r.get('forkCount', 0)}\n"
    )
    if lang_str:
        result += f"  languages: {lang_str}\n"
    if license_name:
        result += f"  license: {license_name}\n"
    if r.get("isArchived"):
        result += "  ⚠ ARCHIVED\n"
    return result


# ── Branches ───────────────────────────────────────────────────

def tool_github_list_branches(repo: str = "", limit: int = 20) -> str:
    """List branches in a repository."""
    args = ["api", "repos/{owner}/{repo}/branches", "--paginate", "--jq",
            '.[] | "\\(.name)  \\(if .protected then "[protected]" else "" end)"']
    if repo:
        args = ["api", f"repos/{repo}/branches", "--paginate", "--jq",
                '.[] | "\\(.name)  \\(if .protected then "[protected]" else "" end)"']
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error listing branches: {output}"
    lines = output.strip().split("\n")
    if len(lines) > limit:
        lines = lines[:limit]
        lines.append(f"... ({len(output.strip().split(chr(10)))} total)")
    return "\n".join(lines) if lines else "No branches found."


# ── Commits ────────────────────────────────────────────────────

def tool_github_list_commits(repo: str = "", branch: str = "", limit: int = 10) -> str:
    """List recent commits."""
    args = ["api", "repos/{owner}/{repo}/commits",
            "--jq", '.[] | "\\(.sha[:7])  \\(.commit.author.name)  \\(.commit.message | split("\\n")[0])"']
    if repo:
        args[1] = f"repos/{repo}/commits"
    params = ["-f", f"per_page={limit}"]
    if branch:
        params.extend(["-f", f"sha={branch}"])
    args.extend(params)
    rc, output = _run_gh(*args)
    if rc != 0:
        # Fallback to git log for current repo
        args2 = ["log", f"--oneline", f"-{limit}"]
        try:
            result = subprocess.run(["git"] + args2, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return f"Error listing commits: {output}"
    return output.strip() if output.strip() else "No commits found."


# ── Workflow Runs ──────────────────────────────────────────────

def tool_github_list_runs(repo: str = "", limit: int = 10, status: str = "") -> str:
    """List recent workflow runs (CI/CD)."""
    args = ["run", "list", "--limit", str(limit),
            "--json", "databaseId,name,status,conclusion,headBranch,event,createdAt"]
    if status:
        args.extend(["--status", status])
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error listing runs: {output}"
    try:
        runs = json.loads(output)
    except json.JSONDecodeError:
        return f"Error parsing runs: {output}"
    if not runs:
        return "No workflow runs found."
    lines = []
    for r in runs:
        conclusion = r.get("conclusion", "") or r.get("status", "")
        branch = r.get("headBranch", "")
        lines.append(
            f"#{r['databaseId']}  {r['name']}  {conclusion}  "
            f"branch={branch}  trigger={r.get('event', '')}"
        )
    return "\n".join(lines)


def tool_github_view_run(run_id: int, repo: str = "") -> str:
    """View details of a specific workflow run."""
    args = ["run", "view", str(run_id),
            "--json", "databaseId,name,status,conclusion,headBranch,event,"
            "createdAt,updatedAt,jobs,url"]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error viewing run #{run_id}: {output}"
    try:
        run = json.loads(output)
    except json.JSONDecodeError:
        return f"Error parsing run: {output}"
    result = (
        f"Run #{run.get('databaseId')}: {run.get('name')}\n"
        f"  status: {run.get('status')}  conclusion: {run.get('conclusion') or 'in progress'}\n"
        f"  branch: {run.get('headBranch')}  trigger: {run.get('event')}\n"
        f"  url: {run.get('url', '')}\n"
    )
    jobs = run.get("jobs", [])
    if jobs:
        result += "\n  Jobs:\n"
        for j in jobs:
            result += f"    {j.get('name', '?')}  {j.get('conclusion') or j.get('status', '?')}\n"
    return result


def tool_github_rerun(run_id: int, repo: str = "", failed_only: bool = False) -> str:
    """Re-run a workflow run."""
    args = ["run", "rerun", str(run_id)]
    if failed_only:
        args.append("--failed")
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error re-running #{run_id}: {output}"
    return f"Workflow run #{run_id} re-triggered.\n{output}"


# ── Releases ───────────────────────────────────────────────────

def tool_github_list_releases(repo: str = "", limit: int = 5) -> str:
    """List releases."""
    args = ["release", "list", "--limit", str(limit)]
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error listing releases: {output}"
    return output if output else "No releases found."


def tool_github_create_release(tag: str, title: str = "", notes: str = "",
                                draft: bool = False, prerelease: bool = False,
                                repo: str = "") -> str:
    """Create a new release."""
    args = ["release", "create", tag]
    if title:
        args.extend(["--title", title])
    if notes:
        args.extend(["--notes", notes])
    else:
        args.append("--generate-notes")
    if draft:
        args.append("--draft")
    if prerelease:
        args.append("--prerelease")
    if repo:
        args.extend(["--repo", repo])
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error creating release: {output}"
    return f"Release created: {output}"


# ── Search ─────────────────────────────────────────────────────

def tool_github_search(query: str, type: str = "repos", limit: int = 10) -> str:
    """Search GitHub for repos, issues, PRs, or code."""
    if type == "repos":
        args = ["search", "repos", query, "--limit", str(limit),
                "--json", "fullName,description,stargazersCount,language,updatedAt"]
        rc, output = _run_gh(*args)
        if rc != 0:
            return f"Error searching: {output}"
        try:
            results = json.loads(output)
        except json.JSONDecodeError:
            return output
        lines = []
        for r in results:
            stars = r.get("stargazersCount", 0) or r.get("stargazerCount", 0)
            lang = r.get("language", "") or ""
            desc = r.get("description", "") or ""
            if len(desc) > 60:
                desc = desc[:60] + "..."
            parts = [r.get("fullName", "?")]
            if lang:
                parts.append(lang)
            if stars:
                parts.append(f"★{stars}")
            if desc:
                parts.append(f"— {desc}")
            lines.append("  ".join(parts))
        return "\n".join(lines) if lines else "No results."
    elif type == "issues":
        args = ["search", "issues", query, "--limit", str(limit),
                "--json", "repository,number,title,state,author"]
        rc, output = _run_gh(*args)
        if rc != 0:
            return f"Error searching: {output}"
        try:
            results = json.loads(output)
        except json.JSONDecodeError:
            return output
        lines = []
        for r in results:
            repo_name = r.get("repository", {}).get("nameWithOwner", "?")
            author = r.get("author", {}).get("login", "?")
            lines.append(f"{repo_name}#{r.get('number')}  {r.get('title')}  [{r.get('state')}] by {author}")
        return "\n".join(lines) if lines else "No results."
    elif type == "prs":
        args = ["search", "prs", query, "--limit", str(limit),
                "--json", "repository,number,title,state,author"]
        rc, output = _run_gh(*args)
        if rc != 0:
            return f"Error searching: {output}"
        try:
            results = json.loads(output)
        except json.JSONDecodeError:
            return output
        lines = []
        for r in results:
            repo_name = r.get("repository", {}).get("nameWithOwner", "?")
            author = r.get("author", {}).get("login", "?")
            lines.append(f"{repo_name}#{r.get('number')}  {r.get('title')}  [{r.get('state')}] by {author}")
        return "\n".join(lines) if lines else "No results."
    elif type == "code":
        args = ["search", "code", query, "--limit", str(limit),
                "--json", "repository,path"]
        rc, output = _run_gh(*args)
        if rc != 0:
            return f"Error searching: {output}"
        try:
            results = json.loads(output)
        except json.JSONDecodeError:
            return output
        lines = []
        for r in results:
            repo_name = r.get("repository", {}).get("fullName", "?")
            lines.append(f"{repo_name}: {r.get('path', '?')}")
        return "\n".join(lines) if lines else "No results."
    else:
        return f"Unknown search type '{type}'. Use: repos, issues, prs, or code."


# ── Notifications ──────────────────────────────────────────────

def tool_github_notifications(limit: int = 10) -> str:
    """List unread GitHub notifications."""
    args = ["api", "notifications", "--method", "GET", "--jq",
            '.[] | "\\(.subject.type): \\(.subject.title)  [\\(.repository.full_name)]"',
            "-f", f"per_page={limit}"]
    rc, output = _run_gh(*args)
    if rc != 0:
        return f"Error fetching notifications: {output}"
    return output.strip() if output.strip() else "No unread notifications."


# ── Registry wrappers ──────────────────────────────────────────

@register_tool("github_list_prs")
def _dispatch_list_prs(args):
    return tool_github_list_prs(
        repo=args.get("repo", ""),
        state=args.get("state", "open"),
        limit=args.get("limit", 10),
    )

@register_tool("github_view_pr")
def _dispatch_view_pr(args):
    return tool_github_view_pr(
        pr_number=args.get("pr_number"),
        repo=args.get("repo", ""),
    )

@register_tool("github_create_pr")
def _dispatch_create_pr(args):
    return tool_github_create_pr(
        title=args.get("title", ""),
        body=args.get("body", ""),
        base=args.get("base", ""),
        head=args.get("head", ""),
        repo=args.get("repo", ""),
        draft=args.get("draft", False),
    )

@register_tool("github_merge_pr")
def _dispatch_merge_pr(args):
    return tool_github_merge_pr(
        pr_number=args.get("pr_number"),
        repo=args.get("repo", ""),
        method=args.get("method", "merge"),
        delete_branch=args.get("delete_branch", True),
    )

@register_tool("github_close_pr")
def _dispatch_close_pr(args):
    return tool_github_close_pr(
        pr_number=args.get("pr_number"),
        repo=args.get("repo", ""),
        comment=args.get("comment", ""),
    )

@register_tool("github_pr_diff")
def _dispatch_pr_diff(args):
    return tool_github_pr_diff(
        pr_number=args.get("pr_number"),
        repo=args.get("repo", ""),
    )

@register_tool("github_pr_comments")
def _dispatch_pr_comments(args):
    return tool_github_pr_comments(
        pr_number=args.get("pr_number"),
        repo=args.get("repo", ""),
    )

@register_tool("github_comment_pr")
def _dispatch_comment_pr(args):
    return tool_github_comment_pr(
        pr_number=args.get("pr_number"),
        body=args.get("body", ""),
        repo=args.get("repo", ""),
    )

@register_tool("github_list_issues")
def _dispatch_list_issues(args):
    return tool_github_list_issues(
        repo=args.get("repo", ""),
        state=args.get("state", "open"),
        labels=args.get("labels", ""),
        limit=args.get("limit", 10),
    )

@register_tool("github_view_issue")
def _dispatch_view_issue(args):
    return tool_github_view_issue(
        issue_number=args.get("issue_number"),
        repo=args.get("repo", ""),
    )

@register_tool("github_create_issue")
def _dispatch_create_issue(args):
    return tool_github_create_issue(
        title=args.get("title", ""),
        body=args.get("body", ""),
        labels=args.get("labels", ""),
        assignee=args.get("assignee", ""),
        repo=args.get("repo", ""),
    )

@register_tool("github_close_issue")
def _dispatch_close_issue(args):
    return tool_github_close_issue(
        issue_number=args.get("issue_number"),
        comment=args.get("comment", ""),
        repo=args.get("repo", ""),
    )

@register_tool("github_comment_issue")
def _dispatch_comment_issue(args):
    return tool_github_comment_issue(
        issue_number=args.get("issue_number"),
        body=args.get("body", ""),
        repo=args.get("repo", ""),
    )

@register_tool("github_list_repos")
def _dispatch_list_repos(args):
    return tool_github_list_repos(
        owner=args.get("owner", ""),
        limit=args.get("limit", 10),
    )

@register_tool("github_view_repo")
def _dispatch_view_repo(args):
    return tool_github_view_repo(
        repo=args.get("repo", ""),
    )

@register_tool("github_list_branches")
def _dispatch_list_branches(args):
    return tool_github_list_branches(
        repo=args.get("repo", ""),
        limit=args.get("limit", 20),
    )

@register_tool("github_list_commits")
def _dispatch_list_commits(args):
    return tool_github_list_commits(
        repo=args.get("repo", ""),
        branch=args.get("branch", ""),
        limit=args.get("limit", 10),
    )

@register_tool("github_list_runs")
def _dispatch_list_runs(args):
    return tool_github_list_runs(
        repo=args.get("repo", ""),
        limit=args.get("limit", 10),
        status=args.get("status", ""),
    )

@register_tool("github_view_run")
def _dispatch_view_run(args):
    return tool_github_view_run(
        run_id=args.get("run_id"),
        repo=args.get("repo", ""),
    )

@register_tool("github_rerun")
def _dispatch_rerun(args):
    return tool_github_rerun(
        run_id=args.get("run_id"),
        repo=args.get("repo", ""),
        failed_only=args.get("failed_only", False),
    )

@register_tool("github_list_releases")
def _dispatch_list_releases(args):
    return tool_github_list_releases(
        repo=args.get("repo", ""),
        limit=args.get("limit", 5),
    )

@register_tool("github_create_release")
def _dispatch_create_release(args):
    return tool_github_create_release(
        tag=args.get("tag", ""),
        title=args.get("title", ""),
        notes=args.get("notes", ""),
        draft=args.get("draft", False),
        prerelease=args.get("prerelease", False),
        repo=args.get("repo", ""),
    )

@register_tool("github_search")
def _dispatch_search(args):
    return tool_github_search(
        query=args.get("query", ""),
        type=args.get("type", "repos"),
        limit=args.get("limit", 10),
    )

@register_tool("github_notifications")
def _dispatch_notifications(args):
    return tool_github_notifications(
        limit=args.get("limit", 10),
    )
