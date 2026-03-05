"""
GitHub tool implementations — list PRs, view PR details, merge PRs via gh CLI.
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

@register_tool("github_merge_pr")
def _dispatch_merge_pr(args):
    return tool_github_merge_pr(
        pr_number=args.get("pr_number"),
        repo=args.get("repo", ""),
        method=args.get("method", "merge"),
        delete_branch=args.get("delete_branch", True),
    )
