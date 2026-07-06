"""Domain-owned tool schemas."""

TOOL_NAMES = {
    "github_close_issue",
    "github_close_pr",
    "github_comment_issue",
    "github_comment_pr",
    "github_create_issue",
    "github_create_pr",
    "github_create_release",
    "github_list_branches",
    "github_list_commits",
    "github_list_issues",
    "github_list_prs",
    "github_list_releases",
    "github_list_repos",
    "github_list_runs",
    "github_merge_pr",
    "github_notifications",
    "github_pr_comments",
    "github_pr_diff",
    "github_rerun",
    "github_search",
    "github_view_issue",
    "github_view_pr",
    "github_view_repo",
    "github_view_run",
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "github_list_prs",
            "description": "List pull requests for a GitHub repository. Defaults to the repo in the current directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository in owner/repo format (optional, defaults to current repo)",
                    },
                    "state": {
                        "type": "string",
                        "description": "Filter by state: open, closed, merged, all (default: open)",
                    },
                    "limit": {"type": "integer", "description": "Maximum PRs to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_view_pr",
            "description": "View details of a specific pull request including status checks, review state, and "
            "description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_pr",
            "description": "Create a new pull request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "PR title"},
                    "body": {"type": "string", "description": "PR description/body"},
                    "base": {
                        "type": "string",
                        "description": "Base branch to merge into (default: repo default branch)",
                    },
                    "head": {"type": "string", "description": "Head branch with changes (default: current branch)"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "draft": {"type": "boolean", "description": "Create as draft PR (default: false)"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_merge_pr",
            "description": "Merge a pull request. Supports merge, squash, or rebase strategies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number to merge"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "method": {
                        "type": "string",
                        "description": "Merge method: merge, squash, or rebase (default: merge)",
                    },
                    "delete_branch": {
                        "type": "boolean",
                        "description": "Delete the branch after merging (default: true)",
                    },
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_close_pr",
            "description": "Close a pull request without merging.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "comment": {"type": "string", "description": "Optional comment to leave before closing"},
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_pr_diff",
            "description": "View the diff/changes of a pull request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_pr_comments",
            "description": "List comments on a pull request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_comment_pr",
            "description": "Add a comment to a pull request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer", "description": "The pull request number"},
                    "body": {"type": "string", "description": "Comment text"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["pr_number", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_issues",
            "description": "List issues for a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "state": {"type": "string", "description": "Filter by state: open, closed, all (default: open)"},
                    "labels": {"type": "string", "description": "Filter by label name"},
                    "limit": {"type": "integer", "description": "Maximum issues to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_view_issue",
            "description": "View details of a specific issue including comments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_number": {"type": "integer", "description": "The issue number"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["issue_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_issue",
            "description": "Create a new issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue description"},
                    "labels": {"type": "string", "description": "Comma-separated label names"},
                    "assignee": {"type": "string", "description": "GitHub username to assign"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_close_issue",
            "description": "Close an issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_number": {"type": "integer", "description": "The issue number"},
                    "comment": {"type": "string", "description": "Optional comment to leave before closing"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["issue_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_comment_issue",
            "description": "Add a comment to an issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_number": {"type": "integer", "description": "The issue number"},
                    "body": {"type": "string", "description": "Comment text"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["issue_number", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_repos",
            "description": "List repositories for a user/org, or your own repos if no owner specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "GitHub username or org (optional, defaults to authenticated user)",
                    },
                    "limit": {"type": "integer", "description": "Maximum repos to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_view_repo",
            "description": "View detailed information about a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository in owner/repo format (optional, defaults to current repo)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_branches",
            "description": "List branches in a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "limit": {"type": "integer", "description": "Maximum branches to show (default 20)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_commits",
            "description": "List recent commits in a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "branch": {"type": "string", "description": "Branch name (default: default branch)"},
                    "limit": {"type": "integer", "description": "Maximum commits to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_runs",
            "description": "List recent GitHub Actions workflow runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "limit": {"type": "integer", "description": "Maximum runs to return (default 10)"},
                    "status": {
                        "type": "string",
                        "description": "Filter by status: queued, in_progress, completed, failure, success",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_view_run",
            "description": "View details of a specific workflow run including job results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer", "description": "The workflow run ID"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_rerun",
            "description": "Re-run a GitHub Actions workflow run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer", "description": "The workflow run ID to re-run"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "failed_only": {"type": "boolean", "description": "Only re-run failed jobs (default: false)"},
                },
                "required": ["run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_releases",
            "description": "List releases for a repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                    "limit": {"type": "integer", "description": "Maximum releases to return (default 5)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_release",
            "description": "Create a new GitHub release with a tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "Tag name for the release (e.g. v1.0.0)"},
                    "title": {"type": "string", "description": "Release title"},
                    "notes": {"type": "string", "description": "Release notes (auto-generated if omitted)"},
                    "draft": {"type": "boolean", "description": "Create as draft (default: false)"},
                    "prerelease": {"type": "boolean", "description": "Mark as pre-release (default: false)"},
                    "repo": {"type": "string", "description": "Repository in owner/repo format (optional)"},
                },
                "required": ["tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_search",
            "description": "Search GitHub for repositories, issues, pull requests, or code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "type": {
                        "type": "string",
                        "description": "What to search: repos, issues, prs, or code (default: repos)",
                        "enum": ["repos", "issues", "prs", "code"],
                    },
                    "limit": {"type": "integer", "description": "Maximum results (default 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_notifications",
            "description": "List unread GitHub notifications.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Maximum notifications (default 10)"}},
                "required": [],
            },
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
