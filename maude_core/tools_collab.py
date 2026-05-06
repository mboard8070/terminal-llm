"""
Collaboration tools — lazy-import registrations.
"""

from tool_registry import register_tool


@register_tool("mesh_status")
def _dispatch_mesh_status(args):
    from collab_tools import execute_collab_tool

    return execute_collab_tool("mesh_status", args)


@register_tool("dispatch_task")
def _dispatch_dispatch_task(args):
    from collab_tools import execute_collab_tool

    return execute_collab_tool("dispatch_task", args)


@register_tool("create_project")
def _dispatch_create_project(args):
    from collab_tools import execute_collab_tool

    return execute_collab_tool("create_project", args)


@register_tool("list_projects")
def _dispatch_list_projects(args):
    from collab_tools import execute_collab_tool

    return execute_collab_tool("list_projects", args)


@register_tool("add_to_project")
def _dispatch_add_to_project(args):
    from collab_tools import execute_collab_tool

    return execute_collab_tool("add_to_project", args)


@register_tool("list_tasks")
def _dispatch_list_tasks(args):
    from collab_tools import execute_collab_tool

    return execute_collab_tool("list_tasks", args)
