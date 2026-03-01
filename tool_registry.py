"""
Tool Registry — central dispatcher for MAUDE tool functions.

Provides:
  @register_tool(name, cacheable=False)   – decorator to register a handler
  register_prefix(prefix, module, func)   – register prefix-based dispatch
  get_handler(name) -> Callable | None     – lookup by exact name, then prefix
  is_cacheable(name) -> bool               – whether results should be cached
"""

from typing import Callable, Optional

_REGISTRY: dict[str, Callable] = {}
_CACHEABLE: set[str] = set()
_PREFIX_HANDLERS: dict[str, tuple[str, str]] = {}     # prefix -> (module_path, executor_name)
_PREFIX_CALLABLES: dict[str, Callable] = {}            # prefix -> callable(name, arguments)


def register_tool(name: str, *, cacheable: bool = False):
    """Decorator to register a tool handler function.

    Usage:
        @register_tool("read_file")
        def _dispatch_read_file(args):
            return tool_read_file(args.get("path", ""))
    """
    def decorator(func: Callable):
        _REGISTRY[name] = func
        if cacheable:
            _CACHEABLE.add(name)
        return func
    return decorator


def register_prefix(prefix: str, module_path: str = None, executor_name: str = None, *, handler: Callable = None):
    """Register a prefix-based handler for tool families.

    Either provide (module_path, executor_name) for lazy-import dispatch
    where the executor is called as executor(tool_name, arguments),
    or provide handler= as a callable(tool_name, arguments).

    Usage:
        register_prefix("browser_", "browser", "execute_browser_tool")
        register_prefix("skill_", handler=my_skill_handler)
    """
    if handler is not None:
        _PREFIX_CALLABLES[prefix] = handler
    else:
        _PREFIX_HANDLERS[prefix] = (module_path, executor_name)


def get_handler(name: str) -> Optional[Callable]:
    """Look up a handler: exact match first, then prefix fallback."""
    handler = _REGISTRY.get(name)
    if handler is not None:
        return handler

    # Prefix-based lookup: callables first, then lazy-import
    for prefix, callable_handler in _PREFIX_CALLABLES.items():
        if name.startswith(prefix):
            def _wrap_callable(h, tool_name):
                def _handler(arguments):
                    return h(tool_name, arguments)
                return _handler
            return _wrap_callable(callable_handler, name)

    for prefix, (module_path, executor_name) in _PREFIX_HANDLERS.items():
        if name.startswith(prefix):
            def _make_prefix_handler(mod, func, tool_name):
                def _handler(arguments):
                    import importlib
                    m = importlib.import_module(mod)
                    executor = getattr(m, func)
                    return executor(tool_name, arguments)
                return _handler
            return _make_prefix_handler(module_path, executor_name, name)

    return None


def is_cacheable(name: str) -> bool:
    """Return True if the tool's results should be cached."""
    return name in _CACHEABLE
