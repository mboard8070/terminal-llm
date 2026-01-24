"""
MAUDE Skills/Plugins Framework.

Extensible tools that can be shared, installed, and managed.
"""

import os
import json
import importlib.util
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Any, Optional
from functools import wraps
from rich.console import Console

console = Console()


@dataclass
class SkillMetadata:
    """Metadata for a skill."""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "MAUDE"
    triggers: List[str] = None  # Keywords that activate this skill
    parameters: Dict[str, Any] = None  # JSON schema for parameters
    requires: List[str] = None  # Dependencies (pip packages)

    def __post_init__(self):
        self.triggers = self.triggers or [self.name]
        self.parameters = self.parameters or {}
        self.requires = self.requires or []


class Skill:
    """A MAUDE skill (plugin)."""

    def __init__(self, metadata: SkillMetadata, execute_fn: Callable):
        self.metadata = metadata
        self.execute = execute_fn
        self.enabled = True

    def to_tool_definition(self) -> dict:
        """Convert skill to OpenAI-compatible tool definition."""
        params = self.metadata.parameters or {
            "type": "object",
            "properties": {},
            "required": []
        }
        return {
            "type": "function",
            "function": {
                "name": f"skill_{self.metadata.name}",
                "description": self.metadata.description,
                "parameters": params
            }
        }


# Global skill registry
_skills: Dict[str, Skill] = {}


def skill(
    name: str,
    description: str = "",
    version: str = "1.0.0",
    author: str = "MAUDE",
    triggers: List[str] = None,
    parameters: Dict[str, Any] = None,
    requires: List[str] = None
):
    """Decorator to register a function as a skill."""
    def decorator(fn: Callable):
        metadata = SkillMetadata(
            name=name,
            description=description or fn.__doc__ or "",
            version=version,
            author=author,
            triggers=triggers,
            parameters=parameters,
            requires=requires
        )
        _skills[name] = Skill(metadata, fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


class SkillManager:
    """Manages skill discovery, loading, and execution."""

    USER_SKILLS_DIR = Path.home() / ".config" / "maude" / "skills"
    BUILTIN_DIR = Path(__file__).parent / "builtin"
    CONFIG_FILE = Path.home() / ".config" / "maude" / "skills.json"

    def __init__(self):
        self.USER_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_config()
        self._load_builtin_skills()
        self._load_user_skills()

    def _load_config(self):
        """Load skill configuration (enabled/disabled state)."""
        self.config = {}
        if self.CONFIG_FILE.exists():
            try:
                self.config = json.loads(self.CONFIG_FILE.read_text())
            except:
                pass

    def _save_config(self):
        """Save skill configuration."""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.CONFIG_FILE.write_text(json.dumps(self.config, indent=2))

    def _load_builtin_skills(self):
        """Load built-in skills."""
        if self.BUILTIN_DIR.exists():
            for skill_file in self.BUILTIN_DIR.glob("*.py"):
                if skill_file.name.startswith("_"):
                    continue
                self._load_skill_file(skill_file)

    def _load_user_skills(self):
        """Load user-installed skills."""
        for skill_file in self.USER_SKILLS_DIR.glob("*.py"):
            if skill_file.name.startswith("_"):
                continue
            self._load_skill_file(skill_file)

    def _load_skill_file(self, path: Path):
        """Load a skill from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            console.print(f"[yellow]Failed to load skill {path.name}: {e}[/yellow]")

    def reload_skills(self):
        """Reload all skills."""
        global _skills
        _skills.clear()
        self._load_builtin_skills()
        self._load_user_skills()
        # Apply enabled/disabled state from config
        for name, enabled in self.config.get("enabled", {}).items():
            if name in _skills:
                _skills[name].enabled = enabled

    def list_skills(self) -> List[Skill]:
        """List all registered skills."""
        return list(_skills.values())

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return _skills.get(name)

    def execute_skill(self, name: str, **kwargs) -> str:
        """Execute a skill."""
        skill = self.get_skill(name)
        if not skill:
            available = ", ".join(_skills.keys())
            return f"Error: Skill '{name}' not found. Available: {available}"
        if not skill.enabled:
            return f"Error: Skill '{name}' is disabled. Use /skills enable {name}"
        try:
            console.print(f"[dim cyan]  -> Running skill: {name}[/dim cyan]")
            result = skill.execute(**kwargs)
            return result
        except TypeError as e:
            # Handle missing required parameters
            return f"Error: Missing parameters for skill '{name}': {e}"
        except Exception as e:
            return f"Error executing skill '{name}': {e}"

    def enable_skill(self, name: str) -> str:
        """Enable a skill."""
        if name not in _skills:
            return f"Error: Skill '{name}' not found"
        _skills[name].enabled = True
        if "enabled" not in self.config:
            self.config["enabled"] = {}
        self.config["enabled"][name] = True
        self._save_config()
        return f"Skill '{name}' enabled"

    def disable_skill(self, name: str) -> str:
        """Disable a skill."""
        if name not in _skills:
            return f"Error: Skill '{name}' not found"
        _skills[name].enabled = False
        if "enabled" not in self.config:
            self.config["enabled"] = {}
        self.config["enabled"][name] = False
        self._save_config()
        return f"Skill '{name}' disabled"

    def get_tool_definitions(self) -> List[dict]:
        """Get OpenAI-compatible tool definitions for all enabled skills."""
        return [
            skill.to_tool_definition()
            for skill in _skills.values()
            if skill.enabled
        ]

    def get_skills_summary(self) -> str:
        """Get a formatted summary of skills for the LLM system prompt."""
        if not _skills:
            return ""

        lines = ["AVAILABLE SKILLS (use skill_<name> tool):"]
        for name, skill in _skills.items():
            status = "✓" if skill.enabled else "✗"
            lines.append(f"  [{status}] {name}: {skill.metadata.description}")
        return "\n".join(lines)


def handle_skills_command(args: list, manager: SkillManager) -> str:
    """Handle /skills command."""
    if not args:
        # List all skills
        skills = manager.list_skills()
        if not skills:
            return "No skills installed.\n\nBuilt-in skills are in: skills/builtin/\nUser skills go in: ~/.config/maude/skills/"

        lines = ["Installed Skills:\n"]
        for skill in skills:
            status = "[green]✓[/green]" if skill.enabled else "[red]✗[/red]"
            lines.append(f"  {status} {skill.metadata.name:15} - {skill.metadata.description}")
            lines.append(f"      v{skill.metadata.version} by {skill.metadata.author}")

        lines.append("\nCommands:")
        lines.append("  /skills enable <name>   - Enable a skill")
        lines.append("  /skills disable <name>  - Disable a skill")
        lines.append("  /skills info <name>     - Show skill details")
        lines.append("  /skills reload          - Reload all skills")
        lines.append("  /skills run <name> ...  - Run a skill directly")
        return "\n".join(lines)

    action = args[0].lower()

    if action == "enable" and len(args) > 1:
        return manager.enable_skill(args[1])

    elif action == "disable" and len(args) > 1:
        return manager.disable_skill(args[1])

    elif action == "reload":
        manager.reload_skills()
        return f"Reloaded {len(manager.list_skills())} skills"

    elif action == "info" and len(args) > 1:
        skill = manager.get_skill(args[1])
        if not skill:
            return f"Skill '{args[1]}' not found"
        m = skill.metadata
        lines = [
            f"Skill: {m.name}",
            f"Description: {m.description}",
            f"Version: {m.version}",
            f"Author: {m.author}",
            f"Status: {'Enabled' if skill.enabled else 'Disabled'}",
            f"Triggers: {', '.join(m.triggers)}",
        ]
        if m.requires:
            lines.append(f"Dependencies: {', '.join(m.requires)}")
        if m.parameters.get("properties"):
            lines.append("Parameters:")
            for pname, pinfo in m.parameters["properties"].items():
                req = " (required)" if pname in m.parameters.get("required", []) else ""
                lines.append(f"  - {pname}: {pinfo.get('description', pinfo.get('type', ''))}{req}")
        return "\n".join(lines)

    elif action == "run" and len(args) > 1:
        skill_name = args[1]
        # Parse remaining args as key=value pairs
        kwargs = {}
        for arg in args[2:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                # Try to parse as JSON for numbers/bools
                try:
                    kwargs[key] = json.loads(value)
                except:
                    kwargs[key] = value
            else:
                # Positional - use as first parameter
                skill = manager.get_skill(skill_name)
                if skill and skill.metadata.parameters.get("properties"):
                    first_param = list(skill.metadata.parameters["properties"].keys())[0]
                    kwargs[first_param] = arg

        return manager.execute_skill(skill_name, **kwargs)

    return f"Unknown skills command: {action}"


# Convenience function to get global manager
_manager: Optional[SkillManager] = None

def get_skill_manager() -> SkillManager:
    """Get or create the global skill manager."""
    global _manager
    if _manager is None:
        _manager = SkillManager()
    return _manager
