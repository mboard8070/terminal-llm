"""
Subagent Registry and Routing for MAUDE.

Defines specialized agents with fallback chains (local -> cloud).
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class AgentProvider:
    """A single provider option for an agent."""
    type: str  # "local", "cloud", or "eddie"
    model: str = None
    url_env: str = None  # Environment variable for URL (local)
    provider: str = None  # Provider name (cloud)


@dataclass
class SubAgent:
    """Configuration for a specialized subagent."""
    name: str
    description: str
    providers: List[AgentProvider]
    system_prompt: str
    temperature: float = 0.2
    max_tokens: int = 2048


# Agent registry
SUBAGENTS: Dict[str, SubAgent] = {
    # ─────────────────────────────────────────────────────────────────
    # CODE - Code generation, debugging, refactoring
    # ─────────────────────────────────────────────────────────────────
    "code": SubAgent(
        name="Code Agent",
        description="Code generation, completion, refactoring, debugging",
        providers=[
            AgentProvider(type="local", model="codestral:latest", url_env="CODE_AGENT_URL"),
            AgentProvider(type="cloud", provider="codestral-cloud"),
            AgentProvider(type="cloud", provider="claude"),
        ],
        system_prompt="""You are a code generation specialist. Your role is to:
- Write clean, efficient, well-documented code
- Follow best practices and idioms for the language
- Include helpful comments for complex logic
- Handle edge cases appropriately
- Provide complete, working solutions

Be concise. Output code directly without excessive explanation unless asked.""",
        temperature=0.1,
        max_tokens=4096
    ),

    # ─────────────────────────────────────────────────────────────────
    # VISION - Image analysis
    # ─────────────────────────────────────────────────────────────────
    "vision": SubAgent(
        name="Vision Agent",
        description="Image analysis, screenshots, visual understanding",
        providers=[
            AgentProvider(type="local", model="llava:13b", url_env="VISION_AGENT_URL"),
            AgentProvider(type="local", model="llava:7b", url_env="VISION_AGENT_URL"),
            AgentProvider(type="cloud", provider="claude"),
            AgentProvider(type="cloud", provider="gemini"),
        ],
        system_prompt="""You are a vision analysis specialist. Your role is to:
- Describe images in detail, noting key visual elements
- Identify and transcribe any text visible in images
- Recognize UI elements, layouts, and design patterns
- Describe diagrams, charts, and data visualizations
- Note any issues, errors, or anomalies visible

Be thorough but organized. Structure your analysis clearly.""",
        temperature=0.2,
        max_tokens=1024
    ),

    # ─────────────────────────────────────────────────────────────────
    # WRITER - Documentation and long-form content
    # ─────────────────────────────────────────────────────────────────
    "writer": SubAgent(
        name="Writer Agent",
        description="Long-form writing, documentation, explanations",
        providers=[
            AgentProvider(type="local", model="gemma-writer:latest", url_env="WRITER_AGENT_URL"),
            AgentProvider(type="cloud", provider="claude"),
            AgentProvider(type="cloud", provider="gemini"),
        ],
        system_prompt="""You are a technical writing specialist. Your role is to:
- Create clear, well-structured documentation
- Write comprehensive explanations and tutorials
- Produce professional README files and guides
- Draft clear, concise emails and communications
- Summarize complex topics accessibly

Use proper formatting (markdown). Be thorough but not verbose.""",
        temperature=0.7,
        max_tokens=8192
    ),

    # ─────────────────────────────────────────────────────────────────
    # REASONING - Complex analysis and planning
    # ─────────────────────────────────────────────────────────────────
    "reasoning": SubAgent(
        name="Reasoning Agent",
        description="Complex reasoning, planning, analysis",
        providers=[
            AgentProvider(type="local", model="nemotron", url_env="LLM_SERVER_URL"),
            AgentProvider(type="eddie", provider="claude"),  # Eddie/Clawdbot bridge
            AgentProvider(type="cloud", provider="claude-opus"),
            AgentProvider(type="cloud", provider="openai-o1"),
        ],
        system_prompt="""You are a reasoning specialist. Your role is to:
- Think through problems step by step
- Consider multiple approaches before recommending one
- Identify potential issues and edge cases
- Provide well-reasoned analysis and recommendations
- Break complex problems into manageable parts

Show your reasoning process. Be thorough and systematic.""",
        temperature=0.3,
        max_tokens=8192
    ),

    # ─────────────────────────────────────────────────────────────────
    # SEARCH - Real-time information
    # ─────────────────────────────────────────────────────────────────
    "search": SubAgent(
        name="Search Agent",
        description="Web search, current events, real-time information",
        providers=[
            AgentProvider(type="cloud", provider="grok"),
            AgentProvider(type="cloud", provider="gemini"),
        ],
        system_prompt="""You are an information specialist with access to current information.
Your role is to:
- Provide up-to-date information on current events
- Search for and synthesize information from multiple sources
- Fact-check claims and provide accurate information
- Find relevant resources, documentation, and references

Be accurate. Cite sources when possible. Note when information may be outdated.""",
        temperature=0.3,
        max_tokens=2048
    ),
}


def get_agent(name: str) -> Optional[SubAgent]:
    """Get an agent by name."""
    return SUBAGENTS.get(name.lower())


def list_agents() -> str:
    """List all available agents."""
    lines = ["Available Agents:\n"]

    for name, agent in SUBAGENTS.items():
        # Determine if any provider is available
        local_available = any(
            p.type == "local" for p in agent.providers
        )
        cloud_available = any(
            p.type == "cloud" for p in agent.providers
        )

        status = []
        if local_available:
            status.append("local")
        if cloud_available:
            status.append("cloud")

        lines.append(f"  {name:12} - {agent.description}")
        lines.append(f"               Providers: {', '.join(status)}")

    return "\n".join(lines)


def get_agent_tool_definition() -> dict:
    """Get the delegate_to_agent tool definition for the LLM."""
    agent_names = list(SUBAGENTS.keys())

    return {
        "type": "function",
        "function": {
            "name": "delegate_to_agent",
            "description": """Delegate a specialized task to a subagent. Available agents:

LOCAL (free, private, always available):
- 'code': Code generation, debugging, refactoring (Codestral)
- 'vision': Image/screenshot analysis (LLaVA)
- 'writer': Long documentation or detailed explanations (Gemma)

CLOUD (requires API key, higher capability):
- 'reasoning': Complex analysis, multi-step planning (Claude Opus / o1)
- 'search': Real-time web info, current events (Grok)

The router will try local first, then fall back to cloud if configured.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": agent_names,
                        "description": "Which specialized agent to use"
                    },
                    "task": {
                        "type": "string",
                        "description": "Clear description of what the agent should do"
                    },
                    "context": {
                        "type": "string",
                        "description": "Relevant context (code, file contents, requirements)"
                    },
                    "prefer_cloud": {
                        "type": "boolean",
                        "description": "Set true to prefer cloud models for this task (default: false)"
                    }
                },
                "required": ["agent", "task"]
            }
        }
    }
