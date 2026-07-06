"""Domain-owned tool schemas."""

TOOL_NAMES = {
    "activity_feed",
    "add_to_project",
    "create_project",
    "dispatch_task",
    "execute_plan",
    "gpu_processes",
    "list_projects",
    "list_tasks",
    "mesh_status",
    "node_status",
    "palace_kg_add_fact",
    "palace_kg_query",
    "palace_recall",
    "palace_search",
    "palace_status",
    "run_agent",
    "run_agents",
    "scheduler_status",
    "session_list",
    "skill_calc",
    "skill_convert",
    "skill_copy_clipboard",
    "skill_crypto",
    "skill_datetime",
    "skill_generate_3d",
    "skill_generate_image",
    "skill_hyperframes",
    "skill_note",
    "skill_paste_clipboard",
    "skill_schedule",
    "skill_screenshot",
    "skill_stock",
    "skill_system_info",
    "skill_todo",
    "skill_weather",
    "system_stats",
    "workflow_create",
    "workflow_delete",
    "workflow_get",
    "workflow_history",
    "workflow_list",
    "workflow_run",
    "workflow_schedule",
    "workflow_unschedule",
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "palace_search",
            "description": "Optional deep-archive search over MemPalace drawers. Use only when recall_memory does "
            "not find enough context or when the user explicitly asks for MemPalace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "wing": {"type": "string", "description": "Optional wing name to narrow the search"},
                    "room": {"type": "string", "description": "Optional room name to narrow the search"},
                    "n_results": {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "palace_recall",
            "description": "Browse MemPalace drawers by wing/room without a query (L2 on-demand retrieval). Use "
            "when you want to see everything in a topic area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wing": {"type": "string", "description": "Wing name"},
                    "room": {"type": "string", "description": "Room name"},
                    "n_results": {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "palace_status",
            "description": "Show MemPalace stats: palace path, total drawers, and whether L0 identity is configured.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "palace_kg_add_fact",
            "description": "Record a structured fact in the MemPalace knowledge graph as a (subject, predicate, "
            "object) triple. Use for durable relational facts like 'Matt works_at NVIDIA' that "
            "should survive rewrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Entity the fact is about"},
                    "predicate": {"type": "string", "description": "Relationship (e.g. 'works_at', 'owns', 'prefers')"},
                    "object": {"type": "string", "description": "Target of the relationship"},
                    "confidence": {"type": "number", "description": "0.0–1.0 confidence (default 1.0)"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "palace_kg_query",
            "description": "Query the MemPalace knowledge graph for all facts involving an entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity name to look up"},
                    "direction": {
                        "type": "string",
                        "description": "Relationship direction (default 'outgoing')",
                        "enum": ["outgoing", "incoming", "both"],
                    },
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_agent",
            "description": "Dispatch a task to a specialized agent that has tool access (can search the web, "
            "read/write files, run commands).\n"
            "Available agents:\n"
            "- 'code': Code generation, debugging, refactoring (has file ops + shell)\n"
            "- 'research': Multi-step web research, gathering info (has web + file read)\n"
            "- 'writer': Documentation, long-form content (has file ops + web)\n"
            "- 'reasoning': Complex analysis, planning (has file ops + web + shell)\n"
            "- 'search': Quick web lookups (has web_search + web_browse)\n"
            "\n"
            "Use this for tasks that require the agent to DO work (search, read, write) — not just "
            "answer from knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": ["code", "research", "writer", "reasoning", "search"],
                        "description": "Which specialized agent to use",
                    },
                    "task": {"type": "string", "description": "Clear description of what the agent should do"},
                    "context": {
                        "type": "string",
                        "description": "Relevant context (code snippets, file paths, requirements)",
                    },
                },
                "required": ["agent", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_agents",
            "description": "Dispatch multiple tasks to agents in parallel. Each agent runs independently with its "
            "own tool access. Use this when you need to research/investigate multiple things at "
            "once, or split a large task across specialists.\n"
            "\n"
            "Example: research two topics simultaneously, or have one agent write code while another "
            "researches docs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent": {
                                    "type": "string",
                                    "enum": ["code", "research", "writer", "reasoning", "search"],
                                    "description": "Which agent to use",
                                },
                                "task": {"type": "string", "description": "What this agent should do"},
                                "context": {"type": "string", "description": "Optional context for this agent"},
                            },
                            "required": ["agent", "task"],
                        },
                        "description": "Array of agent tasks to run in parallel",
                    }
                },
                "required": ["tasks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_plan",
            "description": "Execute a multi-stage tool plan. Each stage's tools run in parallel; stages run "
            "sequentially. Use this when you need multiple tools across dependent steps — it "
            "collapses several round-trips into one call.\n"
            "\n"
            "Reference earlier results in later stages with $N.M (stage N, tool index M). Example: "
            "$0.1 = result from stage 0, second tool.\n"
            "\n"
            "Example plan — search 3 files in parallel, then edit based on findings:\n"
            "stages: [\n"
            '  [{"name": "search_file", "args": {"path": "src/main.py", "pattern": "TODO"}},\n'
            '   {"name": "search_file", "args": {"path": "src/utils.py", "pattern": "TODO"}},\n'
            '   {"name": "read_file", "args": {"path": "README.md"}}],\n'
            '  [{"name": "run_command", "args": {"command": "grep -rn \'FIXME\' src/"}}]\n'
            "]",
            "parameters": {
                "type": "object",
                "properties": {
                    "stages": {
                        "type": "array",
                        "description": "Ordered list of stages. Each stage is an array "
                        "of tool calls that run in parallel.",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Tool name"},
                                    "args": {"type": "object", "description": "Tool arguments"},
                                },
                                "required": ["name", "args"],
                            },
                        },
                    }
                },
                "required": ["stages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_stats",
            "description": "Get system stats: CPU, RAM, disk usage, GPU temperature/utilization/memory. Use when "
            "the user asks about system health, resource usage, or monitoring.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gpu_processes",
            "description": "Show what processes are currently using the GPU and how much VRAM each is consuming.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "session_list",
            "description": "List recent conversation sessions across all channels (CLI, Telegram, etc.) with "
            "message counts and timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max sessions to return (default 20, max 50)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activity_feed",
            "description": "Show recent activity from the chat sync log — messages across all channels with "
            "timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of recent activities (default 20, max 100)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scheduler_status",
            "description": "Show all scheduled MAUDE tasks — cron schedules, run counts, last results, "
            "enabled/disabled status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "node_status",
            "description": "Show connected nodes and service status — Spark services (gateway, llama, telegram), "
            "Tailscale peers, remote client heartbeats.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_schedule",
            "description": "Schedule tasks for MAUDE to run automatically. Supports natural language schedules like "
            "'every morning', 'weekdays at 9am', 'every 30 minutes', 'in 2 hours'. Actions: schedule "
            "(create), list, cancel, pause, resume, run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["schedule", "list", "cancel", "pause", "resume", "run"],
                        "description": "Action to perform",
                        "default": "list",
                    },
                    "task_name": {
                        "type": "string",
                        "description": "Name/label for the scheduled task (for 'schedule' action)",
                    },
                    "when": {
                        "type": "string",
                        "description": "When to run — natural language ('every morning', "
                        "'daily at 3pm', 'every 30 minutes', 'in 2 "
                        "hours') or cron expression",
                    },
                    "prompt": {"type": "string", "description": "What MAUDE should do when the task runs"},
                    "task_id": {"type": "string", "description": "Task ID (for cancel/pause/resume/run actions)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_system_info",
            "description": "Get system information (CPU, memory, disk, GPU, network)",
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "enum": ["all", "cpu", "memory", "disk", "gpu", "network", "os"],
                        "description": "Which component to query (default: all)",
                        "default": "all",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_calc",
            "description": "Evaluate mathematical expressions safely",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_convert",
            "description": "Convert between units (length, weight, temperature, data)",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_note",
            "description": "Save and retrieve quick notes",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["save", "get", "list", "delete", "search"],
                        "description": "Action to perform",
                        "default": "list",
                    },
                    "key": {"type": "string", "description": "Note key/title"},
                    "content": {"type": "string", "description": "Note content (for save action)"},
                    "query": {"type": "string", "description": "Search query (for search action)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_todo",
            "description": "Simple todo list management",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "list", "done", "remove", "clear"],
                        "description": "Action to perform",
                        "default": "list",
                    },
                    "task": {"type": "string", "description": "Task description or number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_generate_image",
            "description": "Generate images using Flux AI via ComfyUI. Automatically routes to the best available "
            "endpoint on the mesh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the image to generate"},
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default: 1024)",
                        "default": 1024,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels (default: 1024)",
                        "default": 1024,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (-1 for random)",
                        "default": -1,
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of sampling steps (default: 28)",
                        "default": 28,
                    },
                    "lora": {
                        "type": "string",
                        "description": "Optional LoRA name (e.g. 'stillion-style', 'marker-mech-style')",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_datetime",
            "description": "Get current time, convert timezones, calculate date differences",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["now", "convert", "diff", "add"],
                        "description": "Action to perform",
                        "default": "now",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (e.g., 'America/New_York', 'Europe/London', 'UTC')",
                    },
                    "from_tz": {"type": "string", "description": "Source timezone for conversion"},
                    "to_tz": {"type": "string", "description": "Target timezone for conversion"},
                    "date1": {"type": "string", "description": "First date (YYYY-MM-DD format)"},
                    "date2": {"type": "string", "description": "Second date (YYYY-MM-DD format)"},
                    "days": {"type": "integer", "description": "Number of days to add/subtract"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_generate_3d",
            "description": "Generate 3D models from text or images. Routes to the best available 3D generation "
            "endpoint on the mesh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text description of the 3D model to generate"},
                    "image_path": {"type": "string", "description": "Optional path to an image to use as reference"},
                    "output_format": {
                        "type": "string",
                        "enum": ["glb", "obj", "fbx", "stl"],
                        "description": "Output format for the 3D model",
                        "default": "glb",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Specific provider to use (auto, meshy, local, etc.)",
                        "default": "auto",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["generate", "status", "list"],
                        "description": "Action to perform",
                        "default": "generate",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_stock",
            "description": "Get stock prices, quotes, and basic financial data",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., 'AAPL', 'NVDA', 'GOOGL')"},
                    "action": {
                        "type": "string",
                        "enum": ["quote", "info", "compare"],
                        "description": "Action to perform (default: quote)",
                        "default": "quote",
                    },
                    "symbols": {
                        "type": "string",
                        "description": "Comma-separated symbols for compare action (e.g., 'AAPL,NVDA,MSFT')",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_crypto",
            "description": "Get cryptocurrency prices",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Crypto symbol (e.g., 'BTC', 'ETH', 'SOL')",
                        "default": "BTC",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_hyperframes",
            "description": "Create, lint, diagnose, and render HyperFrames HTML/CSS/JS video compositions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["doctor", "browser_ensure", "init", "lint", "render"],
                        "description": "HyperFrames action to run. Default: doctor",
                        "default": "doctor",
                    },
                    "project_path": {"type": "string", "description": "Path to a HyperFrames project for lint/render"},
                    "name": {"type": "string", "description": "Project name for init"},
                    "example": {"type": "string", "description": "HyperFrames init example/template. Default: blank"},
                    "tailwind": {"type": "boolean", "description": "Enable Tailwind support during init"},
                    "install_skills": {
                        "type": "boolean",
                        "description": "Allow HyperFrames init to install its own agent skills",
                    },
                    "output": {"type": "string", "description": "Optional render output path"},
                    "format": {"type": "string", "enum": ["mp4", "mov", "webm"], "description": "Render format"},
                    "fps": {"type": "integer", "enum": [24, 30, 60], "description": "Render frame rate"},
                    "quality": {
                        "type": "string",
                        "enum": ["draft", "standard", "high"],
                        "description": "Render quality",
                    },
                    "docker": {"type": "boolean", "description": "Use Docker mode for deterministic rendering"},
                    "gpu": {"type": "boolean", "description": "Use hardware video encoding when available"},
                    "workers": {"type": "string", "description": "Render worker count, e.g. auto, 1, 2, 4"},
                    "share": {
                        "type": "boolean",
                        "description": "Copy rendered video to Maude's shared download folder",
                    },
                    "timeout": {"type": "integer", "description": "Render timeout in seconds"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_screenshot",
            "description": "Take a screenshot of a remote device's screen. Captures the screen, transfers the image "
            "back, and saves it locally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Target device — hostname, client_id, or "
                        "platform (e.g. 'windows', 'macos'). Leave "
                        "empty for default.",
                        "default": "",
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Custom save path for the screenshot. Leave "
                        "empty for auto-generated path in "
                        "the configured screenshots directory.",
                        "default": "",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_copy_clipboard",
            "description": "Copy text to a device's clipboard. Sends text to the clipboard of the target machine "
            "(Mac, Windows, or Linux).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to copy to the clipboard"},
                    "target": {
                        "type": "string",
                        "description": "Target device — hostname, client_id, or "
                        "platform (e.g. 'windows', 'macos', 'macbook'). "
                        "Leave empty for default.",
                        "default": "",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_paste_clipboard",
            "description": "Read/paste the clipboard contents from a device. Returns whatever text is currently on "
            "the target machine's clipboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Target device — hostname, client_id, or "
                        "platform (e.g. 'windows', 'macos', 'macbook'). "
                        "Leave empty for default.",
                        "default": "",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_weather",
            "description": "Get current weather and forecast for any location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location (e.g., 'New York', 'Charleston, WV')",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mesh_status",
            "description": "Show who's online across all devices and what they're doing. Shows presence, recent "
            "activity, and task status across the MAUDE mesh network.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_task",
            "description": "Dispatch a task to a specific device or platform on the mesh network. Use capability "
            "'SHELL' for shell commands or 'LLM' for AI tasks. IMPORTANT: do NOT invent a client_id "
            "— call mesh_status first to get the exact client_id of an online device, then pass it "
            "as target_client_id. For platform targeting, use target_platform with 'windows', "
            "'macos', or 'linux'. Client-targeted SHELL tasks are picked up by the client within ~10 "
            "seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The task prompt or command to execute"},
                    "target": {
                        "type": "string",
                        "description": "Target hostname, client_id, or platform name. Leave empty for local execution.",
                    },
                    "capability": {
                        "type": "string",
                        "description": "Task type: 'SHELL' for commands, 'LLM' for AI tasks",
                        "enum": ["SHELL", "LLM"],
                    },
                    "project_id": {"type": "string", "description": "Optional project ID to associate with this task"},
                    "target_client_id": {
                        "type": "string",
                        "description": "The exact client_id of an online "
                        "device, obtained by calling "
                        "mesh_status. Do not guess or invent "
                        "this value — if you don't have the "
                        "real client_id, call mesh_status "
                        "first.",
                    },
                    "target_platform": {
                        "type": "string",
                        "description": "Target a platform: 'windows', "
                        "'macos', or 'linux'. Only use if at "
                        "least one client of that platform is "
                        "currently online (check with "
                        "mesh_status).",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": "Create a new collaboration project to group conversations, files, and tasks together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name"},
                    "description": {"type": "string", "description": "Project description"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "List all collaboration projects across the mesh.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_project",
            "description": "Add a conversation or file to a project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID"},
                    "conversation_id": {"type": "string", "description": "Conversation ID to link"},
                    "file_path": {"type": "string", "description": "File path to link"},
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List dispatched tasks and their status (pending, running, completed, failed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status",
                        "enum": ["pending", "running", "completed", "failed"],
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_create",
            "description": "Create a repeatable browser workflow. Each step has an 'action' (open, navigate, click, "
            "type, extract, screenshot, fill_form, select, wait, close), a 'label' for "
            "identification, and action-specific parameters. Extract steps are tracked for change "
            "detection between runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Human-readable workflow name (e.g. 'Competitor Price Monitor')",
                    },
                    "steps": {
                        "type": "array",
                        "description": "List of steps. Each step is an object with "
                        "'action', 'label', and action-specific fields: "
                        "open/navigate need 'url', click/extract need "
                        "'selector', type needs 'selector' and 'text', "
                        "fill_form needs 'fields' (object), select needs "
                        "'selector' and 'value', wait needs 'seconds'.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "open",
                                        "navigate",
                                        "click",
                                        "type",
                                        "extract",
                                        "screenshot",
                                        "fill_form",
                                        "select",
                                        "wait",
                                        "close",
                                    ],
                                },
                                "label": {"type": "string"},
                                "url": {"type": "string"},
                                "selector": {"type": "string"},
                                "text": {"type": "string"},
                                "seconds": {"type": "number"},
                                "value": {"type": "string"},
                                "fields": {"type": "object", "additionalProperties": {"type": "string"}},
                            },
                            "required": ["action", "label"],
                        },
                    },
                    "description": {"type": "string", "description": "What this workflow does and why"},
                    "notify_email": {
                        "type": "string",
                        "description": "Email address to notify when changes are detected (optional)",
                    },
                },
                "required": ["name", "steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_run",
            "description": "Execute a saved browser workflow. Runs all steps sequentially, compares extract results "
            "to the previous run for change detection, and sends email notification if changes are "
            "found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "The workflow ID (lowercase-hyphenated name)"}
                },
                "required": ["workflow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_list",
            "description": "List all saved browser workflows with their step counts and schedules.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_get",
            "description": "View the full definition of a saved workflow including all steps.",
            "parameters": {
                "type": "object",
                "properties": {"workflow_id": {"type": "string", "description": "The workflow ID"}},
                "required": ["workflow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_delete",
            "description": "Delete a saved workflow and its run history.",
            "parameters": {
                "type": "object",
                "properties": {"workflow_id": {"type": "string", "description": "The workflow ID to delete"}},
                "required": ["workflow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_history",
            "description": "View recent run history for a workflow — timestamps, change counts, and errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "The workflow ID"},
                    "limit": {"type": "integer", "description": "Number of recent runs to show (default 5)"},
                },
                "required": ["workflow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_schedule",
            "description": "Schedule a workflow to run automatically on a cron expression. Examples: '0 8 * * *' "
            "(daily 8am), '0 */4 * * *' (every 4 hours), '0 9 * * 1-5' (weekday mornings).",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "The workflow ID to schedule"},
                    "cron": {"type": "string", "description": "Cron expression (e.g. '0 8 * * *' for daily at 8am)"},
                },
                "required": ["workflow_id", "cron"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_unschedule",
            "description": "Remove the cron schedule from a workflow. The workflow can still be run manually.",
            "parameters": {
                "type": "object",
                "properties": {"workflow_id": {"type": "string", "description": "The workflow ID to unschedule"}},
                "required": ["workflow_id"],
            },
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
