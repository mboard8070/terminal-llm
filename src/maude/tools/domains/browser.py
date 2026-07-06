"""Domain-owned tool schemas."""

TOOL_NAMES = {
    "browser_check_session",
    "browser_click",
    "browser_close",
    "browser_extract",
    "browser_fill_form",
    "browser_login",
    "browser_navigate",
    "browser_open",
    "browser_screenshot",
    "browser_select",
    "browser_snapshot",
    "browser_type",
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "browser_open",
            "description": "Open a URL in a headless Chromium browser. Returns the page title and a text summary. "
            "Starts a persistent browser session that maintains cookies and state across calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to open (e.g. 'https://example.com')"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click an element on the current page. Accepts a snapshot ref (e.g. '@e5' from "
            "browser_snapshot), CSS selector, visible text, or aria-label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Snapshot ref (e.g. '@e5'), CSS selector, "
                        "visible text (e.g. 'Sign In'), or "
                        "aria-label",
                    }
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Type text into an input field or textarea on the current page. Use a snapshot ref (e.g. "
            "'@e3'), or 'active' for the focused element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Snapshot ref (e.g. '@e3'), CSS selector, placeholder text, label, or 'active'",
                    },
                    "text": {"type": "string", "description": "The text to type into the field"},
                },
                "required": ["selector", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Navigate to a new URL in the current browser session. Preserves cookies and login state "
            "from previous pages.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to navigate to"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_snapshot",
            "description": "Get a text-based accessibility tree of the current page. Interactive elements (buttons, "
            "links, inputs) are tagged with refs like [@e1], [@e2] that you can pass to "
            "browser_click or browser_type. Use this INSTEAD of browser_screenshot when you need to "
            "interact — it's faster and more precise than vision. Call after browser_open or "
            "browser_navigate.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "Take a screenshot of the current browser page and analyze it with the LLaVA vision "
            "model. Returns a description of the visual layout, content, and interactive elements.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_extract",
            "description": "Extract text content from the current page or a specific element. Limited to 10,000 "
            "characters. Use with no selector for full page text, or pass a CSS selector for a "
            "specific section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to extract from a "
                        "specific element. Omit for full page text.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_fill_form",
            "description": "Fill multiple form fields at once. Each key in the fields object is a CSS selector (or "
            "label/placeholder), and each value is the text to type into that field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "object",
                        "description": "Object mapping selectors to values. Example: "
                        '{"#username": "alice", "#password": "secret"}',
                        "additionalProperties": {"type": "string"},
                    }
                },
                "required": ["fields"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_select",
            "description": "Select an option from a dropdown/select element. Tries matching by value, then label, "
            "then numeric index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector for the <select> element"},
                    "value": {
                        "type": "string",
                        "description": "The option value, visible label, or numeric index to select",
                    },
                },
                "required": ["selector", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_login",
            "description": "Open a VISIBLE (non-headless) browser window for manual login to a website. Accepts "
            "shorthand names like 'x', 'linkedin', 'instagram', 'facebook', 'github', 'reddit', "
            "'google', 'tiktok', 'pinterest', 'bluesky' or a full URL. Opens a visible Chromium "
            "window on the local display for manual login. Requires a graphical session. Leave "
            "browser open after login.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Site to log into — shorthand name (e.g. 'x', "
                        "'linkedin', 'instagram') or full URL (e.g. "
                        "'https://example.com/login')",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_check_session",
            "description": "Check if a saved login session is still valid for a website. Opens the site headlessly "
            "and looks for a logged-in indicator element. Returns VALID if found, EXPIRED if not.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Site to check — shorthand name (e.g. 'x', 'linkedin') or full URL",
                    },
                    "logged_in_selector": {
                        "type": "string",
                        "description": "CSS selector, text, or aria-label "
                        "that indicates a logged-in state "
                        "(e.g. "
                        "'nav[aria-label=\"Primary\"]', "
                        "'Home', 'Profile')",
                    },
                },
                "required": ["url", "logged_in_selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_close",
            "description": "Close the browser session and free resources. The session will also auto-close after 5 "
            "minutes of inactivity.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
