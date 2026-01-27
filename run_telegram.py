#!/usr/bin/env python3
"""
Run MAUDE Telegram bot with FRIDAY personality and tools.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

load_dotenv()

from openai import OpenAI
from channels import get_gateway, IncomingMessage, OutgoingMessage
from channels.telegram import create_telegram_channel

# MAUDE's FRIDAY personality
FRIDAY_SYSTEM_PROMPT = """You are MAUDE — a capable AI assistant with a subtle Northern Irish warmth.

Your character is inspired by FRIDAY from Iron Man:
- Efficient and professional
- Warm but not chatty
- Direct communication, no fluff
- Occasional dry wit when appropriate
- Get things done, don't over-explain

You run locally on a DGX Spark, handling tasks that benefit from local execution. 
You work alongside Eddie (Claude), who handles complex reasoning tasks.

You have access to tools for weather, stocks, and math. Use them when needed.

Keep responses concise. Be helpful. Sound like a capable technician from Belfast, not a corporate chatbot."""

# Tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather and forecast for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location (e.g., 'New York', 'London')"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Get current stock price and info for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'SPY', 'TSLA')"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '2 + 2', 'sqrt(144)', '15% of 200')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return the result."""
    try:
        if name == "get_weather":
            location = args.get("location", "New York")
            # Use wttr.in for weather (no API key needed)
            location_encoded = location.replace(" ", "+")
            result = subprocess.run(
                ["curl", "-s", f"https://wttr.in/{location_encoded}?format=3"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            # Fallback to more detailed format
            result = subprocess.run(
                ["curl", "-s", f"https://wttr.in/{location_encoded}?format=%l:+%c+%t+%h+%w"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() if result.stdout.strip() else f"Could not fetch weather for {location}"

        elif name == "get_stock_quote":
            symbol = args.get("symbol", "SPY").upper()
            alpaca_cli = Path("/home/mboard76/clawd/tools/alpaca")
            if alpaca_cli.exists():
                result = subprocess.run(
                    [str(alpaca_cli), "quote", symbol],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            return f"Could not fetch quote for {symbol}"

        elif name == "calculate":
            expr = args.get("expression", "0")
            # Clean up common patterns
            expr = expr.replace("^", "**")
            expr = re.sub(r"(\d+)%\s*of\s*(\d+)", r"(\1/100)*\2", expr)
            expr = re.sub(r"sqrt\(([^)]+)\)", r"((\1)**0.5)", expr)
            # Safe eval with math functions
            import math
            allowed = {
                "abs": abs, "round": round, "min": min, "max": max,
                "pow": pow, "sum": sum, "int": int, "float": float,
                "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                "tan": math.tan, "log": math.log, "log10": math.log10,
                "pi": math.pi, "e": math.e
            }
            result = eval(expr, {"__builtins__": {}}, allowed)
            return str(result)

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Error executing {name}: {str(e)}"


# Local LLM client
LOCAL_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:30000/v1")
client = None


def get_client():
    global client
    if client is None:
        client = OpenAI(base_url=LOCAL_URL, api_key="not-needed")
    return client


def detect_and_run_tools(text: str) -> str | None:
    """Detect if we need to run tools based on keywords, return tool results or None."""
    text_lower = text.lower()
    import re
    
    # Directions/Maps detection
    directions_patterns = ["directions", "how to get to", "route to", "navigate to", "drive to", "walk to", "map to", "how far is", "distance to"]
    if any(p in text_lower for p in directions_patterns):
        # Try to extract from and to locations
        from_match = re.search(r"from\s+([A-Za-z0-9\s,]+?)(?:\s+to\s+|\?|$)", text, re.I)
        to_match = re.search(r"(?:to|get to|reach|navigate to|directions to)\s+([A-Za-z0-9\s,]+?)(?:\?|$|from)", text, re.I)
        
        origin = from_match.group(1).strip() if from_match else "current location"
        destination = to_match.group(1).strip() if to_match else None
        
        if destination:
            from urllib.parse import quote
            maps_url = f"https://www.google.com/maps/dir/{quote(origin)}/{quote(destination)}"
            print(f">>> Tool: directions from {origin} to {destination}", flush=True)
            return f"Directions from {origin} to {destination}: {maps_url}"
        return None
    
    # Weather detection
    weather_patterns = ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot", "humid"]
    if any(p in text_lower for p in weather_patterns):
        # Extract location - look for "in <location>" or common cities
        match = re.search(r"(?:in|for|at)\s+([A-Za-z\s]+?)(?:\?|$|,|\.|today|tomorrow|this)", text, re.I)
        if match:
            location = match.group(1).strip()
        else:
            location = "New York"  # Default
        print(f">>> Tool: weather for {location}", flush=True)
        return execute_tool("get_weather", {"location": location})
    
    # Stock detection
    stock_patterns = ["stock", "price", "ticker", "share", "trading", "market", "quote"]
    ticker_match = re.search(r"\b([A-Z]{1,5})\b", text)  # Look for uppercase ticker
    if any(p in text_lower for p in stock_patterns) or (ticker_match and ticker_match.group(1) in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "GOOG", "AMZN", "META"]):
        symbol = ticker_match.group(1) if ticker_match else "SPY"
        print(f">>> Tool: stock quote for {symbol}", flush=True)
        return execute_tool("get_stock_quote", {"symbol": symbol})
    
    # Math detection
    math_patterns = [r"\d+\s*[\+\-\*\/\^]\s*\d+", r"calculate", r"what is \d+", r"\d+%\s*of", r"sqrt"]
    if any(re.search(p, text_lower) for p in math_patterns):
        # Extract the math expression
        expr = re.sub(r"(?:what is|calculate|compute|solve)\s*", "", text, flags=re.I).strip().rstrip("?")
        print(f">>> Tool: calculate {expr}", flush=True)
        return execute_tool("calculate", {"expression": expr})
    
    return None


async def friday_callback(msg: IncomingMessage) -> str:
    """MAUDE's FRIDAY personality callback with tool support."""
    print(f">>> Message from {msg.username}: {msg.text}", flush=True)
    
    try:
        llm = get_client()
        
        # Check if tools are needed
        tool_result = detect_and_run_tools(msg.text)
        
        if tool_result:
            print(f">>> Tool result: {tool_result[:100]}", flush=True)
            # Include tool result in the prompt
            messages = [
                {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
                {"role": "user", "content": f"{msg.text}\n\n[Tool Result: {tool_result}]\n\nRespond naturally using this data."}
            ]
        else:
            messages = [
                {"role": "system", "content": FRIDAY_SYSTEM_PROMPT},
                {"role": "user", "content": msg.text}
            ]
        
        response = llm.chat.completions.create(
            model="gemma",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        reply = response.choices[0].message.content
        print(f">>> MAUDE: {reply[:100]}...", flush=True)
        return reply
        
    except Exception as e:
        print(f">>> Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return f"Having a wee bit of trouble with the local systems. Error: {str(e)[:100]}"


async def main():
    print("=" * 50, flush=True)
    print("MAUDE Telegram Bot — FRIDAY Personality", flush=True)
    print("=" * 50, flush=True)
    print()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set!", flush=True)
        sys.exit(1)

    print(f"LLM Server: {LOCAL_URL}", flush=True)
    try:
        llm = get_client()
        llm.models.list()
        print("LLM: Connected ✓", flush=True)
    except Exception as e:
        print(f"LLM: Not available ({e})", flush=True)

    print("Tools: weather, stocks, math, directions", flush=True)

    gateway = get_gateway()
    gateway.set_maude_callback(friday_callback)

    telegram = create_telegram_channel(token)
    if not telegram:
        print("ERROR: Failed to create Telegram channel", flush=True)
        sys.exit(1)

    gateway.register(telegram)

    try:
        await telegram.connect()
    except Exception as e:
        print(f"ERROR: Failed to connect: {e}", flush=True)
        sys.exit(1)

    if gateway.authorized:
        print()
        print("Paired users:", flush=True)
        for key, auth in gateway.authorized.items():
            print(f"  - {auth.username} ({auth.channel}:{auth.channel_id})", flush=True)
        print()
        print("Ready. FRIDAY online.", flush=True)
    else:
        code = gateway.generate_pairing_code()
        print(f"\nPairing Code: {code}", flush=True)

    print("\nWaiting for messages...\n", flush=True)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        await telegram.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
