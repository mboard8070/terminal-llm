#!/usr/bin/env python3
"""
MAUDE Client - Local interface connecting to Spark server for inference.

Run the SSH tunnel first:
  ssh -L 30000:localhost:30000 mboard76@spark-e26c -N &

Then run this client:
  python maude_client.py
"""

import os
import sys
import json
import time
import requests
from typing import Optional, Generator

from config import (
    SERVER_HOST, SERVER_LLM_PORT, MODEL_NAME,
    CONTEXT_SIZE, TEMPERATURE, CLIENT_NAME
)
from client_tools import TOOLS, execute_tool

# API endpoint
API_URL = f"http://{SERVER_HOST}:{SERVER_LLM_PORT}/v1/chat/completions"

# Conversation history
messages = []

# System prompt
SYSTEM_PROMPT = f"""You are MAUDE (Multi-Agent Unified Dispatch Engine), a helpful AI assistant.

You are running as a CLIENT on the user's local machine (Mac or PC), connected to
a Spark server for inference. You have access to:

LOCAL TOOLS (operate on the user's machine):
- read_file, write_file, edit_file - Local file operations
- list_directory - Browse local filesystem
- search_files - Search local files
- run_command - Run local shell commands

SERVER TOOLS (operate on Spark server):
- upload_to_server - Send local files to server
- download_from_server - Get files from server
- list_server_files - Browse server filesystem
- run_server_command - Run commands on server
- send_to_server_maude - Message the server MAUDE instance

When the user asks to work with files, clarify if they mean LOCAL files or SERVER files.
Use the appropriate tools based on context.

Current client: {CLIENT_NAME}
"""


def check_server_connection() -> bool:
    """Check if the LLM server is reachable."""
    try:
        response = requests.get(
            f"http://{SERVER_HOST}:{SERVER_LLM_PORT}/v1/models",
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def stream_chat(user_message: str) -> Generator[str, None, None]:
    """Send message and stream the response."""
    messages.append({"role": "user", "content": user_message})

    # Build request
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": TEMPERATURE,
        "max_tokens": 4096,
        "stream": True
    }

    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=300)
        response.raise_for_status()

        full_content = ""
        tool_calls = []
        current_tool_call = None

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')
            if not line.startswith('data: '):
                continue

            data = line[6:]
            if data == '[DONE]':
                break

            try:
                chunk = json.loads(data)
                delta = chunk.get('choices', [{}])[0].get('delta', {})

                # Handle content
                if 'content' in delta and delta['content']:
                    full_content += delta['content']
                    yield delta['content']

                # Handle tool calls
                if 'tool_calls' in delta:
                    for tc in delta['tool_calls']:
                        idx = tc.get('index', 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({
                                'id': '',
                                'type': 'function',
                                'function': {'name': '', 'arguments': ''}
                            })

                        if 'id' in tc:
                            tool_calls[idx]['id'] = tc['id']
                        if 'function' in tc:
                            if 'name' in tc['function']:
                                tool_calls[idx]['function']['name'] = tc['function']['name']
                            if 'arguments' in tc['function']:
                                tool_calls[idx]['function']['arguments'] += tc['function']['arguments']

            except json.JSONDecodeError:
                continue

        # Process tool calls if any
        if tool_calls and tool_calls[0]['function']['name']:
            yield "\n"

            # Save assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": tool_calls
            })

            # Execute each tool
            for tc in tool_calls:
                func_name = tc['function']['name']
                try:
                    args = json.loads(tc['function']['arguments'])
                except:
                    args = {}

                yield f"\n[Tool: {func_name}]\n"

                result = execute_tool(func_name, args)

                # Truncate long results
                if len(result) > 3000:
                    result = result[:3000] + "\n... (truncated)"

                yield f"{result}\n"

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc['id'],
                    "content": result
                })

            # Get follow-up response
            yield "\n"
            for chunk in stream_chat_continuation():
                yield chunk

        else:
            # No tool calls, save assistant message
            if full_content:
                messages.append({"role": "assistant", "content": full_content})

    except requests.exceptions.ConnectionError:
        yield "\n[Error: Cannot connect to server. Is the SSH tunnel running?]\n"
        yield "Run: ssh -L 30000:localhost:30000 mboard76@spark-e26c -N &\n"
    except Exception as e:
        yield f"\n[Error: {e}]\n"


def stream_chat_continuation() -> Generator[str, None, None]:
    """Continue chat after tool execution."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": TEMPERATURE,
        "max_tokens": 4096,
        "stream": True
    }

    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=300)
        response.raise_for_status()

        full_content = ""
        tool_calls = []

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')
            if not line.startswith('data: '):
                continue

            data = line[6:]
            if data == '[DONE]':
                break

            try:
                chunk = json.loads(data)
                delta = chunk.get('choices', [{}])[0].get('delta', {})

                if 'content' in delta and delta['content']:
                    full_content += delta['content']
                    yield delta['content']

                if 'tool_calls' in delta:
                    for tc in delta['tool_calls']:
                        idx = tc.get('index', 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({
                                'id': '',
                                'type': 'function',
                                'function': {'name': '', 'arguments': ''}
                            })
                        if 'id' in tc:
                            tool_calls[idx]['id'] = tc['id']
                        if 'function' in tc:
                            if 'name' in tc['function']:
                                tool_calls[idx]['function']['name'] = tc['function']['name']
                            if 'arguments' in tc['function']:
                                tool_calls[idx]['function']['arguments'] += tc['function']['arguments']

            except json.JSONDecodeError:
                continue

        # Handle recursive tool calls (limit depth)
        if tool_calls and tool_calls[0]['function']['name']:
            messages.append({
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": tool_calls
            })

            for tc in tool_calls:
                func_name = tc['function']['name']
                try:
                    args = json.loads(tc['function']['arguments'])
                except:
                    args = {}

                yield f"\n[Tool: {func_name}]\n"
                result = execute_tool(func_name, args)

                if len(result) > 3000:
                    result = result[:3000] + "\n... (truncated)"

                yield f"{result}\n"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc['id'],
                    "content": result
                })

            # One more continuation (prevent infinite loops)
            yield "\n"
            for chunk in final_response():
                yield chunk
        else:
            if full_content:
                messages.append({"role": "assistant", "content": full_content})

    except Exception as e:
        yield f"\n[Error: {e}]\n"


def final_response() -> Generator[str, None, None]:
    """Get final response without tool calls."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "temperature": TEMPERATURE,
        "max_tokens": 2048,
        "stream": True
    }

    try:
        response = requests.post(API_URL, json=payload, stream=True, timeout=120)
        full_content = ""

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8')
            if not line.startswith('data: '):
                continue
            data = line[6:]
            if data == '[DONE]':
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                if 'content' in delta and delta['content']:
                    full_content += delta['content']
                    yield delta['content']
            except:
                continue

        if full_content:
            messages.append({"role": "assistant", "content": full_content})

    except Exception as e:
        yield f"\n[Error: {e}]\n"


def print_banner():
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                     MAUDE CLIENT                               ║
    ║         Connected to Spark Server for Inference                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"    Server: {SERVER_HOST}:{SERVER_LLM_PORT}")
    print(f"    Model:  {MODEL_NAME}")
    print(f"    Client: {CLIENT_NAME}")
    print()


def main():
    """Main chat loop."""
    print_banner()

    # Check connection
    print("Checking server connection...", end=" ", flush=True)
    if check_server_connection():
        print("OK")
    else:
        print("FAILED")
        print("\nCannot connect to server. Make sure the SSH tunnel is running:")
        print(f"  ssh -L {SERVER_LLM_PORT}:localhost:{SERVER_LLM_PORT} mboard76@spark-e26c -N &")
        print("\nThen restart this client.")
        sys.exit(1)

    print("\nType 'quit' to exit, 'clear' to reset conversation.\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input.lower() == 'clear':
                messages.clear()
                print("Conversation cleared.")
                continue

            print("\nMAUDE: ", end="", flush=True)
            for chunk in stream_chat(user_input):
                print(chunk, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
