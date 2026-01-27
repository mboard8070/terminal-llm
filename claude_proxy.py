#!/usr/bin/env python3
"""
Claude Proxy for MAUDE - Uses Clawdbot's OAuth tokens.

This allows MAUDE to use Claude without a separate API key,
sharing Eddie's OAuth authentication.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import anthropic
from anthropic import Anthropic

# Clawdbot auth file location
CLAWDBOT_AUTH_FILE = Path.home() / ".clawdbot" / "agents" / "main" / "agent" / "auth-profiles.json"


def load_oauth_tokens() -> Optional[Dict[str, Any]]:
    """Load OAuth tokens from Clawdbot's auth file."""
    if not CLAWDBOT_AUTH_FILE.exists():
        print(f"Auth file not found: {CLAWDBOT_AUTH_FILE}", file=sys.stderr)
        return None
    
    with open(CLAWDBOT_AUTH_FILE) as f:
        data = json.load(f)
    
    profiles = data.get("profiles", {})
    
    # Find the Anthropic OAuth profile
    for name, profile in profiles.items():
        if "anthropic" in name.lower() and profile.get("access"):
            return {
                "access_token": profile["access"],
                "refresh_token": profile.get("refresh"),
                "expires": profile.get("expires", 0)
            }
    
    return None


def get_claude_client() -> Optional[Anthropic]:
    """Get an authenticated Claude client using Clawdbot's OAuth."""
    tokens = load_oauth_tokens()
    if not tokens:
        print("No Anthropic OAuth tokens found in Clawdbot config", file=sys.stderr)
        return None
    
    # Check if token is expired (with 5 min buffer)
    if tokens["expires"] < (time.time() * 1000) + 300000:
        print("Warning: OAuth token may be expired. Eddie needs to refresh it.", file=sys.stderr)
    
    # Create client with OAuth token
    return Anthropic(api_key=tokens["access_token"])


def call_claude(
    prompt: str,
    system: str = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    temperature: float = 0.3
) -> Optional[str]:
    """Make a Claude API call using Clawdbot's OAuth."""
    client = get_claude_client()
    if not client:
        return None
    
    try:
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature
        }
        
        if system:
            kwargs["system"] = system
        
        response = client.messages.create(**kwargs)
        return response.content[0].text
        
    except anthropic.AuthenticationError as e:
        print(f"Auth error (token may need refresh): {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Claude API error: {e}", file=sys.stderr)
        return None


# Export the access token for use in MAUDE's provider system
def get_api_key() -> Optional[str]:
    """Get the OAuth access token for use as API key."""
    tokens = load_oauth_tokens()
    if tokens:
        return tokens["access_token"]
    return None


if __name__ == "__main__":
    # Test the connection
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        result = call_claude(prompt)
        if result:
            print(result)
        else:
            print("Failed to call Claude")
            sys.exit(1)
    else:
        # Just test auth
        tokens = load_oauth_tokens()
        if tokens:
            print(f"OAuth tokens loaded successfully")
            print(f"Token expires: {tokens['expires']}")
            
            # Quick test
            client = get_claude_client()
            if client:
                print("Claude client initialized")
                # Tiny test call
                try:
                    resp = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=10,
                        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}]
                    )
                    print(f"Test response: {resp.content[0].text}")
                except Exception as e:
                    print(f"Test call failed: {e}")
        else:
            print("No OAuth tokens found")
            sys.exit(1)
