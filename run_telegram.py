#!/usr/bin/env python3
"""
Run MAUDE Telegram bot standalone.

Usage:
    python run_telegram.py

Set TELEGRAM_BOT_TOKEN in .env or environment first.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure unbuffered output for background running
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

load_dotenv()

from channels import get_gateway, IncomingMessage, OutgoingMessage
from channels.telegram import create_telegram_channel


async def simple_maude_callback(msg: IncomingMessage) -> str:
    """Simple callback that echoes and shows it's working."""
    print(f">>> Message from {msg.username}: {msg.text}", flush=True)
    response = f"MAUDE received: {msg.text}\n\n(Full MAUDE integration coming soon!)"
    print(f">>> Responding: {response[:50]}...", flush=True)
    return response


async def main():
    print("MAUDE Telegram Bot", flush=True)
    print()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set!", flush=True)
        print("Add it to .env or run: export TELEGRAM_BOT_TOKEN=your_token", flush=True)
        sys.exit(1)

    gateway = get_gateway()
    gateway.set_maude_callback(simple_maude_callback)

    # Create and register Telegram channel
    telegram = create_telegram_channel(token)
    if not telegram:
        print("ERROR: Failed to create Telegram channel", flush=True)
        sys.exit(1)

    gateway.register(telegram)

    try:
        await telegram.connect()
    except Exception as e:
        print(f"ERROR: Failed to connect: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Check if already paired
    if gateway.authorized:
        print("Already paired users:", flush=True)
        for key, auth in gateway.authorized.items():
            print(f"  - {auth.username} ({auth.channel}:{auth.channel_id})", flush=True)
        print()
        print("Ready to receive messages!", flush=True)
    else:
        # Generate pairing code for new users
        code = gateway.generate_pairing_code()
        print()
        print(f"Pairing Code: {code}", flush=True)
        print()
        print("In Telegram, send this to your bot:", flush=True)
        print(f"  /pair {code}", flush=True)

    print()
    print("Press Ctrl+C to stop", flush=True)
    print()
    print("Waiting for messages...", flush=True)

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        await telegram.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
