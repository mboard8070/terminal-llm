#!/usr/bin/env python3
"""Simple Telegram bot test."""

import asyncio
import sys
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

TOKEN = "8305054637:AAFKPdZcaeWNylnOFthN-f-7nKXGbvQNU1o"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Echo the message."""
    print(f">>> Got message: {update.message.text}", flush=True)
    await update.message.reply_text(f"Echo: {update.message.text}")
    print(">>> Sent reply", flush=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start."""
    print(">>> Got /start command", flush=True)
    await update.message.reply_text("Hello! Send me a message.")

async def main():
    print("Starting bot...", flush=True)
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot running. Send a message to @maude_username_bot", flush=True)
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=False)

    print("Polling started. Waiting for messages...", flush=True)

    # Run forever until killed
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

    print("Stopping...", flush=True)
    await app.updater.stop()
    await app.stop()
    await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
