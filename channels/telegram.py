"""
Telegram Bot Channel for MAUDE.

Enables interaction with MAUDE via Telegram.
"""

import os
import asyncio
from typing import Callable, Optional
from datetime import datetime
from rich.console import Console

from channels import Channel, IncomingMessage, OutgoingMessage

console = Console()

# Try to import telegram library
try:
    from telegram import Update, Bot
    from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    console.print("[yellow]python-telegram-bot not installed. Run: pip install python-telegram-bot[/yellow]")


class TelegramChannel(Channel):
    """Telegram bot channel."""

    name = "telegram"

    def __init__(self, token: str = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.app: Optional[Application] = None
        self.bot: Optional[Bot] = None
        self._message_handler: Optional[Callable] = None
        self.connected = False

        if not self.token:
            console.print("[yellow]TELEGRAM_BOT_TOKEN not set[/yellow]")

    async def connect(self):
        """Connect to Telegram."""
        if not TELEGRAM_AVAILABLE:
            raise RuntimeError("python-telegram-bot not installed")

        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")

        try:
            self.app = Application.builder().token(self.token).build()
            self.bot = self.app.bot

            # Register handlers
            self.app.add_handler(CommandHandler("start", self._handle_start))
            self.app.add_handler(CommandHandler("pair", self._handle_pair))
            self.app.add_handler(CommandHandler("help", self._handle_help))
            self.app.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._handle_message
            ))
            self.app.add_handler(MessageHandler(
                filters.PHOTO,
                self._handle_photo
            ))

            # Initialize and start
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=True)

            self.connected = True
            bot_info = await self.bot.get_me()
            console.print(f"[green]Telegram connected: @{bot_info.username}[/green]")

        except Exception as e:
            console.print(f"[red]Telegram connection failed: {e}[/red]")
            raise

    async def disconnect(self):
        """Disconnect from Telegram."""
        if self.app:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
            except:
                pass
        self.connected = False
        console.print("[dim]Telegram disconnected[/dim]")

    async def send(self, channel_id: str, message: OutgoingMessage):
        """Send a message to a Telegram chat."""
        if not self.bot:
            return

        try:
            # Telegram has a 4096 character limit
            text = message.text
            if len(text) > 4000:
                # Split into chunks
                chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                for chunk in chunks:
                    await self.bot.send_message(
                        chat_id=int(channel_id),
                        text=chunk,
                        parse_mode=ParseMode.MARKDOWN if message.parse_mode == "markdown" else None,
                        reply_to_message_id=int(message.reply_to) if message.reply_to else None
                    )
            else:
                await self.bot.send_message(
                    chat_id=int(channel_id),
                    text=text,
                    parse_mode=ParseMode.MARKDOWN if message.parse_mode == "markdown" else None,
                    reply_to_message_id=int(message.reply_to) if message.reply_to else None
                )

            # Send attachments if any
            for attachment in message.attachments or []:
                if attachment.startswith("http"):
                    await self.bot.send_photo(chat_id=int(channel_id), photo=attachment)
                else:
                    with open(attachment, 'rb') as f:
                        await self.bot.send_photo(chat_id=int(channel_id), photo=f)

        except Exception as e:
            console.print(f"[red]Telegram send error: {e}[/red]")

    def set_message_handler(self, callback: Callable):
        """Register message handler."""
        self._message_handler = callback

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        await update.message.reply_text(
            "Welcome to MAUDE!\n\n"
            "To authorize this chat, get a pairing code from MAUDE CLI:\n"
            "  /channels pair\n\n"
            "Then send it here:\n"
            "  /pair <code>\n\n"
            "Once paired, you can chat directly with MAUDE."
        )

    async def _handle_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pair command."""
        if not context.args:
            await update.message.reply_text("Usage: /pair <code>")
            return

        code = context.args[0]

        # Create incoming message for pairing
        msg = IncomingMessage(
            channel="telegram",
            channel_id=str(update.effective_chat.id),
            user_id=str(update.effective_user.id),
            username=update.effective_user.username or update.effective_user.first_name or "Unknown",
            text=f"/pair {code}",
            timestamp=datetime.now().isoformat()
        )

        if self._message_handler:
            await self._message_handler(msg)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await update.message.reply_text(
            "MAUDE Commands:\n\n"
            "/start - Welcome message\n"
            "/pair <code> - Authorize this chat\n"
            "/help - Show this help\n\n"
            "Just send a message to chat with MAUDE!"
        )

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text message."""
        if not self._message_handler:
            return

        msg = IncomingMessage(
            channel="telegram",
            channel_id=str(update.effective_chat.id),
            user_id=str(update.effective_user.id),
            username=update.effective_user.username or update.effective_user.first_name or "Unknown",
            text=update.message.text,
            timestamp=datetime.now().isoformat(),
            reply_to=str(update.message.reply_to_message.message_id) if update.message.reply_to_message else None,
            raw=update
        )

        await self._message_handler(msg)

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming photo."""
        if not self._message_handler:
            return

        # Get the largest photo
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        msg = IncomingMessage(
            channel="telegram",
            channel_id=str(update.effective_chat.id),
            user_id=str(update.effective_user.id),
            username=update.effective_user.username or update.effective_user.first_name or "Unknown",
            text=update.message.caption or "What's in this image?",
            timestamp=datetime.now().isoformat(),
            attachments=[file.file_path],  # Telegram file URL
            raw=update
        )

        await self._message_handler(msg)


def create_telegram_channel(token: str = None) -> Optional[TelegramChannel]:
    """Create a Telegram channel if available."""
    if not TELEGRAM_AVAILABLE:
        return None

    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return None

    return TelegramChannel(token)
