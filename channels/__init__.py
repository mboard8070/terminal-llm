"""
MAUDE Multi-Channel Messaging Gateway.

Connect MAUDE to messaging platforms for remote interaction.
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()


@dataclass
class IncomingMessage:
    """Unified message format from any channel."""
    channel: str           # "telegram", "discord", "cli"
    channel_id: str        # Chat/channel ID
    user_id: str           # Sender ID
    username: str          # Display name
    text: str              # Message content
    timestamp: str = ""    # ISO timestamp
    attachments: List[str] = field(default_factory=list)  # Image URLs/paths
    reply_to: Optional[str] = None  # Message being replied to
    raw: Any = None        # Original platform message


@dataclass
class OutgoingMessage:
    """Response to send back."""
    text: str
    channel: str = ""
    channel_id: str = ""
    attachments: List[str] = field(default_factory=list)
    reply_to: Optional[str] = None
    parse_mode: str = "markdown"


class Channel(ABC):
    """Base class for messaging channels."""

    name: str = "base"
    connected: bool = False

    @abstractmethod
    async def connect(self):
        """Connect to the channel."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the channel."""
        pass

    @abstractmethod
    async def send(self, channel_id: str, message: OutgoingMessage):
        """Send a message to a specific chat."""
        pass

    @abstractmethod
    def set_message_handler(self, callback: Callable):
        """Register message handler."""
        pass


@dataclass
class AuthorizedUser:
    """An authorized user/chat."""
    channel: str
    channel_id: str
    user_id: str
    username: str
    paired_at: str
    enabled: bool = True


class ChannelGateway:
    """Central hub managing all messaging channels."""

    CONFIG_FILE = Path.home() / ".config" / "maude" / "channels.json"
    AUTH_FILE = Path.home() / ".config" / "maude" / "authorized.json"

    def __init__(self):
        self.channels: Dict[str, Channel] = {}
        self.maude_callback: Optional[Callable] = None
        self.authorized: Dict[str, AuthorizedUser] = {}
        self.pending_pairs: Dict[str, dict] = {}  # pairing_code -> {channel, channel_id, ...}
        self.running = False
        self._load_authorized()

    def _load_authorized(self):
        """Load authorized users."""
        if self.AUTH_FILE.exists():
            try:
                data = json.loads(self.AUTH_FILE.read_text())
                for key, user_data in data.items():
                    self.authorized[key] = AuthorizedUser(**user_data)
            except:
                pass

    def _save_authorized(self):
        """Save authorized users."""
        self.AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self.authorized.items()}
        self.AUTH_FILE.write_text(json.dumps(data, indent=2))

    def set_maude_callback(self, callback: Callable):
        """Set the MAUDE processing callback."""
        self.maude_callback = callback

    def register(self, channel: Channel):
        """Register a channel."""
        self.channels[channel.name] = channel
        channel.set_message_handler(self._handle_message)
        console.print(f"[dim]Registered channel: {channel.name}[/dim]")

    def _make_auth_key(self, channel: str, channel_id: str) -> str:
        """Create authorization key."""
        return f"{channel}:{channel_id}"

    def is_authorized(self, channel: str, channel_id: str) -> bool:
        """Check if a chat is authorized."""
        key = self._make_auth_key(channel, channel_id)
        auth = self.authorized.get(key)
        return auth is not None and auth.enabled

    def generate_pairing_code(self) -> str:
        """Generate a new pairing code."""
        import random
        import string
        code = ''.join(random.choices(string.digits, k=6))
        self.pending_pairs[code] = {
            "created": datetime.now().isoformat(),
            "used": False
        }
        return code

    def complete_pairing(self, code: str, channel: str, channel_id: str,
                         user_id: str, username: str) -> bool:
        """Complete pairing with a code."""
        if code not in self.pending_pairs:
            return False

        pair_info = self.pending_pairs[code]
        if pair_info.get("used"):
            return False

        # Mark as used
        pair_info["used"] = True

        # Authorize the chat
        key = self._make_auth_key(channel, channel_id)
        self.authorized[key] = AuthorizedUser(
            channel=channel,
            channel_id=channel_id,
            user_id=user_id,
            username=username,
            paired_at=datetime.now().isoformat()
        )
        self._save_authorized()

        # Clean up old codes
        del self.pending_pairs[code]

        return True

    async def _handle_message(self, msg: IncomingMessage):
        """Route incoming message to MAUDE and send response."""
        # Check for pairing command
        if msg.text.startswith("/pair "):
            code = msg.text[6:].strip()
            if self.complete_pairing(code, msg.channel, msg.channel_id,
                                     msg.user_id, msg.username):
                await self.send(msg.channel, msg.channel_id,
                               OutgoingMessage(text="Paired successfully! You can now chat with MAUDE."))
            else:
                await self.send(msg.channel, msg.channel_id,
                               OutgoingMessage(text="Invalid or expired pairing code."))
            return

        # Check authorization
        if not self.is_authorized(msg.channel, msg.channel_id):
            await self.send(msg.channel, msg.channel_id,
                           OutgoingMessage(text="Not authorized. Use /pair <code> with a code from MAUDE CLI."))
            return

        # Process with MAUDE
        if self.maude_callback:
            try:
                response = await self.maude_callback(msg)
                if response:
                    await self.send(msg.channel, msg.channel_id,
                                   OutgoingMessage(text=response, reply_to=msg.reply_to))
            except Exception as e:
                console.print(f"[red]Error processing message: {e}[/red]")
                await self.send(msg.channel, msg.channel_id,
                               OutgoingMessage(text=f"Error: {e}"))

    async def send(self, channel: str, channel_id: str, message: OutgoingMessage):
        """Send a message through a specific channel."""
        if channel not in self.channels:
            console.print(f"[yellow]Channel '{channel}' not registered[/yellow]")
            return

        message.channel = channel
        message.channel_id = channel_id

        try:
            await self.channels[channel].send(channel_id, message)
        except Exception as e:
            console.print(f"[red]Error sending to {channel}: {e}[/red]")

    async def broadcast(self, message: OutgoingMessage):
        """Send a message to all authorized chats."""
        for key, auth in self.authorized.items():
            if auth.enabled:
                await self.send(auth.channel, auth.channel_id, message)

    async def start(self):
        """Start all channels."""
        self.running = True
        tasks = []
        for channel in self.channels.values():
            try:
                tasks.append(channel.connect())
            except Exception as e:
                console.print(f"[red]Error starting {channel.name}: {e}[/red]")

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        """Stop all channels."""
        self.running = False
        tasks = []
        for channel in self.channels.values():
            if channel.connected:
                tasks.append(channel.disconnect())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def list_channels(self) -> str:
        """List registered channels."""
        if not self.channels:
            return "No channels registered."

        lines = ["Messaging Channels:\n"]
        for name, channel in self.channels.items():
            status = "[green]✓[/green]" if channel.connected else "[red]✗[/red]"
            lines.append(f"  {status} {name}")

        if self.authorized:
            lines.append("\nAuthorized Chats:")
            for key, auth in self.authorized.items():
                status = "[green]✓[/green]" if auth.enabled else "[dim]disabled[/dim]"
                lines.append(f"  {status} {auth.channel}: {auth.username}")

        return "\n".join(lines)


# Global gateway instance
_gateway: Optional[ChannelGateway] = None


def get_gateway() -> ChannelGateway:
    """Get or create the global gateway."""
    global _gateway
    if _gateway is None:
        _gateway = ChannelGateway()
    return _gateway


def handle_channels_command(args: list) -> str:
    """Handle /channels command."""
    gateway = get_gateway()

    if not args:
        return gateway.list_channels()

    action = args[0].lower()

    if action == "pair":
        code = gateway.generate_pairing_code()
        return f"Pairing code: {code}\n\nSend '/pair {code}' to MAUDE in Telegram/Discord to authorize that chat."

    elif action == "unpair" and len(args) > 1:
        key = args[1]
        if key in gateway.authorized:
            del gateway.authorized[key]
            gateway._save_authorized()
            return f"Unpaired: {key}"
        return f"Not found: {key}"

    elif action == "list":
        return gateway.list_channels()

    return f"Unknown command: {action}\n\nUsage: /channels [pair|unpair <key>|list]"
