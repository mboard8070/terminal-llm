"""
Telegram Bot Channel for MAUDE.

Enables interaction with MAUDE via Telegram.
Supports text, photos, and voice messages.
"""

import asyncio
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from datetime import datetime
from rich.console import Console

from channels import Channel, IncomingMessage, OutgoingMessage

console = Console()

# Voice server settings
VOICE_SERVER_URL = os.environ.get("VOICE_SERVER_URL", "wss://localhost:8998/ws")
VOICE_ENABLED = os.environ.get("VOICE_ENABLED", "true").lower() == "true"
VOICE_RESPONSE_ENABLED = os.environ.get("VOICE_RESPONSE_ENABLED", "true").lower() == "true"

def get_telegram_bot_token() -> str | None:
    """Return the MAUDE Telegram bot token, with legacy fallback."""
    return os.environ.get("MAUDE_TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")


# Try to import telegram library
try:
    from telegram import Bot, Update
    from telegram.constants import ChatAction, ParseMode
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    console.print("[yellow]python-telegram-bot not installed. Run: pip install python-telegram-bot[/yellow]")


class TelegramChannel(Channel):
    """Telegram bot channel."""

    name = "telegram"

    def __init__(self, token: str = None):
        self.token = token or get_telegram_bot_token()
        self.app: Application | None = None
        self.bot: Bot | None = None
        self._message_handler: Callable | None = None
        self.connected = False

        if not self.token:
            console.print("[yellow]MAUDE_TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN not set[/yellow]")

    async def connect(self):
        """Connect to Telegram."""
        if not TELEGRAM_AVAILABLE:
            raise RuntimeError("python-telegram-bot not installed")

        if not self.token:
            raise ValueError("MAUDE_TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN not set")

        try:
            self.app = Application.builder().token(self.token).build()
            self.bot = self.app.bot

            # Register handlers
            self.app.add_handler(CommandHandler("start", self._handle_start))
            self.app.add_handler(CommandHandler("pair", self._handle_pair))
            self.app.add_handler(CommandHandler("help", self._handle_help))
            self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
            self.app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
            self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice))
            # Inline-button approvals (e.g. weekly X-article publish).
            self.app.add_handler(CallbackQueryHandler(self._handle_callback_query))

            # Initialize and start
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=False)

            self.connected = True
            bot_info = await self.bot.get_me()
            console.print(f"[green]Telegram connected: @{bot_info.username}[/green]")

        except Exception as e:
            console.print(f"[red]Telegram connection failed: {e}[/red]")
            raise

    async def _handle_callback_query(self, update: "Update", context):
        """Handle inline-button taps for X approvals.

        callback_data is one of:
          xpub:/xskip:<token>     -> weekly article thread (publish_thread.py)
          xrep:/xrepskip:<token>  -> a single drafted reply (publish_reply.py)
        The actual posting runs in the x-growth-ops env (system python3, which
        has the X API deps) so the bot stays decoupled from that toolchain.
        """
        query = update.callback_query
        data = (query.data or "") if query else ""
        try:
            await query.answer()
        except Exception:
            pass

        xgrowth = "/home/mboard76/nvidia-workbench/terminal-llm/shared/x-growth-ops"

        async def _edit(text: str):
            for editor in (
                lambda: query.edit_message_caption(caption=text),
                lambda: query.edit_message_text(text=text),
            ):
                try:
                    await editor()
                    return
                except Exception:
                    continue

        # Route by callback prefix:
        #   xpub:/xskip:      -> weekly article thread (publish_thread.py)
        #   xrep:/xrepskip:   -> a single drafted reply  (publish_reply.py)
        if data.startswith(("xpub:", "xskip:")):
            publisher = f"{xgrowth}/publish_thread.py"
            skip = data.startswith("xskip:")
            working, skipped_msg, posted_label = (
                "Publishing thread to X...",
                "❌ Skipped this week's article.",
                "✅ Posted to X",
            )
        elif data.startswith(("xrep:", "xrepskip:")):
            publisher = f"{xgrowth}/publish_reply.py"
            skip = data.startswith("xrepskip:")
            working, skipped_msg, posted_label = "Posting reply to X...", "❌ Skipped this reply.", "✅ Reply posted"
        else:
            return

        token = data.split(":", 1)[1]
        cmd = ["/usr/bin/python3", publisher, "--token", token] + (["--skip"] if skip else [])

        if skip:
            await self._run_blocking(cmd, timeout=60)
            await _edit(skipped_msg)
            return

        await _edit(working)
        proc = await self._run_blocking(cmd, timeout=600)
        out = (proc.stdout or "").strip().splitlines() if proc else []
        last = out[-1] if out else ""
        if proc and proc.returncode == 0 and last.startswith("PUBLISHED"):
            await _edit(f"{posted_label}: {last.split(' ', 1)[1] if ' ' in last else ''}")
        elif last.startswith("ALREADY_PUBLISHED"):
            await _edit("Already posted.")
        else:
            tail = (proc.stderr or proc.stdout or "no output")[-200:] if proc else "error"
            await _edit(f"❌ Publish failed: {tail}")

    # Matches an X/Twitter status URL anywhere in a message.
    _X_STATUS_RE = re.compile(r"https?://(?:www\.)?(?:twitter\.com|x\.com)/\S*?status(?:es)?/\d+\S*")

    async def _maybe_draft_reply(self, update: Update) -> bool:
        """If the message contains an X status URL, draft a reply to it and send
        an Approve/Skip card. Returns True if handled (caller should stop)."""
        text = update.message.text or ""
        m = self._X_STATUS_RE.search(text)
        if not m:
            return False
        url = m.group(0)
        try:
            await update.message.reply_text("Drafting a reply to that tweet...")
        except Exception:
            pass
        xgrowth = "/home/mboard76/nvidia-workbench/terminal-llm/shared/x-growth-ops"
        cmd = ["/usr/bin/python3", f"{xgrowth}/reply_draft.py", "--url", url]
        proc = await self._run_blocking(cmd, timeout=180)
        out = (proc.stdout or "").strip().splitlines() if proc else []
        last = out[-1] if out else ""
        # reply_draft.py sends the Approve/Skip card itself on success; only
        # surface failures here (success card already arrived).
        if not (proc and proc.returncode == 0 and last.startswith(("DRAFTED", "ALREADY_REPLIED"))):
            tail = last or ((proc.stderr or "")[-200:] if proc else "no output")
            try:
                await update.message.reply_text(f"❌ Couldn't draft a reply: {tail}")
            except Exception:
                pass
        return True

    async def _run_blocking(self, cmd: list, timeout: int):
        """Run a blocking subprocess off the event loop."""

        def run():
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        try:
            return await asyncio.get_event_loop().run_in_executor(None, run)
        except Exception as e:
            console.print(f"[red]callback subprocess failed: {e}[/red]")
            return None

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
                chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
                for chunk in chunks:
                    await self.bot.send_message(
                        chat_id=int(channel_id),
                        text=chunk,
                        parse_mode=ParseMode.MARKDOWN if message.parse_mode == "markdown" else None,
                        reply_to_message_id=int(message.reply_to) if message.reply_to else None,
                    )
            else:
                # Try with markdown first, fall back to plain text if parsing fails
                try:
                    await self.bot.send_message(
                        chat_id=int(channel_id),
                        text=text,
                        parse_mode=ParseMode.MARKDOWN if message.parse_mode == "markdown" else None,
                        reply_to_message_id=int(message.reply_to) if message.reply_to else None,
                    )
                except Exception as e:
                    if "parse" in str(e).lower():
                        # Markdown parsing failed, send as plain text
                        await self.bot.send_message(
                            chat_id=int(channel_id),
                            text=text,
                            reply_to_message_id=int(message.reply_to) if message.reply_to else None,
                        )
                    else:
                        raise

            # Send attachments if any
            for attachment in message.attachments or []:
                if attachment.startswith("http"):
                    await self.bot.send_photo(chat_id=int(channel_id), photo=attachment)
                else:
                    with open(attachment, "rb") as f:
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
            timestamp=datetime.now().isoformat(),
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
        print(f">>> Telegram: received from {update.effective_user.first_name}: {update.message.text}", flush=True)

        # Paste-a-target reply flow: if the message contains an X/Twitter status
        # URL, draft a reply to it and send an Approve/Skip card instead of
        # routing the URL to the MAUDE chat agent.
        if await self._maybe_draft_reply(update):
            return

        if not self._message_handler:
            print(">>> Telegram: No message handler set!", flush=True)
            await update.message.reply_text("Bot is starting up, please try again...")
            return

        msg = IncomingMessage(
            channel="telegram",
            channel_id=str(update.effective_chat.id),
            user_id=str(update.effective_user.id),
            username=update.effective_user.username or update.effective_user.first_name or "Unknown",
            text=update.message.text,
            timestamp=datetime.now().isoformat(),
            reply_to=str(update.message.reply_to_message.message_id) if update.message.reply_to_message else None,
            raw=update,
        )

        print(">>> Telegram: calling handler...", flush=True)

        # Send immediate typing indicator
        chat_id = update.effective_chat.id
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except:
            pass

        # Keep typing indicator active until handler completes
        stop_typing = asyncio.Event()

        async def keep_typing():
            await asyncio.sleep(4)  # Wait before first refresh
            while not stop_typing.is_set():
                try:
                    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                except:
                    pass
                await asyncio.sleep(4)

        typing_task = asyncio.create_task(keep_typing())
        try:
            await self._message_handler(msg)
        finally:
            stop_typing.set()
            typing_task.cancel()

        print(">>> Telegram: handler done", flush=True)

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
            raw=update,
        )

        await self._message_handler(msg)

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming voice message - uses voice server for speech-to-speech."""
        print(f">>> Telegram: received voice from {update.effective_user.first_name}", flush=True)

        if not self._message_handler:
            await update.message.reply_text("Bot is starting up, please try again...")
            return

        try:
            # Get voice file
            voice = update.message.voice or update.message.audio
            file = await context.bot.get_file(voice.file_id)

            # Download the voice file
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_file:
                ogg_path = ogg_file.name
                await file.download_to_drive(ogg_path)

            # Convert OGG to WAV (16kHz mono for voice server/Whisper)
            wav_path = ogg_path.replace(".ogg", ".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path], capture_output=True, check=True
            )

            # Try voice server first for full speech-to-speech
            if VOICE_ENABLED:
                response_audio = await self._process_with_voice_server(wav_path)
                if response_audio:
                    # Convert response to OGG for Telegram
                    response_ogg = wav_path.replace(".wav", "_response.ogg")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", response_audio, "-c:a", "libopus", response_ogg],
                        capture_output=True,
                        check=True,
                    )

                    # Send voice response
                    await self.send_voice(str(update.effective_chat.id), response_ogg)

                    # Clean up
                    os.unlink(ogg_path)
                    os.unlink(wav_path)
                    os.unlink(response_audio)
                    os.unlink(response_ogg)
                    print(">>> Telegram: voice server response sent", flush=True)
                    return

            # Fallback: Transcribe and process as text
            transcribed_text = await self._transcribe_audio(wav_path)

            # Clean up input files
            os.unlink(ogg_path)

            if not transcribed_text:
                await update.message.reply_text("Sorry, I couldn't understand the audio.")
                os.unlink(wav_path)
                return

            print(f">>> Telegram: transcribed: {transcribed_text}", flush=True)

            # Create message with transcribed text
            msg = IncomingMessage(
                channel="telegram",
                channel_id=str(update.effective_chat.id),
                user_id=str(update.effective_user.id),
                username=update.effective_user.username or update.effective_user.first_name or "Unknown",
                text=transcribed_text,
                timestamp=datetime.now().isoformat(),
                metadata={"voice": True, "duration": voice.duration, "respond_with_voice": VOICE_RESPONSE_ENABLED},
                raw=update,
            )

            print(">>> Telegram: calling handler for voice...", flush=True)
            await self._message_handler(msg)
            print(">>> Telegram: voice handler done", flush=True)
            os.unlink(wav_path)

        except subprocess.CalledProcessError as e:
            console.print(f"[red]FFmpeg error: {e}[/red]")
            await update.message.reply_text("Error processing audio. Is ffmpeg installed?")
        except Exception as e:
            console.print(f"[red]Voice handling error: {e}[/red]")
            import traceback

            traceback.print_exc()
            await update.message.reply_text(f"Error processing voice message: {str(e)[:100]}")

    async def _process_with_voice_server(self, wav_path: str) -> str | None:
        """Process audio through voice server for speech-to-speech response."""
        try:
            import json
            import ssl

            import websockets

            # Voice server uses self-signed SSL
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            async with websockets.connect(VOICE_SERVER_URL, ssl=ssl_context) as ws:
                # Send configuration
                config_msg = {
                    "type": "config",
                    "voice": "NATF2",
                    "role": "You are MAUDE, a helpful AI assistant. Keep responses concise for voice.",
                }
                await ws.send(json.dumps(config_msg))

                # Read and send audio
                with open(wav_path, "rb") as f:
                    audio_data = f.read()

                await ws.send(audio_data)

                # Signal end of input
                await ws.send(json.dumps({"type": "end_of_input"}))

                # Collect response audio
                response_chunks = []
                transcript = None

                while True:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        if isinstance(response, bytes):
                            response_chunks.append(response)
                        else:
                            data = json.loads(response)
                            if data.get("type") == "transcript":
                                transcript = data.get("text")
                                print(f">>> Voice server transcript: {transcript}", flush=True)
                            elif data.get("type") == "end_of_response":
                                break
                    except TimeoutError:
                        break

                if response_chunks:
                    # Write response audio to file
                    response_path = wav_path.replace(".wav", "_pp_response.wav")
                    with open(response_path, "wb") as f:
                        for chunk in response_chunks:
                            f.write(chunk)
                    return response_path

        except ImportError:
            console.print("[yellow]websockets not installed for voice server[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Voice server error: {e}, falling back to text[/yellow]")

        return None

    async def _transcribe_audio(self, wav_path: str) -> str | None:
        """Transcribe audio file using Whisper."""
        try:
            # Try faster-whisper first (local)
            from faster_whisper import WhisperModel

            model = WhisperModel("base", device="cuda", compute_type="float16")
            segments, _ = model.transcribe(wav_path)
            return " ".join(segment.text for segment in segments).strip()
        except ImportError:
            pass
        except Exception as e:
            console.print(f"[yellow]faster-whisper error: {e}, trying whisper...[/yellow]")

        try:
            # Try original whisper
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(wav_path)
            return result["text"].strip()
        except ImportError:
            pass
        except Exception as e:
            console.print(f"[yellow]whisper error: {e}[/yellow]")

        # Fall back to OpenAI Whisper API if available
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key)
                with open(wav_path, "rb") as f:
                    transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
                return transcript.text
            except Exception as e:
                console.print(f"[yellow]OpenAI Whisper error: {e}[/yellow]")

        console.print("[red]No Whisper backend available[/red]")
        return None

    async def send_voice(self, channel_id: str, audio_path: str, caption: str = None):
        """Send a voice message to a Telegram chat."""
        if not self.bot:
            return

        try:
            with open(audio_path, "rb") as audio_file:
                await self.bot.send_voice(chat_id=int(channel_id), voice=audio_file, caption=caption)
        except Exception as e:
            console.print(f"[red]Telegram send_voice error: {e}[/red]")


def create_telegram_channel(token: str = None) -> TelegramChannel | None:
    """Create a Telegram channel if available."""
    if not TELEGRAM_AVAILABLE:
        return None

    token = token or get_telegram_bot_token()
    if not token:
        return None

    return TelegramChannel(token)
