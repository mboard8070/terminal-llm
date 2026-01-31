"""
Telegram Bot Channel for MAUDE.

Enables interaction with MAUDE via Telegram.
Supports text, photos, and voice messages.
"""

import os
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime
from rich.console import Console

from channels import Channel, IncomingMessage, OutgoingMessage

console = Console()

# Voice/PersonaPlex settings
PERSONAPLEX_URL = os.environ.get("PERSONAPLEX_URL", "wss://localhost:8998/ws")
PERSONAPLEX_ENABLED = os.environ.get("PERSONAPLEX_ENABLED", "true").lower() == "true"
VOICE_RESPONSE_ENABLED = os.environ.get("VOICE_RESPONSE_ENABLED", "true").lower() == "true"

# Try to import telegram library
try:
    from telegram import Update, Bot
    from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
    from telegram.constants import ParseMode, ChatAction
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
            self.app.add_handler(MessageHandler(
                filters.VOICE | filters.AUDIO,
                self._handle_voice
            ))

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
                # Try with markdown first, fall back to plain text if parsing fails
                try:
                    await self.bot.send_message(
                        chat_id=int(channel_id),
                        text=text,
                        parse_mode=ParseMode.MARKDOWN if message.parse_mode == "markdown" else None,
                        reply_to_message_id=int(message.reply_to) if message.reply_to else None
                    )
                except Exception as e:
                    if "parse" in str(e).lower():
                        # Markdown parsing failed, send as plain text
                        await self.bot.send_message(
                            chat_id=int(channel_id),
                            text=text,
                            reply_to_message_id=int(message.reply_to) if message.reply_to else None
                        )
                    else:
                        raise

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
        print(f">>> Telegram: received from {update.effective_user.first_name}: {update.message.text}", flush=True)

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
            raw=update
        )

        print(f">>> Telegram: calling handler...", flush=True)

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

        print(f">>> Telegram: handler done", flush=True)

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

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming voice message - uses PersonaPlex for speech-to-speech."""
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

            # Convert OGG to WAV (16kHz mono for PersonaPlex/Whisper)
            wav_path = ogg_path.replace(".ogg", ".wav")
            subprocess.run([
                "ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path
            ], capture_output=True, check=True)

            # Try PersonaPlex first for full speech-to-speech
            if PERSONAPLEX_ENABLED:
                response_audio = await self._process_with_personaplex(wav_path)
                if response_audio:
                    # Convert response to OGG for Telegram
                    response_ogg = wav_path.replace(".wav", "_response.ogg")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", response_audio, "-c:a", "libopus", response_ogg
                    ], capture_output=True, check=True)

                    # Send voice response
                    await self.send_voice(str(update.effective_chat.id), response_ogg)

                    # Clean up
                    os.unlink(ogg_path)
                    os.unlink(wav_path)
                    os.unlink(response_audio)
                    os.unlink(response_ogg)
                    print(f">>> Telegram: PersonaPlex voice response sent", flush=True)
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
                raw=update
            )

            print(f">>> Telegram: calling handler for voice...", flush=True)
            await self._message_handler(msg)
            print(f">>> Telegram: voice handler done", flush=True)
            os.unlink(wav_path)

        except subprocess.CalledProcessError as e:
            console.print(f"[red]FFmpeg error: {e}[/red]")
            await update.message.reply_text("Error processing audio. Is ffmpeg installed?")
        except Exception as e:
            console.print(f"[red]Voice handling error: {e}[/red]")
            import traceback
            traceback.print_exc()
            await update.message.reply_text(f"Error processing voice message: {str(e)[:100]}")

    async def _process_with_personaplex(self, wav_path: str) -> Optional[str]:
        """Process audio through PersonaPlex for speech-to-speech response."""
        try:
            import websockets
            import ssl
            import json

            # PersonaPlex uses self-signed SSL
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            async with websockets.connect(PERSONAPLEX_URL, ssl=ssl_context) as ws:
                # Send configuration
                config_msg = {
                    "type": "config",
                    "voice": "NATF2",
                    "role": "You are MAUDE, a helpful AI assistant. Keep responses concise for voice."
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
                                print(f">>> PersonaPlex transcript: {transcript}", flush=True)
                            elif data.get("type") == "end_of_response":
                                break
                    except asyncio.TimeoutError:
                        break

                if response_chunks:
                    # Write response audio to file
                    response_path = wav_path.replace(".wav", "_pp_response.wav")
                    with open(response_path, "wb") as f:
                        for chunk in response_chunks:
                            f.write(chunk)
                    return response_path

        except ImportError:
            console.print("[yellow]websockets not installed for PersonaPlex[/yellow]")
        except Exception as e:
            console.print(f"[yellow]PersonaPlex error: {e}, falling back to text[/yellow]")

        return None

    async def _transcribe_audio(self, wav_path: str) -> Optional[str]:
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
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f
                    )
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
            with open(audio_path, 'rb') as audio_file:
                await self.bot.send_voice(
                    chat_id=int(channel_id),
                    voice=audio_file,
                    caption=caption
                )
        except Exception as e:
            console.print(f"[red]Telegram send_voice error: {e}[/red]")


def create_telegram_channel(token: str = None) -> Optional[TelegramChannel]:
    """Create a Telegram channel if available."""
    if not TELEGRAM_AVAILABLE:
        return None

    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return None

    return TelegramChannel(token)
