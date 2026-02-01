"""
Google Tools for MAUDE - Gmail and Google Drive integration.

Setup:
1. Create a project at https://console.cloud.google.com
2. Enable Gmail API and Google Drive API
3. Create OAuth 2.0 credentials (Desktop app)
4. Download credentials.json to ~/.config/maude/credentials.json
5. Run: python google_tools.py --auth (one-time setup)
"""

import os
import base64
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Google API imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# Configuration
CONFIG_DIR = Path.home() / ".config" / "maude"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
TOKEN_FILE = CONFIG_DIR / "google_token.json"

# Scopes for Gmail and Drive
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.file',
]


def get_credentials() -> Optional[Credentials]:
    """Get or refresh Google API credentials."""
    if not GOOGLE_AVAILABLE:
        return None

    creds = None

    # Load existing token
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif CREDENTIALS_FILE.exists():
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        else:
            return None

        # Save credentials
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())

    return creds


def check_google_setup() -> str:
    """Check if Google APIs are properly configured."""
    if not GOOGLE_AVAILABLE:
        return "Error: Google API libraries not installed. Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"

    if not CREDENTIALS_FILE.exists():
        return f"Error: credentials.json not found at {CREDENTIALS_FILE}. Download from Google Cloud Console."

    if not TOKEN_FILE.exists():
        return f"Error: Not authenticated. Run: python google_tools.py --auth"

    creds = get_credentials()
    if not creds:
        return "Error: Could not load credentials. Try re-authenticating."

    return "OK"


# ─────────────────────────────────────────────────────────────────────────────
# Gmail Functions
# ─────────────────────────────────────────────────────────────────────────────

def gmail_list_messages(query: str = "", max_results: int = 10) -> str:
    """List Gmail messages matching a query."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('gmail', 'v1', credentials=creds)

        # Search for messages
        results = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()

        messages = results.get('messages', [])

        if not messages:
            return f"No messages found for query: '{query}'" if query else "No messages found."

        # Get details for each message
        output = []
        for msg in messages:
            msg_data = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['From', 'Subject', 'Date']
            ).execute()

            headers = {h['name']: h['value'] for h in msg_data.get('payload', {}).get('headers', [])}
            snippet = msg_data.get('snippet', '')[:100]

            output.append(f"ID: {msg['id']}")
            output.append(f"  From: {headers.get('From', 'Unknown')}")
            output.append(f"  Subject: {headers.get('Subject', '(no subject)')}")
            output.append(f"  Date: {headers.get('Date', 'Unknown')}")
            output.append(f"  Preview: {snippet}...")
            output.append("")

        return f"Found {len(messages)} messages:\n\n" + "\n".join(output)

    except Exception as e:
        return f"Error listing emails: {e}"


def gmail_read_message(message_id: str) -> str:
    """Read a specific Gmail message by ID."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('gmail', 'v1', credentials=creds)

        msg = service.users().messages().get(
            userId='me',
            id=message_id,
            format='full'
        ).execute()

        headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}

        # Extract body
        body = ""
        payload = msg.get('payload', {})

        if 'body' in payload and payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        elif 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break

        output = [
            f"From: {headers.get('From', 'Unknown')}",
            f"To: {headers.get('To', 'Unknown')}",
            f"Subject: {headers.get('Subject', '(no subject)')}",
            f"Date: {headers.get('Date', 'Unknown')}",
            "",
            "Body:",
            body[:5000] if body else "(no text content)"
        ]

        return "\n".join(output)

    except Exception as e:
        return f"Error reading email: {e}"


def gmail_send_message(to: str, subject: str, body: str, cc: str = None) -> str:
    """Send an email via Gmail."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('gmail', 'v1', credentials=creds)

        # Get sender email
        profile = service.users().getProfile(userId='me').execute()
        sender = profile.get('emailAddress', '')

        # Create message
        message = MIMEMultipart()
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject
        if cc:
            message['cc'] = cc

        message.attach(MIMEText(body, 'plain'))

        # Encode and send
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        sent = service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()

        return f"Email sent successfully. Message ID: {sent['id']}"

    except Exception as e:
        return f"Error sending email: {e}"


def gmail_search(query: str) -> str:
    """Search Gmail with a query (same syntax as Gmail search box)."""
    return gmail_list_messages(query=query, max_results=10)


# ─────────────────────────────────────────────────────────────────────────────
# Google Drive Functions
# ─────────────────────────────────────────────────────────────────────────────

def drive_list_files(query: str = "", max_results: int = 20) -> str:
    """List files in Google Drive."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)

        # Build query
        q = query if query else None

        results = service.files().list(
            q=q,
            pageSize=max_results,
            fields="files(id, name, mimeType, size, modifiedTime, webViewLink)"
        ).execute()

        files = results.get('files', [])

        if not files:
            return "No files found."

        output = []
        for f in files:
            size = f.get('size', 'N/A')
            if size != 'N/A':
                size = f"{int(size) / 1024:.1f} KB"

            output.append(f"Name: {f['name']}")
            output.append(f"  ID: {f['id']}")
            output.append(f"  Type: {f['mimeType']}")
            output.append(f"  Size: {size}")
            output.append(f"  Modified: {f.get('modifiedTime', 'Unknown')}")
            if f.get('webViewLink'):
                output.append(f"  Link: {f['webViewLink']}")
            output.append("")

        return f"Found {len(files)} files:\n\n" + "\n".join(output)

    except Exception as e:
        return f"Error listing files: {e}"


def drive_search(query: str) -> str:
    """Search Google Drive for files by name or content."""
    # Convert simple query to Drive query syntax
    drive_query = f"name contains '{query}' or fullText contains '{query}'"
    return drive_list_files(query=drive_query, max_results=20)


def drive_read_file(file_id: str) -> str:
    """Read content of a text file from Google Drive."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)

        # Get file metadata
        file_meta = service.files().get(fileId=file_id, fields='name, mimeType').execute()
        name = file_meta.get('name', 'unknown')
        mime_type = file_meta.get('mimeType', '')

        # Handle Google Docs/Sheets/Slides
        export_types = {
            'application/vnd.google-apps.document': ('text/plain', 'txt'),
            'application/vnd.google-apps.spreadsheet': ('text/csv', 'csv'),
            'application/vnd.google-apps.presentation': ('text/plain', 'txt'),
        }

        if mime_type in export_types:
            export_mime, _ = export_types[mime_type]
            content = service.files().export(fileId=file_id, mimeType=export_mime).execute()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        else:
            # Regular file download
            content = service.files().get_media(fileId=file_id).execute()
            if isinstance(content, bytes):
                try:
                    content = content.decode('utf-8')
                except UnicodeDecodeError:
                    return f"File '{name}' is binary and cannot be displayed as text."

        # Truncate if too long
        if len(content) > 10000:
            content = content[:10000] + "\n\n[Content truncated - file is larger than 10KB]"

        return f"File: {name}\nType: {mime_type}\n\nContent:\n{content}"

    except Exception as e:
        return f"Error reading file: {e}"


def drive_upload_file(local_path: str, folder_id: str = None) -> str:
    """Upload a local file to Google Drive."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        path = Path(local_path).expanduser()
        if not path.exists():
            return f"Error: File not found: {local_path}"

        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)

        file_metadata = {'name': path.name}
        if folder_id:
            file_metadata['parents'] = [folder_id]

        # Guess mime type
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(path))

        media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()

        return f"Uploaded successfully!\nName: {file['name']}\nID: {file['id']}\nLink: {file.get('webViewLink', 'N/A')}"

    except Exception as e:
        return f"Error uploading file: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI for authentication
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--auth" in sys.argv:
        print("Google API Authentication Setup")
        print("=" * 40)

        if not GOOGLE_AVAILABLE:
            print("Error: Google API libraries not installed.")
            print("Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            sys.exit(1)

        if not CREDENTIALS_FILE.exists():
            print(f"Error: credentials.json not found at {CREDENTIALS_FILE}")
            print("\nTo set up:")
            print("1. Go to https://console.cloud.google.com")
            print("2. Create a project and enable Gmail API and Drive API")
            print("3. Create OAuth 2.0 credentials (Desktop app)")
            print("4. Download credentials.json")
            print(f"5. Save it to {CREDENTIALS_FILE}")
            sys.exit(1)

        print("Starting authentication flow...")
        print("(If no browser opens, copy the URL and open it manually)\n")

        # Manual auth flow for headless servers
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)

        # Try local server first, fall back to console/manual
        try:
            creds = flow.run_local_server(port=0)
        except Exception:
            # Headless server - use manual flow
            auth_url, _ = flow.authorization_url(prompt='consent')
            print(f"Open this URL in your browser:\n\n{auth_url}\n")
            code = input("Enter the authorization code: ").strip()
            flow.fetch_token(code=code)
            creds = flow.credentials

        # Save token
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())

        if creds:
            print(f"\nAuthentication successful!")
            print(f"Token saved to {TOKEN_FILE}")
        else:
            print("Authentication failed.")
            sys.exit(1)

    elif "--test" in sys.argv:
        print("Testing Google API connection...")
        status = check_google_setup()
        print(f"Status: {status}")

        if status == "OK":
            print("\nTesting Gmail...")
            print(gmail_list_messages(max_results=3))

            print("\nTesting Drive...")
            print(drive_list_files(max_results=3))

    else:
        print("Google Tools for MAUDE")
        print("Usage:")
        print("  python google_tools.py --auth   # Authenticate with Google")
        print("  python google_tools.py --test   # Test connection")
