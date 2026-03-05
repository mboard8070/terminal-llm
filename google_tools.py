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
import datetime
import uuid
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

# Scopes for Gmail, Drive, Sheets, Slides, and Calendar
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/presentations',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/contacts',
    'https://www.googleapis.com/auth/youtube',
    'https://www.googleapis.com/auth/youtube.force-ssl',
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


def drive_create_folder(name: str, parent_id: str = None) -> str:
    """Create a folder in Google Drive."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)

        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]

        folder = service.files().create(
            body=file_metadata,
            fields='id, name, webViewLink'
        ).execute()

        return f"Folder created successfully!\nName: {folder['name']}\nID: {folder['id']}\nLink: {folder.get('webViewLink', 'N/A')}"

    except Exception as e:
        return f"Error creating folder: {e}"


def drive_create_doc(name: str, folder_id: str = None, folder_name: str = None, content: str = "") -> str:
    """Create a Google Doc in Google Drive."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)

        # Resolve folder_name to folder_id if provided
        if folder_name and not folder_id:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = service.files().list(q=query, fields='files(id, name)', pageSize=5).execute()
            folders = results.get('files', [])
            if folders:
                folder_id = folders[0]['id']
            else:
                # Create the folder
                folder_meta = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
                folder = service.files().create(body=folder_meta, fields='id').execute()
                folder_id = folder['id']

        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.document'
        }
        if folder_id:
            file_metadata['parents'] = [folder_id]

        doc = service.files().create(
            body=file_metadata,
            fields='id, name, webViewLink'
        ).execute()

        result = f"Google Doc created successfully!\nName: {doc['name']}\nID: {doc['id']}\nLink: {doc.get('webViewLink', 'N/A')}"

        # If content provided, try to add it using the Docs API
        if content:
            try:
                docs_service = build('docs', 'v1', credentials=creds)
                docs_service.documents().batchUpdate(
                    documentId=doc['id'],
                    body={
                        'requests': [{
                            'insertText': {
                                'location': {'index': 1},
                                'text': content
                            }
                        }]
                    }
                ).execute()
                result += "\nContent added successfully."
            except Exception as e:
                result += f"\nNote: Doc created but could not add content (enable Google Docs API for this feature)."

        return result

    except Exception as e:
        return f"Error creating Google Doc: {e}"


def drive_create_sheet(name: str, folder_id: str = None, folder_name: str = None) -> str:
    """Create a Google Sheet in Google Drive."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)

        # Resolve folder_name to folder_id if provided
        if folder_name and not folder_id:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = service.files().list(q=query, fields='files(id, name)', pageSize=5).execute()
            folders = results.get('files', [])
            if folders:
                folder_id = folders[0]['id']
            else:
                folder_meta = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
                folder = service.files().create(body=folder_meta, fields='id').execute()
                folder_id = folder['id']

        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.spreadsheet'
        }
        if folder_id:
            file_metadata['parents'] = [folder_id]

        sheet = service.files().create(
            body=file_metadata,
            fields='id, name, webViewLink'
        ).execute()

        return f"Google Sheet created successfully!\nName: {sheet['name']}\nID: {sheet['id']}\nLink: {sheet.get('webViewLink', 'N/A')}"

    except Exception as e:
        return f"Error creating Google Sheet: {e}"


def drive_update_doc(doc_id: str, content: str, append: bool = False) -> str:
    """Update content in an existing Google Doc."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        docs_service = build('docs', 'v1', credentials=creds)

        # Get current document to find end index if appending
        doc = docs_service.documents().get(documentId=doc_id).execute()

        requests = []
        if append:
            # Find the end of the document
            end_index = doc.get('body', {}).get('content', [{}])[-1].get('endIndex', 1) - 1
            if end_index < 1:
                end_index = 1
            requests.append({
                'insertText': {
                    'location': {'index': end_index},
                    'text': content
                }
            })
        else:
            # Replace all content - first delete existing, then insert new
            end_index = doc.get('body', {}).get('content', [{}])[-1].get('endIndex', 1) - 1
            if end_index > 1:
                requests.append({
                    'deleteContentRange': {
                        'range': {
                            'startIndex': 1,
                            'endIndex': end_index
                        }
                    }
                })
            requests.append({
                'insertText': {
                    'location': {'index': 1},
                    'text': content
                }
            })

        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={'requests': requests}
        ).execute()

        # Get doc info for response
        drive_service = build('drive', 'v3', credentials=creds)
        file_meta = drive_service.files().get(fileId=doc_id, fields='name, webViewLink').execute()

        action = "appended to" if append else "updated"
        return f"Content {action} successfully!\nDocument: {file_meta.get('name')}\nLink: {file_meta.get('webViewLink', 'N/A')}"

    except Exception as e:
        if "SERVICE_DISABLED" in str(e):
            return "Error: Google Docs API is not enabled. Enable it at: https://console.developers.google.com/apis/api/docs.googleapis.com"
        return f"Error updating document: {e}"


def drive_delete_file(file_id: str) -> str:
    """Delete a file or folder from Google Drive."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)

        # Get file name first for confirmation message
        file_meta = service.files().get(fileId=file_id, fields='name').execute()
        name = file_meta.get('name', 'Unknown')

        service.files().delete(fileId=file_id).execute()

        return f"Successfully deleted: {name} (ID: {file_id})"

    except Exception as e:
        return f"Error deleting file: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Google Contacts Functions
# ─────────────────────────────────────────────────────────────────────────────

def contacts_list(max_results: int = 20, query: str = None) -> str:
    """List contacts, optionally filtered by a search query."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('people', 'v1', credentials=creds)

        if query:
            results = service.people().searchContacts(
                query=query,
                readMask='names,emailAddresses,phoneNumbers',
                pageSize=max_results
            ).execute()
            contacts = [r['person'] for r in results.get('results', []) if 'person' in r]
        else:
            results = service.people().connections().list(
                resourceName='people/me',
                pageSize=max_results,
                personFields='names,emailAddresses,phoneNumbers'
            ).execute()
            contacts = results.get('connections', [])

        if not contacts:
            return f"No contacts found." if not query else f"No contacts found for query: '{query}'"

        output = []
        for person in contacts:
            names = person.get('names', [])
            name = names[0].get('displayName', 'Unknown') if names else 'Unknown'

            emails = person.get('emailAddresses', [])
            email_list = ', '.join(e.get('value', '') for e in emails) if emails else 'None'

            phones = person.get('phoneNumbers', [])
            phone_list = ', '.join(p.get('value', '') for p in phones) if phones else 'None'

            resource_name = person.get('resourceName', 'Unknown')

            output.append(f"Name: {name}")
            output.append(f"  Emails: {email_list}")
            output.append(f"  Phones: {phone_list}")
            output.append(f"  Resource: {resource_name}")
            output.append("")

        return f"Found {len(contacts)} contacts:\n\n" + "\n".join(output)

    except Exception as e:
        return f"Error listing contacts: {e}"


def contacts_get(resource_name: str) -> str:
    """Get detailed info for a single contact by resource name."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('people', 'v1', credentials=creds)

        person = service.people().get(
            resourceName=resource_name,
            personFields='names,emailAddresses,phoneNumbers,addresses,organizations,birthdays,biographies,urls'
        ).execute()

        output = [f"Contact Details ({resource_name})", "=" * 40]

        # Names
        names = person.get('names', [])
        if names:
            name = names[0]
            output.append(f"Name: {name.get('displayName', 'Unknown')}")
            if name.get('givenName'):
                output.append(f"  Given Name: {name['givenName']}")
            if name.get('familyName'):
                output.append(f"  Family Name: {name['familyName']}")

        # Emails
        emails = person.get('emailAddresses', [])
        if emails:
            output.append("Emails:")
            for email in emails:
                label = email.get('type', 'other')
                output.append(f"  [{label}] {email.get('value', '')}")

        # Phone numbers
        phones = person.get('phoneNumbers', [])
        if phones:
            output.append("Phone Numbers:")
            for phone in phones:
                label = phone.get('type', 'other')
                output.append(f"  [{label}] {phone.get('value', '')}")

        # Addresses
        addresses = person.get('addresses', [])
        if addresses:
            output.append("Addresses:")
            for addr in addresses:
                label = addr.get('type', 'other')
                formatted = addr.get('formattedValue', '')
                output.append(f"  [{label}] {formatted}")

        # Organizations
        orgs = person.get('organizations', [])
        if orgs:
            output.append("Organizations:")
            for org in orgs:
                org_name = org.get('name', '')
                title = org.get('title', '')
                if org_name and title:
                    output.append(f"  {org_name} - {title}")
                elif org_name:
                    output.append(f"  {org_name}")
                elif title:
                    output.append(f"  {title}")

        # Birthdays
        birthdays = person.get('birthdays', [])
        if birthdays:
            bday = birthdays[0].get('date', {})
            month = bday.get('month', '')
            day = bday.get('day', '')
            year = bday.get('year', '')
            if year:
                output.append(f"Birthday: {year}-{month:02d}-{day:02d}" if isinstance(month, int) else f"Birthday: {year}-{month}-{day}")
            elif month and day:
                output.append(f"Birthday: {month}/{day}")

        # Biographies
        bios = person.get('biographies', [])
        if bios:
            output.append(f"Notes: {bios[0].get('value', '')}")

        # URLs
        urls = person.get('urls', [])
        if urls:
            output.append("URLs:")
            for url in urls:
                label = url.get('type', 'other')
                output.append(f"  [{label}] {url.get('value', '')}")

        return "\n".join(output)

    except Exception as e:
        return f"Error getting contact: {e}"


def contacts_create(given_name: str, family_name: str = "", email: str = None, phone: str = None, organization: str = None) -> str:
    """Create a new contact."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('people', 'v1', credentials=creds)

        person = {
            'names': [{'givenName': given_name, 'familyName': family_name}]
        }

        if email:
            person['emailAddresses'] = [{'value': email}]

        if phone:
            person['phoneNumbers'] = [{'value': phone}]

        if organization:
            person['organizations'] = [{'name': organization}]

        result = service.people().createContact(body=person).execute()
        resource_name = result.get('resourceName', 'Unknown')

        return f"Contact created successfully!\nName: {given_name} {family_name}\nResource: {resource_name}"

    except Exception as e:
        return f"Error creating contact: {e}"


def contacts_update(resource_name: str, given_name: str = None, family_name: str = None, email: str = None, phone: str = None) -> str:
    """Update an existing contact."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('people', 'v1', credentials=creds)

        # Get current contact to retrieve etag
        current = service.people().get(
            resourceName=resource_name,
            personFields='names,emailAddresses,phoneNumbers'
        ).execute()

        etag = current.get('etag')

        # Build update body
        person = {'etag': etag}

        # Names
        if given_name is not None or family_name is not None:
            current_names = current.get('names', [{}])
            name = current_names[0] if current_names else {}
            updated_name = {}
            updated_name['givenName'] = given_name if given_name is not None else name.get('givenName', '')
            updated_name['familyName'] = family_name if family_name is not None else name.get('familyName', '')
            person['names'] = [updated_name]
        else:
            person['names'] = current.get('names', [])

        # Emails
        if email is not None:
            person['emailAddresses'] = [{'value': email}]
        else:
            person['emailAddresses'] = current.get('emailAddresses', [])

        # Phone numbers
        if phone is not None:
            person['phoneNumbers'] = [{'value': phone}]
        else:
            person['phoneNumbers'] = current.get('phoneNumbers', [])

        result = service.people().updateContact(
            resourceName=resource_name,
            body=person,
            updatePersonFields='names,emailAddresses,phoneNumbers'
        ).execute()

        updated_names = result.get('names', [])
        display_name = updated_names[0].get('displayName', 'Unknown') if updated_names else 'Unknown'

        return f"Contact updated successfully!\nName: {display_name}\nResource: {resource_name}"

    except Exception as e:
        return f"Error updating contact: {e}"


def contacts_delete(resource_name: str) -> str:
    """Delete a contact by resource name."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('people', 'v1', credentials=creds)

        service.people().deleteContact(resourceName=resource_name).execute()

        return f"Contact deleted successfully: {resource_name}"

    except Exception as e:
        return f"Error deleting contact: {e}"


def contacts_search(query: str) -> str:
    """Search contacts by name, email, or phone number."""
    return contacts_list(query=query)


# ─────────────────────────────────────────────────────────────────────────────
# Google Calendar Functions
# ─────────────────────────────────────────────────────────────────────────────

def calendar_list_events(max_results: int = 10, time_min: str = None, time_max: str = None, calendar_id: str = "primary") -> str:
    """List upcoming events from Google Calendar."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('calendar', 'v3', credentials=creds)

        # Default time_min to now if not provided
        if not time_min:
            time_min = datetime.datetime.utcnow().isoformat() + 'Z'

        kwargs = {
            'calendarId': calendar_id,
            'timeMin': time_min,
            'maxResults': max_results,
            'singleEvents': True,
            'orderBy': 'startTime',
        }
        if time_max:
            kwargs['timeMax'] = time_max

        results = service.events().list(**kwargs).execute()
        events = results.get('items', [])

        if not events:
            return "No upcoming events found."

        output = []
        for event in events:
            start = event.get('start', {}).get('dateTime', event.get('start', {}).get('date', 'Unknown'))
            end = event.get('end', {}).get('dateTime', event.get('end', {}).get('date', 'Unknown'))
            summary = event.get('summary', '(No title)')
            location = event.get('location', '')
            description = event.get('description', '')

            output.append(f"Event: {summary}")
            output.append(f"  ID: {event.get('id', 'Unknown')}")
            output.append(f"  Start: {start}")
            output.append(f"  End: {end}")
            if location:
                output.append(f"  Location: {location}")
            if description:
                truncated = description[:200] + "..." if len(description) > 200 else description
                output.append(f"  Description: {truncated}")
            output.append("")

        return f"Found {len(events)} events:\n\n" + "\n".join(output)

    except Exception as e:
        return f"Error listing calendar events: {e}"


def calendar_create_event(summary: str, start: str, end: str, description: str = None, location: str = None, calendar_id: str = "primary") -> str:
    """Create an event in Google Calendar."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('calendar', 'v3', credentials=creds)

        event_body = {
            'summary': summary,
            'start': {'dateTime': start},
            'end': {'dateTime': end},
        }
        if description:
            event_body['description'] = description
        if location:
            event_body['location'] = location

        event = service.events().insert(calendarId=calendar_id, body=event_body).execute()

        return f"Event created successfully!\nSummary: {event.get('summary')}\nID: {event.get('id')}\nLink: {event.get('htmlLink', 'N/A')}"

    except Exception as e:
        return f"Error creating calendar event: {e}"


def calendar_update_event(event_id: str, summary: str = None, start: str = None, end: str = None, description: str = None, location: str = None, calendar_id: str = "primary") -> str:
    """Update an existing event in Google Calendar."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('calendar', 'v3', credentials=creds)

        # Get the existing event
        event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()

        # Update only the fields that are provided
        if summary is not None:
            event['summary'] = summary
        if start is not None:
            event['start'] = {'dateTime': start}
        if end is not None:
            event['end'] = {'dateTime': end}
        if description is not None:
            event['description'] = description
        if location is not None:
            event['location'] = location

        updated_event = service.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event
        ).execute()

        return f"Event updated successfully!\nSummary: {updated_event.get('summary')}\nID: {updated_event.get('id')}\nLink: {updated_event.get('htmlLink', 'N/A')}"

    except Exception as e:
        return f"Error updating calendar event: {e}"


def calendar_delete_event(event_id: str, calendar_id: str = "primary") -> str:
    """Delete an event from Google Calendar."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('calendar', 'v3', credentials=creds)

        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()

        return f"Event deleted successfully. (ID: {event_id})"

    except Exception as e:
        return f"Error deleting calendar event: {e}"


def calendar_search_events(query: str, max_results: int = 10, calendar_id: str = "primary") -> str:
    """Search for events by text in Google Calendar."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('calendar', 'v3', credentials=creds)

        time_min = datetime.datetime.utcnow().isoformat() + 'Z'

        results = service.events().list(
            calendarId=calendar_id,
            q=query,
            timeMin=time_min,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime',
        ).execute()

        events = results.get('items', [])

        if not events:
            return f"No events found matching: '{query}'"

        output = []
        for event in events:
            start = event.get('start', {}).get('dateTime', event.get('start', {}).get('date', 'Unknown'))
            end = event.get('end', {}).get('dateTime', event.get('end', {}).get('date', 'Unknown'))
            summary = event.get('summary', '(No title)')
            location = event.get('location', '')
            description = event.get('description', '')

            output.append(f"Event: {summary}")
            output.append(f"  ID: {event.get('id', 'Unknown')}")
            output.append(f"  Start: {start}")
            output.append(f"  End: {end}")
            if location:
                output.append(f"  Location: {location}")
            if description:
                truncated = description[:200] + "..." if len(description) > 200 else description
                output.append(f"  Description: {truncated}")
            output.append("")

        return f"Found {len(events)} events matching '{query}':\n\n" + "\n".join(output)

    except Exception as e:
        return f"Error searching calendar events: {e}"


def calendar_list_calendars() -> str:
    """List all available calendars."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('calendar', 'v3', credentials=creds)

        results = service.calendarList().list().execute()
        calendars = results.get('items', [])

        if not calendars:
            return "No calendars found."

        output = []
        for cal in calendars:
            primary = " (PRIMARY)" if cal.get('primary') else ""
            output.append(f"Calendar: {cal.get('summary', 'Unknown')}{primary}")
            output.append(f"  ID: {cal.get('id', 'Unknown')}")
            output.append(f"  Access Role: {cal.get('accessRole', 'Unknown')}")
            output.append("")

        return f"Found {len(calendars)} calendars:\n\n" + "\n".join(output)

    except Exception as e:
        return f"Error listing calendars: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Google Slides Functions
# ─────────────────────────────────────────────────────────────────────────────

def slides_get_presentation(presentation_id: str) -> str:
    """Get presentation metadata and slide count."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('slides', 'v1', credentials=creds)

        presentation = service.presentations().get(
            presentationId=presentation_id
        ).execute()

        title = presentation.get('title', 'Untitled')
        slides = presentation.get('slides', [])

        output = [
            f"Title: {title}",
            f"Presentation ID: {presentation_id}",
            f"Number of slides: {len(slides)}",
            ""
        ]

        for i, slide in enumerate(slides):
            slide_id = slide.get('objectId', 'unknown')
            output.append(f"Slide {i + 1}: (objectId: {slide_id})")

            # Look for title text in shape elements
            page_elements = slide.get('pageElements', [])
            for element in page_elements:
                shape = element.get('shape', {})
                if 'text' in shape:
                    text_content = []
                    for text_element in shape['text'].get('textElements', []):
                        text_run = text_element.get('textRun', {})
                        if 'content' in text_run:
                            text_content.append(text_run['content'].strip())
                    combined = ' '.join(t for t in text_content if t)
                    if combined:
                        output.append(f"  Title/Text: {combined}")

            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Error getting presentation: {e}"


def slides_get_slide(presentation_id: str, slide_index: int = 0) -> str:
    """Get content from a specific slide (0-indexed)."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('slides', 'v1', credentials=creds)

        presentation = service.presentations().get(
            presentationId=presentation_id
        ).execute()

        slides = presentation.get('slides', [])

        if slide_index < 0 or slide_index >= len(slides):
            return f"Error: Slide index {slide_index} out of range. Presentation has {len(slides)} slides (0-{len(slides) - 1})."

        slide = slides[slide_index]
        slide_id = slide.get('objectId', 'unknown')

        output = [
            f"Slide {slide_index + 1} (objectId: {slide_id})",
            f"Presentation: {presentation.get('title', 'Untitled')}",
            ""
        ]

        page_elements = slide.get('pageElements', [])
        shape_num = 0

        for element in page_elements:
            shape = element.get('shape', {})
            if 'text' in shape:
                shape_num += 1
                element_id = element.get('objectId', 'unknown')
                output.append(f"Shape {shape_num} (objectId: {element_id}):")

                text_content = []
                for text_element in shape['text'].get('textElements', []):
                    text_run = text_element.get('textRun', {})
                    if 'content' in text_run:
                        text_content.append(text_run['content'])

                combined = ''.join(text_content).strip()
                if combined:
                    output.append(f"  {combined}")
                else:
                    output.append("  (empty)")
                output.append("")

        if shape_num == 0:
            output.append("(No text shapes found on this slide)")

        return "\n".join(output)

    except Exception as e:
        return f"Error getting slide content: {e}"


def slides_create_presentation(title: str, folder_id: str = None) -> str:
    """Create a new Google Slides presentation."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('slides', 'v1', credentials=creds)

        presentation = service.presentations().create(
            body={'title': title}
        ).execute()

        pres_id = presentation.get('presentationId', '')
        pres_url = f"https://docs.google.com/presentation/d/{pres_id}/edit"

        result = f"Presentation created successfully!\nTitle: {title}\nID: {pres_id}\nURL: {pres_url}"

        # If folder_id provided, move via Drive API
        if folder_id:
            try:
                drive_service = build('drive', 'v3', credentials=creds)
                # Get current parents to remove
                file = drive_service.files().get(
                    fileId=pres_id,
                    fields='parents'
                ).execute()
                previous_parents = ",".join(file.get('parents', []))

                drive_service.files().update(
                    fileId=pres_id,
                    addParents=folder_id,
                    removeParents=previous_parents,
                    fields='id, parents'
                ).execute()
                result += f"\nMoved to folder: {folder_id}"
            except Exception as e:
                result += f"\nNote: Presentation created but could not move to folder: {e}"

        return result

    except Exception as e:
        return f"Error creating presentation: {e}"


def slides_add_slide(presentation_id: str, layout: str = "BLANK") -> str:
    """Add a new slide to a presentation."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('slides', 'v1', credentials=creds)

        requests = [
            {
                'createSlide': {
                    'slideLayoutReference': {
                        'predefinedLayout': layout
                    }
                }
            }
        ]

        response = service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': requests}
        ).execute()

        # Extract the new slide's object ID from the response
        create_slide_response = response.get('replies', [{}])[0].get('createSlide', {})
        new_slide_id = create_slide_response.get('objectId', 'unknown')

        return f"Slide added successfully!\nNew slide objectId: {new_slide_id}\nLayout: {layout}\nPresentation ID: {presentation_id}"

    except Exception as e:
        return f"Error adding slide: {e}"


def slides_add_text(presentation_id: str, slide_id: str, text: str, x: float = 100, y: float = 100, width: float = 400, height: float = 300) -> str:
    """Add a text box to a slide."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('slides', 'v1', credentials=creds)

        # Generate a unique object ID for the text box
        textbox_id = f"textbox_{uuid.uuid4().hex[:8]}"

        # Convert points to EMU (English Metric Units): 1 point = 12700 EMU
        emu_x = int(x * 12700)
        emu_y = int(y * 12700)
        emu_width = int(width * 12700)
        emu_height = int(height * 12700)

        requests = [
            {
                'createShape': {
                    'objectId': textbox_id,
                    'shapeType': 'TEXT_BOX',
                    'elementProperties': {
                        'pageObjectId': slide_id,
                        'size': {
                            'width': {'magnitude': emu_width, 'unit': 'EMU'},
                            'height': {'magnitude': emu_height, 'unit': 'EMU'}
                        },
                        'transform': {
                            'scaleX': 1,
                            'scaleY': 1,
                            'translateX': emu_x,
                            'translateY': emu_y,
                            'unit': 'EMU'
                        }
                    }
                }
            },
            {
                'insertText': {
                    'objectId': textbox_id,
                    'text': text,
                    'insertionIndex': 0
                }
            }
        ]

        service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': requests}
        ).execute()

        return f"Text box added successfully!\nTextbox ID: {textbox_id}\nSlide: {slide_id}\nText: {text[:100]}{'...' if len(text) > 100 else ''}\nPosition: ({x}, {y}) Size: {width}x{height} points"

    except Exception as e:
        return f"Error adding text box: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Google Sheets Functions
# ─────────────────────────────────────────────────────────────────────────────

def sheets_read(spreadsheet_id: str, range: str = "Sheet1") -> str:
    """Read data from a Google Sheets spreadsheet."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)

        # Get spreadsheet title
        spreadsheet = service.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            fields='properties.title'
        ).execute()
        title = spreadsheet.get('properties', {}).get('title', 'Unknown')

        # Get values
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range
        ).execute()

        values = result.get('values', [])

        if not values:
            return f"Spreadsheet: {title}\nRange: {range}\n\nNo data found."

        # Format as table
        output = []
        output.append(f"Spreadsheet: {title}")
        output.append(f"Range: {range}")
        output.append(f"Rows: {len(values)}")
        output.append("")

        # Calculate column widths for alignment
        col_count = max(len(row) for row in values)
        col_widths = [0] * col_count
        for row in values:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Build table rows
        for row_idx, row in enumerate(values):
            # Pad row to full column count
            padded_row = list(row) + [''] * (col_count - len(row))
            formatted = ' | '.join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(padded_row)
            )
            output.append(formatted)

            # Add separator after header row
            if row_idx == 0:
                separator = '-+-'.join('-' * col_widths[i] for i in range(col_count))
                output.append(separator)

        return "\n".join(output)

    except Exception as e:
        return f"Error reading spreadsheet: {e}"


def sheets_write(spreadsheet_id: str, range: str, values: list) -> str:
    """Write data to a Google Sheets spreadsheet. Values is a list of lists (rows)."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)

        body = {'values': values}

        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range,
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()

        updated_cells = result.get('updatedCells', 0)
        updated_rows = result.get('updatedRows', 0)
        updated_range = result.get('updatedRange', range)

        return (
            f"Data written successfully!\n"
            f"Range: {updated_range}\n"
            f"Rows updated: {updated_rows}\n"
            f"Cells updated: {updated_cells}"
        )

    except Exception as e:
        return f"Error writing to spreadsheet: {e}"


def sheets_append(spreadsheet_id: str, range: str, values: list) -> str:
    """Append rows to a Google Sheets spreadsheet. Values is a list of lists (rows)."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)

        body = {'values': values}

        result = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=range,
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()

        updates = result.get('updates', {})
        updated_range = updates.get('updatedRange', range)
        updated_rows = updates.get('updatedRows', 0)
        updated_cells = updates.get('updatedCells', 0)

        return (
            f"Data appended successfully!\n"
            f"Range: {updated_range}\n"
            f"Rows appended: {updated_rows}\n"
            f"Cells updated: {updated_cells}"
        )

    except Exception as e:
        return f"Error appending to spreadsheet: {e}"


def sheets_create(title: str, folder_id: str = None) -> str:
    """Create a new Google Sheets spreadsheet."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)

        spreadsheet_body = {
            'properties': {
                'title': title
            }
        }

        spreadsheet = service.spreadsheets().create(
            body=spreadsheet_body,
            fields='spreadsheetId,spreadsheetUrl,properties.title'
        ).execute()

        spreadsheet_id = spreadsheet.get('spreadsheetId')
        spreadsheet_url = spreadsheet.get('spreadsheetUrl')

        # Move to folder if specified
        if folder_id:
            drive_service = build('drive', 'v3', credentials=creds)
            # Get current parents
            file = drive_service.files().get(
                fileId=spreadsheet_id,
                fields='parents'
            ).execute()
            previous_parents = ','.join(file.get('parents', []))

            # Move to new folder
            drive_service.files().update(
                fileId=spreadsheet_id,
                addParents=folder_id,
                removeParents=previous_parents,
                fields='id, parents'
            ).execute()

        result = (
            f"Spreadsheet created successfully!\n"
            f"Title: {title}\n"
            f"ID: {spreadsheet_id}\n"
            f"URL: {spreadsheet_url}"
        )

        if folder_id:
            result += f"\nMoved to folder: {folder_id}"

        return result

    except Exception as e:
        return f"Error creating spreadsheet: {e}"


def sheets_list_sheets(spreadsheet_id: str) -> str:
    """List all sheet tabs in a Google Sheets spreadsheet."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)

        spreadsheet = service.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            fields='properties.title,sheets.properties'
        ).execute()

        title = spreadsheet.get('properties', {}).get('title', 'Unknown')
        sheets = spreadsheet.get('sheets', [])

        if not sheets:
            return f"Spreadsheet: {title}\n\nNo sheets found."

        output = []
        output.append(f"Spreadsheet: {title}")
        output.append(f"Total sheets: {len(sheets)}")
        output.append("")

        for sheet in sheets:
            props = sheet.get('properties', {})
            output.append(f"Sheet: {props.get('title', 'Unknown')}")
            output.append(f"  ID: {props.get('sheetId', 'N/A')}")
            output.append(f"  Index: {props.get('index', 'N/A')}")
            output.append(f"  Type: {props.get('sheetType', 'N/A')}")
            grid_props = props.get('gridProperties', {})
            if grid_props:
                output.append(f"  Rows: {grid_props.get('rowCount', 'N/A')}")
                output.append(f"  Columns: {grid_props.get('columnCount', 'N/A')}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Error listing sheets: {e}"


def sheets_clear(spreadsheet_id: str, range: str) -> str:
    """Clear a range of cells in a Google Sheets spreadsheet."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)

        result = service.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range=range,
            body={}
        ).execute()

        cleared_range = result.get('clearedRange', range)

        return f"Range cleared successfully!\nCleared range: {cleared_range}"

    except Exception as e:
        return f"Error clearing range: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# YouTube Functions
# ─────────────────────────────────────────────────────────────────────────────

def youtube_search(query: str, max_results: int = 5, video_type: str = "video") -> str:
    """Search YouTube for videos, channels, or playlists."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('youtube', 'v3', credentials=creds)

        results = service.search().list(
            part='snippet',
            q=query,
            maxResults=max_results,
            type=video_type
        ).execute()

        items = results.get('items', [])

        if not items:
            return f"No results found for query: '{query}'"

        output = []
        for item in items:
            snippet = item.get('snippet', {})
            title = snippet.get('title', 'Unknown')
            channel = snippet.get('channelTitle', 'Unknown')
            published = snippet.get('publishedAt', 'Unknown')

            kind = item.get('id', {}).get('kind', '')
            if 'video' in kind:
                video_id = item['id'].get('videoId', '')
                url = f"https://youtube.com/watch?v={video_id}"
                output.append(f"Title: {title}")
                output.append(f"  Channel: {channel}")
                output.append(f"  Published: {published}")
                output.append(f"  Video ID: {video_id}")
                output.append(f"  URL: {url}")
            elif 'channel' in kind:
                channel_id = item['id'].get('channelId', '')
                url = f"https://youtube.com/channel/{channel_id}"
                output.append(f"Title: {title}")
                output.append(f"  Channel ID: {channel_id}")
                output.append(f"  Published: {published}")
                output.append(f"  URL: {url}")
            elif 'playlist' in kind:
                playlist_id = item['id'].get('playlistId', '')
                url = f"https://youtube.com/playlist?list={playlist_id}"
                output.append(f"Title: {title}")
                output.append(f"  Channel: {channel}")
                output.append(f"  Published: {published}")
                output.append(f"  Playlist ID: {playlist_id}")
                output.append(f"  URL: {url}")

            output.append("")

        return f"Found {len(items)} results for '{query}':\n\n" + "\n".join(output)

    except Exception as e:
        return f"Error searching YouTube: {e}"


def youtube_get_video(video_id: str) -> str:
    """Get detailed information about a YouTube video."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('youtube', 'v3', credentials=creds)

        results = service.videos().list(
            part='snippet,statistics,contentDetails',
            id=video_id
        ).execute()

        items = results.get('items', [])

        if not items:
            return f"No video found with ID: {video_id}"

        video = items[0]
        snippet = video.get('snippet', {})
        statistics = video.get('statistics', {})
        content_details = video.get('contentDetails', {})

        title = snippet.get('title', 'Unknown')
        channel = snippet.get('channelTitle', 'Unknown')
        published = snippet.get('publishedAt', 'Unknown')
        description = snippet.get('description', '')
        if len(description) > 500:
            description = description[:500] + "..."

        view_count = statistics.get('viewCount', 'N/A')
        like_count = statistics.get('likeCount', 'N/A')
        comment_count = statistics.get('commentCount', 'N/A')
        duration = content_details.get('duration', 'N/A')
        url = f"https://youtube.com/watch?v={video_id}"

        output = [
            f"Title: {title}",
            f"Channel: {channel}",
            f"Published: {published}",
            f"Duration: {duration}",
            f"Views: {view_count}",
            f"Likes: {like_count}",
            f"Comments: {comment_count}",
            f"URL: {url}",
            "",
            f"Description:",
            description
        ]

        return "\n".join(output)

    except Exception as e:
        return f"Error getting video info: {e}"


def youtube_get_channel(channel_id: str) -> str:
    """Get information about a YouTube channel."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('youtube', 'v3', credentials=creds)

        results = service.channels().list(
            part='snippet,statistics,contentDetails',
            id=channel_id
        ).execute()

        items = results.get('items', [])

        if not items:
            return f"No channel found with ID: {channel_id}"

        channel = items[0]
        snippet = channel.get('snippet', {})
        statistics = channel.get('statistics', {})

        title = snippet.get('title', 'Unknown')
        description = snippet.get('description', '')
        if len(description) > 500:
            description = description[:500] + "..."

        subscriber_count = statistics.get('subscriberCount', 'N/A')
        video_count = statistics.get('videoCount', 'N/A')
        view_count = statistics.get('viewCount', 'N/A')
        url = f"https://youtube.com/channel/{channel_id}"

        output = [
            f"Channel: {title}",
            f"Channel ID: {channel_id}",
            f"Subscribers: {subscriber_count}",
            f"Videos: {video_count}",
            f"Total Views: {view_count}",
            f"URL: {url}",
            "",
            f"Description:",
            description
        ]

        return "\n".join(output)

    except Exception as e:
        return f"Error getting channel info: {e}"


def youtube_list_playlists(channel_id: str = None, max_results: int = 10) -> str:
    """List playlists for a channel or the authenticated user."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('youtube', 'v3', credentials=creds)

        if channel_id:
            results = service.playlists().list(
                part='snippet,contentDetails',
                channelId=channel_id,
                maxResults=max_results
            ).execute()
        else:
            results = service.playlists().list(
                part='snippet,contentDetails',
                mine=True,
                maxResults=max_results
            ).execute()

        items = results.get('items', [])

        if not items:
            return "No playlists found."

        output = []
        for item in items:
            snippet = item.get('snippet', {})
            content_details = item.get('contentDetails', {})

            title = snippet.get('title', 'Unknown')
            playlist_id = item.get('id', 'Unknown')
            item_count = content_details.get('itemCount', 'N/A')
            description = snippet.get('description', '')
            if len(description) > 200:
                description = description[:200] + "..."

            output.append(f"Title: {title}")
            output.append(f"  Playlist ID: {playlist_id}")
            output.append(f"  Items: {item_count}")
            if description:
                output.append(f"  Description: {description}")
            output.append("")

        return f"Found {len(items)} playlists:\n\n" + "\n".join(output)

    except Exception as e:
        return f"Error listing playlists: {e}"


def youtube_get_playlist_items(playlist_id: str, max_results: int = 20) -> str:
    """List videos in a YouTube playlist."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('youtube', 'v3', credentials=creds)

        results = service.playlistItems().list(
            part='snippet',
            playlistId=playlist_id,
            maxResults=max_results
        ).execute()

        items = results.get('items', [])

        if not items:
            return f"No items found in playlist: {playlist_id}"

        output = []
        for item in items:
            snippet = item.get('snippet', {})
            position = snippet.get('position', 'N/A')
            title = snippet.get('title', 'Unknown')
            channel = snippet.get('videoOwnerChannelTitle', 'Unknown')
            video_id = snippet.get('resourceId', {}).get('videoId', '')
            url = f"https://youtube.com/watch?v={video_id}"

            output.append(f"{position}. {title}")
            output.append(f"   Video ID: {video_id}")
            output.append(f"   Channel: {channel}")
            output.append(f"   URL: {url}")
            output.append("")

        return f"Playlist {playlist_id} - {len(items)} items:\n\n" + "\n".join(output)

    except Exception as e:
        return f"Error getting playlist items: {e}"


def youtube_create_playlist(title: str, description: str = "", privacy: str = "private") -> str:
    """Create a new YouTube playlist."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('youtube', 'v3', credentials=creds)

        result = service.playlists().insert(
            part='snippet,status',
            body={
                'snippet': {
                    'title': title,
                    'description': description
                },
                'status': {
                    'privacyStatus': privacy
                }
            }
        ).execute()

        playlist_id = result.get('id', 'Unknown')
        url = f"https://youtube.com/playlist?list={playlist_id}"

        return f"Playlist created successfully!\nTitle: {title}\nPlaylist ID: {playlist_id}\nPrivacy: {privacy}\nURL: {url}"

    except Exception as e:
        return f"Error creating playlist: {e}"


def youtube_add_to_playlist(playlist_id: str, video_id: str) -> str:
    """Add a video to a YouTube playlist."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('youtube', 'v3', credentials=creds)

        result = service.playlistItems().insert(
            part='snippet',
            body={
                'snippet': {
                    'playlistId': playlist_id,
                    'resourceId': {
                        'kind': 'youtube#video',
                        'videoId': video_id
                    }
                }
            }
        ).execute()

        item_id = result.get('id', 'Unknown')

        return f"Video added to playlist successfully!\nPlaylist ID: {playlist_id}\nVideo ID: {video_id}\nPlaylist Item ID: {item_id}"

    except Exception as e:
        return f"Error adding video to playlist: {e}"


def youtube_get_comments(video_id: str, max_results: int = 10) -> str:
    """Get comments on a YouTube video."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('youtube', 'v3', credentials=creds)

        results = service.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_results,
            order='relevance'
        ).execute()

        items = results.get('items', [])

        if not items:
            return f"No comments found for video: {video_id}"

        output = []
        for item in items:
            top_comment = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
            author = top_comment.get('authorDisplayName', 'Unknown')
            text = top_comment.get('textDisplay', '')
            if len(text) > 200:
                text = text[:200] + "..."
            like_count = top_comment.get('likeCount', 0)
            published = top_comment.get('publishedAt', 'Unknown')

            output.append(f"Author: {author}")
            output.append(f"  Comment: {text}")
            output.append(f"  Likes: {like_count}")
            output.append(f"  Published: {published}")
            output.append("")

        return f"Found {len(items)} comments for video {video_id}:\n\n" + "\n".join(output)

    except Exception as e:
        return f"Error getting comments: {e}"


def youtube_post_comment(video_id: str, text: str) -> str:
    """Post a comment on a YouTube video."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('youtube', 'v3', credentials=creds)

        result = service.commentThreads().insert(
            part='snippet',
            body={
                'snippet': {
                    'videoId': video_id,
                    'topLevelComment': {
                        'snippet': {
                            'textOriginal': text
                        }
                    }
                }
            }
        ).execute()

        comment_id = result.get('id', 'Unknown')

        return f"Comment posted successfully!\nVideo ID: {video_id}\nComment ID: {comment_id}\nText: {text}"

    except Exception as e:
        return f"Error posting comment: {e}"


def youtube_my_channel() -> str:
    """Get the authenticated user's YouTube channel info."""
    status = check_google_setup()
    if status != "OK":
        return status

    try:
        creds = get_credentials()
        service = build('youtube', 'v3', credentials=creds)

        results = service.channels().list(
            part='snippet,statistics,contentDetails',
            mine=True
        ).execute()

        items = results.get('items', [])

        if not items:
            return "No YouTube channel found for the authenticated user."

        channel = items[0]
        snippet = channel.get('snippet', {})
        statistics = channel.get('statistics', {})
        content_details = channel.get('contentDetails', {})

        title = snippet.get('title', 'Unknown')
        channel_id = channel.get('id', 'Unknown')
        subscriber_count = statistics.get('subscriberCount', 'N/A')
        video_count = statistics.get('videoCount', 'N/A')
        view_count = statistics.get('viewCount', 'N/A')
        uploads_playlist = content_details.get('relatedPlaylists', {}).get('uploads', 'N/A')

        output = [
            f"Channel: {title}",
            f"Channel ID: {channel_id}",
            f"Subscribers: {subscriber_count}",
            f"Videos: {video_count}",
            f"Total Views: {view_count}",
            f"Uploads Playlist ID: {uploads_playlist}",
        ]

        return "\n".join(output)

    except Exception as e:
        return f"Error getting channel info: {e}"


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
