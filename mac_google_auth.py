#!/usr/bin/env python3
"""Run this on your Mac to authenticate with Google."""

from google_auth_oauthlib.flow import InstalledAppFlow
import os
import glob

# Find the credentials file on Desktop
desktop = os.path.expanduser("~/Desktop")
cred_files = glob.glob(os.path.join(desktop, "client_secret_*.json"))

if not cred_files:
    print("Error: No client_secret_*.json file found on Desktop")
    print("Download it from Google Cloud Console first.")
    exit(1)

cred_file = cred_files[0]
print(f"Using: {cred_file}")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

flow = InstalledAppFlow.from_client_secrets_file(cred_file, SCOPES)
creds = flow.run_local_server(port=0)

token_path = "/tmp/google_token.json"
with open(token_path, "w") as f:
    f.write(creds.to_json())

print(f"\nSuccess! Token saved to {token_path}")
print("\nNow run this to copy it to the server:")
print(f"  scp {token_path} mboard76@spark-e26c:.config/maude/google_token.json")
