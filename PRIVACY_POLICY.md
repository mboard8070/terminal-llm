# Privacy Policy

**Last updated:** February 25, 2026

MAUDE (Multi-Agent Unified Dispatch Engine) is a personal, self-hosted AI assistant. This policy describes how data is handled.

## Data Processing

MAUDE runs on your own hardware. All data processing occurs locally on your device unless you explicitly use cloud-connected features.

### Local Processing
- Conversation history is stored locally on the device running MAUDE
- File operations (read, write, edit, search) operate on local filesystems
- Audio transcription can run locally via Whisper
- Image analysis runs locally via LLaVA

### Cloud Services
When you use the following features, data is sent to third-party APIs:

- **Mistral AI / Codestral** — User messages and conversation context are sent to Mistral's API for inference. Subject to [Mistral's privacy policy](https://mistral.ai/terms/#privacy-policy).
- **Google Workspace** — When you use Gmail, Google Drive, Sheets, Calendar, Slides, Contacts, or YouTube tools, MAUDE accesses your Google account data via OAuth 2.0. Only the scopes you authorize are accessed. Subject to [Google's privacy policy](https://policies.google.com/privacy).
- **DuckDuckGo** — Web search queries are sent to DuckDuckGo. Subject to [DuckDuckGo's privacy policy](https://duckduckgo.com/privacy).

### What MAUDE Does NOT Do
- Does not collect analytics or telemetry
- Does not share your data with any party beyond the third-party APIs you choose to use
- Does not store data on any server you do not control
- Does not sell or monetize your data

## Data Storage

- **Conversations** are stored locally in memory and optional local files
- **Google OAuth tokens** are stored locally at `~/.config/maude/google_token.json`
- **Shared files** are stored in the `shared/` directory on your device
- **No data is stored in the cloud** beyond what third-party services retain per their own policies

## Google API Usage

MAUDE's use of Google APIs adheres to the [Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy), including the Limited Use requirements. MAUDE:

- Only requests scopes necessary for the features you use
- Does not transfer Google user data to third parties except as necessary to provide the service
- Does not use Google user data for advertising
- Allows you to revoke access at any time via [Google Account Permissions](https://myaccount.google.com/permissions)

## Data Retention

All data is retained locally on your hardware. You can delete it at any time by:
- Clearing conversation history (`/clear` command)
- Removing local files and configuration (`~/.config/maude/`)
- Revoking third-party access (Google, Mistral) through their respective account settings

## Contact

For questions about this policy, open an issue at [github.com/mboard8070/terminal-llm](https://github.com/mboard8070/terminal-llm/issues).
