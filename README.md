# Job AI (Streamlit + Ollama)

One Place for all your job difficulties. Get a Job today with Job AI.

## What this app does

- Creates an account flow via LinkedIn, Google, or manual onboarding.
- Supports LinkedIn profile JSON import and saves user profile to `data/profile_<id>.txt`.
- Uses Ollama model `gpt-oss:120b-cloud` for:
  - job scanning by role + geography,
  - ATS resume generation,
  - recruiter message generation,
  - referral message generation.
- Includes a Playwright LinkedIn browser agent with **dry-run mode** for safer validation before sending actions.

## Prerequisites

- Python 3.10+
- Ollama running locally (or reachable endpoint)
- Model available in Ollama: `gpt-oss:120b-cloud`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Create `.env`:

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b-cloud

# Optional OAuth links in UI
LINKEDIN_CLIENT_ID=
LINKEDIN_REDIRECT_URI=
GOOGLE_CLIENT_ID=
GOOGLE_REDIRECT_URI=

# Required for LinkedIn browser automation
LINKEDIN_SESSION_COOKIE=your_li_at_cookie
LINKEDIN_CSRF_TOKEN="ajax:xxxxxxxx"
```

Run app:

```bash
streamlit run app.py
```

## LinkedIn / Google auth behavior

- The app now provides real OAuth authorization link generation for LinkedIn and Google when client IDs + redirect URIs are configured.
- In production you should exchange `code` for tokens server-side and store securely.

## Tools/services used and API key instructions

### 1) Ollama
- Purpose: LLM inference (jobs, resume, messages).
- API key: **none** for local Ollama.
- Hosted Ollama-compatible services may require provider API keys (set provider endpoint in `OLLAMA_BASE_URL`).

### 2) LinkedIn OAuth (optional but recommended for production auth)
- Create app in LinkedIn Developer Portal.
- Configure OAuth redirect URI.
- Enable scopes/products needed.
- Save Client ID/Secret in environment/secret manager.

### 3) Google OAuth (optional)
- Create OAuth client in Google Cloud Console.
- Configure consent screen + redirect URI.
- Save Client ID/Secret in environment/secret manager.

## Safety

- Respect LinkedIn Terms and local laws.
- Use dry-run before enabling real actions.
- Never hardcode credentials.
