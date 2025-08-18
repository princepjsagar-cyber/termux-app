## Community Growth Bot

Telegram bot to help grow communities organically. Built with `python-telegram-bot`.

### Setup

1. Create a bot with `@BotFather` and copy the token.
2. Copy `.env.example` to `.env` and fill in your details:

```bash
cp .env.example .env
```

3. Install dependencies (prefer a virtual environment):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the bot:

```bash
python bot.py
```

### Environment variables

- `BOT_TOKEN`: Your bot token from `@BotFather`.
- `ADMIN_ID` (optional): Your Telegram numeric user ID.
- `VT_API_KEY` (optional): VirusTotal API key for `/scanurl`.
- `GSB_API_KEY` (optional): Google Safe Browsing API key for `/scanurl`.

### Notes

- The bot uses HTML parse mode for safe, clean formatting.
- For `/invite`, ensure the bot is an admin in the group with permission to manage invite links.

### Commands

- `/securitytips` — Practical OWASP-style guidance.
- `/scanurl <url>` — Checks URL reputation (VirusTotal/GSB if keys set).
- `/headers <url>` — Audits common security headers and recommends fixes.
- `/securitynews` — Latest items from CISA Known Exploited Vulnerabilities feed.
- `/disclose` — Responsible disclosure guide and email template.
