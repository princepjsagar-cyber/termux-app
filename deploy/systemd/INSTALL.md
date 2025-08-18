## Systemd install (root server)

1. Create data dirs and copy env file:
```bash
sudo mkdir -p /var/lib/telegram-bot/data /var/lib/telegram-bot/backups
sudo cp /workspace/deploy/systemd/telegram-bot.env.example /etc/telegram-bot.env
sudo chmod 600 /etc/telegram-bot.env
sudo chown root:root /etc/telegram-bot.env
```

2. Edit `/etc/telegram-bot.env` and set your tokens and keys.

3. Install the service unit:
```bash
sudo cp /workspace/deploy/systemd/telegram-bot.service /etc/systemd/system/telegram-bot.service
sudo systemctl daemon-reload
sudo systemctl enable telegram-bot
sudo systemctl start telegram-bot
```

4. Check status and logs:
```bash
systemctl status telegram-bot
journalctl -u telegram-bot -f
```

Notes:
- The service uses `/usr/bin/python3` to run `/workspace/bot.py`. Adjust `WorkingDirectory` and path if your code lives elsewhere.
- Ensure the Python dependencies are installed on the server (prefer a venv) and that the service `User` has read access to the code and write access to `DATA_DIR` and `BACKUP_DIR`.