const express = require('express');
const path = require('path');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static UI
const publicDir = path.join(__dirname, 'public');
app.use(express.static(publicDir));

// Health endpoint
app.get('/health', (req, res) => {
  res.json({ ok: true, status: 'online' });
});

// Simple stub endpoints to back demo UI actions
app.post('/api/scan', (req, res) => {
  const { url } = req.body || {};
  res.json({ message: `Scan completed for ${url || 'unknown target'}`, issues: [
    'Outdated jQuery library detected',
    'Missing security headers',
    'Form without CSRF protection'
  ]});
});

app.post('/api/pentest', (req, res) => {
  const { target } = req.body || {};
  res.json({ message: `Penetration test completed for ${target || 'unknown target'}`, exploited: 2 });
});

app.get('/api/network', (req, res) => {
  res.json({ message: 'Network analysis complete', connections: 42 });
});

app.post('/api/password-audit', (req, res) => {
  const { password } = req.body || {};
  const strength = !password ? 'unknown' : password.length >= 12 ? 'strong' : password.length >= 8 ? 'medium' : 'weak';
  res.json({ strength });
});

// Telegram bot setup
const TelegramBot = require('node-telegram-bot-api');
const token = process.env.TELEGRAM_BOT_TOKEN;

if (!token) {
  console.warn('Warning: TELEGRAM_BOT_TOKEN not set. Bot will not start.');
}

let bot = null;
if (token) {
  // Use polling for simplicity in this environment
  bot = new TelegramBot(token, { polling: true });

  bot.onText(/\/start/, (msg) => {
    bot.sendMessage(msg.chat.id, 'Advanced Bot is online. Use /scan, /users, /monitor, /emergency');
  });

  bot.onText(/\/scan(?:\s+(.*))?/, (msg, match) => {
    const target = (match && match[1]) || 'your systems';
    bot.sendMessage(msg.chat.id, `Security scan initiated on ${target}. Scanning for vulnerabilities...`);
  });

  bot.onText(/\/users/, (msg) => {
    bot.sendMessage(msg.chat.id, '5 users currently online. 3 admin users, 2 regular users.');
  });

  bot.onText(/\/monitor/, (msg) => {
    bot.sendMessage(msg.chat.id, 'Real-time monitoring activated. Tracking system activities.');
  });

  bot.onText(/\/emergency/, (msg) => {
    bot.sendMessage(msg.chat.id, 'EMERGENCY LOCKDOWN INITIATED! All non-essential functions disabled.');
  });

  bot.on('message', (msg) => {
    // Fallback handler for non-commands
    if (!msg.text.startsWith('/')) {
      bot.sendMessage(msg.chat.id, 'Ask me anything or use /scan, /users, /monitor, /emergency');
    }
  });
}

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});

