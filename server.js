const express = require('express');
const path = require('path');
const cors = require('cors');
const http = require('http');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const server = http.createServer(app);
const { Server } = require('socket.io');
const io = new Server(server, { cors: { origin: '*', methods: ['GET', 'POST'] } });
const fetch = require('node-fetch');

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

// Bot status/info
app.get('/api/bot-info', async (req, res) => {
  try {
    if (!bot) return res.status(200).json({ online: false, reason: 'no-token' });
    const me = await bot.getMe();
    return res.status(200).json({ online: true, me });
  } catch (err) {
    return res.status(200).json({ online: false, error: String(err && err.message || err) });
  }
});

// Visible commands list for UI
const visibleCommands = [
  { command: '/help', description: 'Show all available commands' },
  { command: '/start', description: 'Welcome message and quick tips' },
  { command: '/status', description: 'Show bot/server status' },
  { command: '/scan <target>', description: 'Run a simulated security scan' },
  { command: '/users', description: 'Show online users (simulated)' },
  { command: '/monitor', description: 'Start monitoring (simulated)' },
  { command: '/ai <lang>|<question>', description: 'Mega AI: real-time contextual answer translated to <lang>' },
  { command: '/translate <lang>|<text>', description: 'Translate text to <lang> (e.g., /translate en|hola mundo)' },
  { command: '/idea <topic>', description: 'Generate advanced technology ideas' }
];

app.get('/api/commands', (req, res) => {
  res.json({ commands: visibleCommands });
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

// Translation endpoint
const translate = require('@vitalets/google-translate-api');
app.post('/api/translate', async (req, res) => {
  try {
    const { text, to } = req.body || {};
    if (!text || !to) return res.status(400).json({ error: 'text and to required' });
    const result = await translate(text, { to });
    res.json({ text: result.text, from: result.from?.language?.iso || 'auto', to });
  } catch (e) {
    res.status(500).json({ error: 'translation_failed', details: String(e && e.message || e) });
  }
});

// Advanced ideas endpoint (simulated generator)
app.post('/api/ideas', (req, res) => {
  const { topic } = req.body || {};
  const base = (topic && String(topic).trim()) || 'emerging tech';
  const ideas = [
    `Self-healing microservices for ${base} using eBPF-driven anomaly remediation`,
    `Federated on-device learning for privacy-preserving ${base} analytics`,
    `Digital twin platform for ${base} with real-time graph event streams`,
    `Zero-trust mesh for ${base} leveraging hardware-backed keys and SPIFFE`,
    `Multi-agent workflow orchestration for ${base} with verifiable execution logs`
  ];
  res.json({ topic: base, ideas });
});

// Smart answer endpoint (demo using simple knowledge and optional external news)
app.post('/api/answer', async (req, res) => {
  try {
    const { question } = req.body || {};
    if (!question) return res.status(400).json({ error: 'question required' });
    let snippets = [];
    try {
      const r = await fetch('https://worldtimeapi.org/api/ip');
      if (r.ok) {
        const j = await r.json();
        snippets.push(`Current time: ${j.datetime}`);
        snippets.push(`Timezone: ${j.timezone}`);
      }
    } catch (_) {}
    const answer = `Based on my analysis, ${question} intersects with current context: ${snippets.join(' | ') || 'context unavailable'}.`;
    res.json({ answer, context: snippets });
  } catch (e) {
    res.status(500).json({ error: 'answer_failed' });
  }
});

// Unified AI endpoint: answer + translation
app.post('/api/ai', async (req, res) => {
  try {
    const { question, to } = req.body || {};
    if (!question) return res.status(400).json({ error: 'question required' });
    // build base answer
    let snippets = [];
    try {
      const r = await fetch('https://worldtimeapi.org/api/ip');
      if (r.ok) {
        const j = await r.json();
        snippets.push(`Current time: ${j.datetime}`);
        snippets.push(`Timezone: ${j.timezone}`);
      }
    } catch (_) {}
    let answer = `Based on my analysis, ${question} intersects with current context: ${snippets.join(' | ') || 'context unavailable'}.`;
    let translated = answer;
    if (to) {
      try {
        const resTr = await translate(answer, { to });
        translated = resTr.text;
      } catch (_) {}
    }
    res.json({ answer: translated, base: answer, context: snippets, to: to || 'en' });
  } catch (e) {
    res.status(500).json({ error: 'ai_failed' });
  }
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

  // Register visible commands in Telegram menu
  try {
    bot.setMyCommands(visibleCommands.map(c => ({ command: c.command.split(' ')[0].replace('/', ''), description: c.description })));
  } catch (e) {
    console.warn('Failed to set bot commands:', e && e.message ? e.message : e);
  }

  bot.onText(/\/start/, (msg) => {
    const welcome = 'Advanced Bot is online. Use /help to view all commands.';
    bot.sendMessage(msg.chat.id, welcome);
    io.emit('feed', { type: 'bot', message: `Start from @${msg.from?.username || 'user'} in chat ${msg.chat.id}` });
  });

  bot.onText(/\/help/, (msg) => {
    const text = visibleCommands.map(c => `${c.command} — ${c.description}`).join('\n');
    bot.sendMessage(msg.chat.id, text);
  });

  bot.onText(/\/status/, async (msg) => {
    const me = await bot.getMe();
    bot.sendMessage(msg.chat.id, `Status: online as @${me.username}.`);
  });

  bot.onText(/\/scan(?:\s+(.*))?/, (msg, match) => {
    const target = (match && match[1]) || 'your systems';
    bot.sendMessage(msg.chat.id, `Security scan initiated on ${target}. Scanning for vulnerabilities...`);
    io.emit('feed', { type: 'scan', message: `Scan requested on ${target}` });
  });

  bot.onText(/\/users/, (msg) => {
    bot.sendMessage(msg.chat.id, '5 users currently online. 3 admin users, 2 regular users.');
    io.emit('feed', { type: 'users', message: 'Users requested' });
  });

  bot.onText(/\/monitor/, (msg) => {
    bot.sendMessage(msg.chat.id, 'Real-time monitoring activated. Tracking system activities.');
    io.emit('feed', { type: 'monitor', message: 'Monitoring activated' });
  });

  // /ai <lang>|<question>
  bot.onText(/^\/ai\s+([a-z]{2})\s*\|\s*([\s\S]+)/i, async (msg, match) => {
    const lang = match && match[1];
    const question = match && match[2];
    if (!lang || !question) return bot.sendMessage(msg.chat.id, 'Usage: /ai <lang>|<question>');
    try {
      const resp = await fetch('http://localhost:' + PORT + '/api/ai', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, to: lang })
      });
      const data = await resp.json();
      bot.sendMessage(msg.chat.id, data.answer || 'Working on it...');
      io.emit('feed', { type: 'ai', message: `AI answered (${lang})` });
    } catch (e) {
      bot.sendMessage(msg.chat.id, 'AI failed to answer.');
    }
  });

  // /translate <lang>|<text>
  bot.onText(/^\/translate\s+([a-z]{2})\s*\|\s*([\s\S]+)/i, async (msg, match) => {
    const lang = match && match[1];
    const text = match && match[2];
    if (!lang || !text) return bot.sendMessage(msg.chat.id, 'Usage: /translate <lang>|<text>');
    try {
      const result = await translate(text, { to: lang });
      bot.sendMessage(msg.chat.id, `Translated (${result.from?.language?.iso || 'auto'} -> ${lang}):\n${result.text}`);
      io.emit('feed', { type: 'translate', message: `Translated to ${lang}` });
    } catch (e) {
      bot.sendMessage(msg.chat.id, 'Translation failed.');
    }
  });

  // /idea <topic>
  bot.onText(/^\/idea\s+([\s\S]+)/i, (msg, match) => {
    const topic = (match && match[1]) || 'emerging tech';
    const ideas = [
      `Self-healing microservices for ${topic}`,
      `Federated learning for ${topic}`,
      `Digital twin platform for ${topic}`
    ];
    bot.sendMessage(msg.chat.id, `Advanced ideas for ${topic}:\n- ${ideas.join('\n- ')}`);
    io.emit('feed', { type: 'idea', message: `Ideas generated for ${topic}` });
  });

  bot.on('message', async (msg) => {
    // Fallback handler for non-commands
    if (!msg.text.startsWith('/')) {
      try {
        const resp = await fetch('http://localhost:' + PORT + '/api/answer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: msg.text })
        });
        const data = await resp.json();
        await bot.sendMessage(msg.chat.id, data.answer || 'I am analyzing your question.');
      } catch (_) {
        await bot.sendMessage(msg.chat.id, 'Ask me anything or use /help');
      }
    }
    io.emit('feed', { type: 'message', message: `Message from @${msg.from?.username || 'user'}: ${msg.text.slice(0, 120)}` });
  });
}

// Real-time heartbeat
setInterval(() => {
  io.emit('feed', { type: 'heartbeat', message: `Heartbeat ${new Date().toISOString()}` });
}, 15000);

server.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});

