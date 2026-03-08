#!/usr/bin/env python3
"""
Telegram Bot for Trading Alerts & Control.

Sends:
  - 📊 Signal alerts (new positions each rebalance)
  - ✅ Order fills / ❌ errors
  - 💰 Daily PnL summary
  - ⚠️ Drawdown warnings
  - 🛑 DD stop triggered / 🟢 resumed
  - 🔄 Heartbeat every cycle

Commands (via Telegram):
  /status   — Current equity, positions, DD
  /pnl      — Recent PnL history
  /stop     — Stop the bot gracefully
  /start    — Resume trading
  /help     — Show commands

Setup:
  1. Talk to @BotFather on Telegram → /newbot → get token
  2. Get your chat_id: talk to @userinfobot
  3. Add to .env:
     TELEGRAM_TOKEN=123456:ABC-DEF...
     TELEGRAM_CHAT_ID=123456789
"""

import os
import sys
import time
import json
import threading
import traceback
from datetime import datetime, timezone
from functools import wraps

import requests


class TelegramBot:
    """Lightweight Telegram bot for trading alerts."""

    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TELEGRAM_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.enabled = bool(self.token and self.chat_id)
        self._polling = False
        self._stop_flag = threading.Event()
        self._command_handlers = {}
        self._last_update_id = 0

        if not self.enabled:
            print("   ⚠️  Telegram not configured (set TELEGRAM_TOKEN + TELEGRAM_CHAT_ID)")

    # ─── Core messaging ────────────────────────────────────────

    def send(self, text, parse_mode="HTML", silent=False):
        """Send a message. Returns True on success."""
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text[:4096],  # Telegram limit
                    "parse_mode": parse_mode,
                    "disable_notification": silent,
                },
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"   ⚠️  Telegram send error: {e}")
            return False

    def send_document(self, filepath, caption=""):
        """Send a file (e.g. equity curve CSV)."""
        if not self.enabled:
            return False
        try:
            with open(filepath, "rb") as f:
                resp = requests.post(
                    f"{self.base_url}/sendDocument",
                    data={"chat_id": self.chat_id, "caption": caption[:1024]},
                    files={"document": f},
                    timeout=30,
                )
            return resp.status_code == 200
        except Exception as e:
            print(f"   ⚠️  Telegram file send error: {e}")
            return False

    # ─── Trading alert formatters ──────────────────────────────

    def alert_cycle_start(self, cycle_num, equity, dd_pct):
        """Alert: new rebalance cycle starting."""
        dd_icon = "🟢" if dd_pct > -0.05 else ("🟡" if dd_pct > -0.15 else "🔴")
        self.send(
            f"🔄 <b>Cycle #{cycle_num}</b>\n"
            f"💰 Equity: <code>${equity:,.2f}</code>\n"
            f"{dd_icon} DD: <code>{dd_pct:.1%}</code>\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            silent=True,
        )

    def alert_positions(self, positions, capital):
        """Alert: new portfolio positions."""
        if not positions:
            self.send("📊 <b>No positions this cycle</b> (DD stop or no signal)")
            return

        longs = [p for p in positions if p["side"] == "long"]
        shorts = [p for p in positions if p["side"] == "short"]
        total = sum(p["usd"] for p in positions)

        lines = [f"📊 <b>New Positions</b> — ${total:.0f}/{capital:.0f}"]
        lines.append("")

        if longs:
            lines.append("🟢 <b>LONG:</b>")
            for p in sorted(longs, key=lambda x: -x["usd"]):
                lines.append(
                    f"  {p['symbol']:<12s} ${p['usd']:>6.0f}  "
                    f"score={p['score']:+.3f}"
                )

        if shorts:
            lines.append("🔴 <b>SHORT:</b>")
            for p in sorted(shorts, key=lambda x: -x["usd"]):
                lines.append(
                    f"  {p['symbol']:<12s} ${p['usd']:>6.0f}  "
                    f"score={p['score']:+.3f}"
                )

        self.send("\n".join(lines))

    def alert_fills(self, results):
        """Alert: order execution results."""
        if not results:
            return

        filled = [r for r in results if r.get("status") == "filled"]
        errors = [r for r in results if r.get("status") == "error"]
        dry = [r for r in results if r.get("status") == "dry_run"]

        lines = []
        if filled:
            lines.append(f"✅ <b>{len(filled)} orders filled</b>")
        if errors:
            lines.append(f"❌ <b>{len(errors)} orders FAILED:</b>")
            for r in errors:
                lines.append(f"  {r['symbol']} {r['side']} ${r['usd']:.0f} — {r.get('error','?')}")
        if dry:
            lines.append(f"📋 {len(dry)} dry-run orders (not executed)")

        if lines:
            self.send("\n".join(lines))

    def alert_cycle_pnl(self, pnl, equity, dd_pct, settled=None):
        """Alert: cycle PnL after settling positions."""
        icon = "🟢" if pnl >= 0 else "🔴"
        dd_icon = "🟢" if dd_pct > -0.05 else ("🟡" if dd_pct > -0.15 else "🔴")

        lines = [
            f"{icon} <b>Cycle PnL: ${pnl:+.2f}</b>",
            f"💰 Equity: <code>${equity:,.2f}</code>",
            f"{dd_icon} DD: <code>{dd_pct:.1%}</code>",
        ]

        # Top/bottom positions
        if settled:
            best = sorted(settled, key=lambda x: x.get("pnl", 0), reverse=True)
            if best and best[0].get("pnl", 0) > 0:
                lines.append(f"\n🏆 Best:  {best[0]['symbol']} ${best[0]['pnl']:+.2f}")
            if len(best) > 1 and best[-1].get("pnl", 0) < 0:
                lines.append(f"💩 Worst: {best[-1]['symbol']} ${best[-1]['pnl']:+.2f}")

        self.send("\n".join(lines))

    def alert_dd_warning(self, dd_pct, equity):
        """Alert: drawdown approaching stop level."""
        self.send(
            f"⚠️ <b>DRAWDOWN WARNING</b>\n"
            f"DD: <code>{dd_pct:.1%}</code>\n"
            f"Equity: <code>${equity:,.2f}</code>\n"
            f"⚠️ Approaching DD stop! Consider manual intervention.",
        )

    def alert_dd_stop(self, dd_pct, equity):
        """Alert: DD stop triggered, trading halted."""
        self.send(
            f"🛑 <b>DD STOP TRIGGERED</b>\n"
            f"DD: <code>{dd_pct:.1%}</code>\n"
            f"Equity: <code>${equity:,.2f}</code>\n"
            f"Trading HALTED. Will resume when DD recovers.",
        )

    def alert_dd_resume(self, dd_pct, equity):
        """Alert: DD recovered, trading resumed."""
        self.send(
            f"🟢 <b>Trading RESUMED</b>\n"
            f"DD recovered to <code>{dd_pct:.1%}</code>\n"
            f"Equity: <code>${equity:,.2f}</code>",
        )

    def alert_error(self, error_msg, context=""):
        """Alert: error occurred."""
        self.send(
            f"🚨 <b>ERROR</b>\n"
            f"Context: {context}\n"
            f"<code>{str(error_msg)[:500]}</code>",
        )

    def alert_daily_summary(self, state):
        """Daily PnL summary."""
        equity = state.get("equity", 0)
        initial = state.get("initial_capital", 0)
        peak = state.get("peak", 0)
        total_pnl = state.get("total_pnl", 0)
        n_cycles = state.get("n_cycles", 0)
        cycle_pnls = state.get("cycle_pnls", [])

        total_ret = equity / initial - 1 if initial > 0 else 0
        dd = equity / peak - 1 if peak > 0 else 0

        # Last 24h PnL (rough: last 2 cycles for 12h rebal)
        recent_pnl = sum(cycle_pnls[-2:]) if len(cycle_pnls) >= 2 else sum(cycle_pnls[-1:])

        wins = sum(1 for p in cycle_pnls if p > 0)
        wr = wins / len(cycle_pnls) if cycle_pnls else 0

        self.send(
            f"📈 <b>Daily Summary</b>\n"
            f"{'─' * 30}\n"
            f"💰 Equity: <code>${equity:,.2f}</code>\n"
            f"📊 Total return: <code>{total_ret:+.1%}</code>\n"
            f"📉 Max DD: <code>{dd:.1%}</code>\n"
            f"🎯 Win rate: <code>{wr:.0%}</code> ({wins}/{len(cycle_pnls)})\n"
            f"📅 Last 24h: <code>${recent_pnl:+.2f}</code>\n"
            f"🔄 Cycles: {n_cycles}\n"
            f"💵 Total PnL: <code>${total_pnl:+.2f}</code>",
        )

    def alert_heartbeat(self, equity, n_positions):
        """Periodic heartbeat — system is alive."""
        self.send(
            f"💓 Alive | ${equity:,.0f} | {n_positions} pos | "
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}",
            silent=True,
        )

    def alert_startup(self, mode, capital, risk_cfg):
        """Alert: bot starting up."""
        self.send(
            f"🚀 <b>Trading Bot Started</b>\n"
            f"Mode: <code>{mode}</code>\n"
            f"Capital: <code>${capital:,.0f}</code>\n"
            f"Risk: kelly={risk_cfg.get('kelly_frac', 0):.0%}, "
            f"DD_stop={risk_cfg.get('dd_stop', 0):.0%}\n"
            f"Positions: {risk_cfg.get('n_long', 0)}L+{risk_cfg.get('n_short', 0)}S\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        )

    def alert_shutdown(self, reason="manual"):
        """Alert: bot shutting down."""
        self.send(
            f"🔌 <b>Trading Bot Stopped</b>\n"
            f"Reason: {reason}\n"
            f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        )

    # ─── Command polling (optional, for interactive control) ───

    def register_command(self, command, handler):
        """Register a /command handler. handler(bot, chat_id, text) -> str."""
        self._command_handlers[command.lstrip("/")] = handler

    def start_polling(self):
        """Start polling for commands in a background thread."""
        if not self.enabled:
            return

        self._polling = True
        self._stop_flag.clear()

        def _poll():
            while not self._stop_flag.is_set():
                try:
                    resp = requests.get(
                        f"{self.base_url}/getUpdates",
                        params={"offset": self._last_update_id + 1, "timeout": 10},
                        timeout=15,
                    )
                    if resp.status_code != 200:
                        time.sleep(5)
                        continue

                    data = resp.json()
                    for update in data.get("result", []):
                        self._last_update_id = update["update_id"]
                        msg = update.get("message", {})
                        text = msg.get("text", "")
                        chat_id = str(msg.get("chat", {}).get("id", ""))

                        # Security: only respond to our chat_id
                        if chat_id != str(self.chat_id):
                            continue

                        if text.startswith("/"):
                            cmd = text.split()[0].lstrip("/").split("@")[0]
                            handler = self._command_handlers.get(cmd)
                            if handler:
                                try:
                                    reply = handler(self, chat_id, text)
                                    if reply:
                                        self.send(reply)
                                except Exception as e:
                                    self.send(f"❌ Command error: {e}")
                            else:
                                self.send(f"Unknown command: /{cmd}\nUse /help")

                except Exception:
                    time.sleep(5)

        thread = threading.Thread(target=_poll, daemon=True, name="tg-poll")
        thread.start()
        print("   📱 Telegram command polling started")

    def stop_polling(self):
        """Stop the command polling thread."""
        self._stop_flag.set()
        self._polling = False


# ─── Default command handlers ─────────────────────────────────

def cmd_help(bot, chat_id, text):
    return (
        "📋 <b>Commands:</b>\n"
        "/status — Current equity, positions, DD\n"
        "/pnl — Recent PnL history\n"
        "/help — This message"
    )


def cmd_status(bot, chat_id, text):
    """Reads state from trading_state.json."""
    state_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "trading_logs", "trading_state.json"
    )
    if not os.path.exists(state_path):
        return "No trading state found."

    with open(state_path) as f:
        state = json.load(f)

    equity = state.get("equity", 0)
    peak = state.get("peak", 0)
    dd = equity / peak - 1 if peak > 0 else 0
    n_cycles = state.get("n_cycles", 0)
    total_pnl = state.get("total_pnl", 0)
    positions = state.get("sim_positions", [])

    lines = [
        f"📊 <b>Status</b>",
        f"💰 Equity: ${equity:,.2f}",
        f"📉 DD: {dd:.1%}",
        f"🔄 Cycles: {n_cycles}",
        f"💵 Total PnL: ${total_pnl:+.2f}",
        f"📌 Open positions: {len(positions)}",
    ]

    if positions:
        lines.append("")
        for p in positions[:10]:
            lines.append(f"  {p.get('side', '?'):>5s} {p.get('symbol', '?'):<12s} ${p.get('usd', 0):.0f}")

    return "\n".join(lines)


def cmd_pnl(bot, chat_id, text):
    """Show recent cycle PnLs."""
    state_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "trading_logs", "trading_state.json"
    )
    if not os.path.exists(state_path):
        return "No trading state found."

    with open(state_path) as f:
        state = json.load(f)

    cycle_pnls = state.get("cycle_pnls", [])
    eq_history = state.get("equity_history", [])

    if not cycle_pnls:
        return "No PnL history yet."

    lines = [f"📊 <b>Recent PnL</b> (last {min(10, len(cycle_pnls))} cycles)"]
    for i, entry in enumerate(eq_history[-10:]):
        pnl = entry.get("pnl", 0)
        equity = entry.get("equity", 0)
        icon = "🟢" if pnl >= 0 else "🔴"
        ts = entry.get("timestamp", "?")[:16]
        lines.append(f"  {icon} {ts} ${pnl:+.2f} → ${equity:,.0f}")

    total = sum(cycle_pnls)
    wins = sum(1 for p in cycle_pnls if p > 0)
    lines.append(f"\nTotal: ${total:+.2f} | WR: {wins}/{len(cycle_pnls)}")

    return "\n".join(lines)


def setup_default_commands(bot):
    """Register all default commands."""
    bot.register_command("help", cmd_help)
    bot.register_command("status", cmd_status)
    bot.register_command("pnl", cmd_pnl)


# ─── Convenience: create bot from .env ─────────────────────────

def create_bot():
    """Create and configure a TelegramBot from environment variables."""
    # Try loading .env
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            # Manual .env loading
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        os.environ[key.strip()] = val.strip().strip("'\"")

    bot = TelegramBot()
    if bot.enabled:
        setup_default_commands(bot)
        bot.start_polling()
    return bot


# ─── CLI test ──────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick test: send a test message."""
    bot = create_bot()
    if not bot.enabled:
        print("Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env or environment")
        sys.exit(1)

    ok = bot.send("🤖 <b>Test message from trading bot!</b>\nBot is working.")
    if ok:
        print("✅ Test message sent!")
    else:
        print("❌ Failed to send message")
