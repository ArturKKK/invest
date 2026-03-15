#!/usr/bin/env bash
# cron_wrapper.sh — Run a cron job and send Telegram alert on failure/success.
#
# Usage:
#   cron_wrapper.sh <job_name> <command> [args...]
#
# Sends ✅ on success, ❌ on failure with last 10 lines of output.
# Requires TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env or environment.
set -uo pipefail

JOB_NAME="${1:?Usage: cron_wrapper.sh <job_name> <command> [args...]}"
shift

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env for TELEGRAM_TOKEN and TELEGRAM_CHAT_ID
if [[ -f "$ROOT/.env" ]]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

TG_TOKEN="${TELEGRAM_TOKEN:-}"
TG_CHAT="${TELEGRAM_CHAT_ID:-}"

send_tg() {
    local text="$1"
    if [[ -n "$TG_TOKEN" && -n "$TG_CHAT" ]]; then
        curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
            -d chat_id="$TG_CHAT" \
            -d parse_mode="HTML" \
            -d text="$text" \
            -d disable_notification="${2:-false}" \
            > /dev/null 2>&1 || true
    fi
}

# Run the actual command, capture output
TMPFILE=$(mktemp /tmp/cron_wrapper.XXXXXX)
START_TS=$(date -u '+%Y-%m-%d %H:%M UTC')

"$@" > "$TMPFILE" 2>&1
EXIT_CODE=$?

END_TS=$(date -u '+%Y-%m-%d %H:%M UTC')

if [[ $EXIT_CODE -eq 0 ]]; then
    # Success — silent notification
    send_tg "✅ <b>Cron OK:</b> ${JOB_NAME}
⏰ ${END_TS}" true
else
    # Failure — loud notification with error details
    TAIL=$(tail -10 "$TMPFILE" | head -c 800)
    send_tg "❌ <b>Cron FAIL:</b> ${JOB_NAME}
⏰ ${END_TS}
Exit code: ${EXIT_CODE}

<pre>${TAIL}</pre>" false
fi

# Also echo to stdout for log file
cat "$TMPFILE"
rm -f "$TMPFILE"
exit $EXIT_CODE
