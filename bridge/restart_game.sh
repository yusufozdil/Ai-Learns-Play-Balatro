#!/usr/bin/env bash
# (Re)start Balatro with BalatroBot (no --fast: BalatroBot issue #234 corrupts the deck order).
# Kills a running instance first. Logs go to ~/balatro-bridge/logs, output to serve.out.
set -u
pkill -f "balatrobot serve" 2>/dev/null
sleep 1
pkill -f "Balatro.exe" 2>/dev/null
sleep 2
cd "$HOME/balatro-bridge" || exit 1
setsid nohup balatrobot serve --logs-path "$HOME/balatro-bridge/logs" > "$HOME/balatro-bridge/serve.out" 2>&1 &
for i in $(seq 1 60); do
  sleep 2
  if balatrobot api health 2>/dev/null | grep -q ok; then
    echo "game up after $((i * 2))s"
    exit 0
  fi
done
echo "game did not come up" >&2
exit 1
