#!/bin/zsh
# Run binance futures/funding/premium downloaders locally through socks5 proxy.
set -e
cd /Users/a.s.tabakov/Developer/invest
export ALL_PROXY='socks5h://192.168.1.1:1080'
export HTTP_PROXY="$ALL_PROXY"
export HTTPS_PROXY="$ALL_PROXY"
exec .venv/bin/python -u src/data/download_binance_futures.py "$@"
