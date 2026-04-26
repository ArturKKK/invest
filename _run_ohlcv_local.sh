#!/bin/zsh
set -e
cd /Users/a.s.tabakov/Developer/invest
export ALL_PROXY='socks5h://192.168.1.1:1080'
export HTTP_PROXY="$ALL_PROXY"
export HTTPS_PROXY="$ALL_PROXY"
exec .venv/bin/python -u _update_ohlcv_vision.py
