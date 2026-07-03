#!/usr/bin/env bash
set -e
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-5000}"
python -m pip install -r requirements.txt
python app.py
