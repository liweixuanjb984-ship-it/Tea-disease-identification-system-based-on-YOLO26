#!/usr/bin/env bash
set -e

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-5000}"
export YOLO_DEVICE="${YOLO_DEVICE:-cpu}"

python -m pip install -r requirements.txt
python app.py
