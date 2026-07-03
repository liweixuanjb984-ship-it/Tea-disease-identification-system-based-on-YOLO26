@echo off
set HOST=0.0.0.0
set PORT=5000
set YOLO_DEVICE=cpu
python -m pip install -r requirements.txt
python app.py
