@echo off
setlocal

echo Installing Microsoft Visual C++ Redistributable x64...
powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%TEMP%\vc_redist.x64.exe'"
"%TEMP%\vc_redist.x64.exe" /install /quiet /norestart

echo Reinstalling Python dependencies...
python -m pip uninstall -y onnxruntime opencv-python opencv-python-headless numpy
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Testing onnxruntime import...
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"

echo Done. If the import still fails, restart Windows Server and run this test again:
echo python -c "import onnxruntime as ort; print(ort.__version__)"
