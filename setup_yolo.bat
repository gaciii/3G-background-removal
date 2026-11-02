@echo off
setlocal
cd /d "%~dp0"
title Cai AI YOLO CPU (mot lan)
echo ==========================================
echo    CAI AI YOLO CPU MOT LAN
echo ==========================================
echo.
if not exist ".venv\Scripts\python.exe" (
  echo [LOI] Khong tim thay .venv\Scripts\python.exe
  echo Hay dat setup_yolo.bat trong thu muc D:\TGMT.
  goto error
)
if not exist "video_auto_yolo.py" (
  echo [LOI] Chua chep video_auto_yolo.py vao D:\TGMT.
  goto error
)
if not exist ".venv_yolo\Scripts\python.exe" (
  echo [1/4] Tao Python rieng cho YOLO - khong sua .venv hien tai...
  ".venv\Scripts\python.exe" -m venv ".venv_yolo"
  if errorlevel 1 goto error
)
echo.
echo [2/4] Nang cap pip trong .venv_yolo...
".venv_yolo\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error
echo.
echo [3/4] Cai PyTorch CPU (download co the rat lon)...
".venv_yolo\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto error
echo.
echo [4/4] Cai YOLO va FFmpeg trong .venv_yolo...
".venv_yolo\Scripts\python.exe" -m pip install "ultralytics>=8.3.0" imageio-ffmpeg==0.6.0
if errorlevel 1 goto error
echo.
echo [OK] Cai dat xong. Dang kiem tra...
".venv_yolo\Scripts\python.exe" -c "import torch, ultralytics, imageio_ffmpeg; print('Torch:',torch.__version__,'- Ultralytics:',ultralytics.__version__);print('CPU san sang')"
if errorlevel 1 goto error
echo.
echo HOAN THANH. Mo AETM bang bieu tuong MO AETM tren Desktop.
pause
exit /b 0
:error
echo.
echo [THAT BAI] 
pause
exit /b 1
