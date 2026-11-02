@echo off
setlocal
cd /d "%~dp0"
title 3G - Anh - Video - Webcam
if not exist ".venv\Scripts\python.exe" (
  echo [3G] Chua tim thay .venv\Scripts\python.exe tai thu muc nay.
  echo [3G] Hay chep cac file vao D:\TGMT, GIU NGUYEN .venv va .venv_yolo.
  pause
  exit /b 1
)
echo ================================================
echo            3G - BAN CHINH (AI THAT)
echo ================================================
echo Dia chi: http://127.0.0.1:7866
echo Khong mo cung luc 2 ban tren cong 7866.
echo Dung: Ctrl+C tai cua so nay.
echo.
".venv\Scripts\python.exe" -u app.py
if errorlevel 1 (
  echo.
  echo [3G] Co loi. Doc thong bao phia tren.
  pause
)
endlocal
