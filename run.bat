@echo off
title Kenali Sampah Elektronikmu - E-Waste Classifier
echo ============================================================
echo   Memulai Server Aplikasi Kenali Sampah Elektronikmu
echo ============================================================
echo.
cd /d "%~dp0"
echo Membuka server di http://127.0.0.1:8000 ...
echo Tekan CTRL+C untuk menghentikan server.
echo.
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause

