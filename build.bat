@echo off
REM ============================================================
REM  MedVision AI — Build Script (VENV VERSION)
REM  Phải chắc chắn có chữ (venv) trước khi chạy file này!
REM ============================================================

echo.
echo ===================================
echo   MedVision AI - Build to .exe
echo ===================================
echo.

REM -- Chuyen doi icon PNG -> ICO --
echo [1/3] Chuyen doi icon PNG sang ICO...
python convert_icon.py

REM -- Xoa cache build cu --
echo [2/3] Xoa cache build cu...
if exist build rmdir /s /q build
if exist dist\MedVisionAI rmdir /s /q dist\MedVisionAI

REM -- Chay PyInstaller --
echo [3/3] Dang build... (co the mat 3-10 phut)
echo.
pyinstaller MedVision.spec --noconfirm

REM -- Kiem tra ket qua --
echo.
if exist "dist\MedVisionAI\MedVisionAI.exe" (
    echo [4/4] BUILD THANH CONG!
    echo.
    echo Thu muc chua app: dist\MedVisionAI\
    echo.
    explorer dist\MedVisionAI
) else (
    echo [4/4] BUILD THAT BAI! Xem log de biet chi tiet.
)

echo.
pause