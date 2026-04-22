@echo off
REM ============================================================
REM  Tạo shortcut MedVision AI ra Desktop
REM  Chạy file này SAU KHI build thành công
REM ============================================================

echo Dang tao shortcut ra Desktop...

REM Lấy đường dẫn tuyệt đối của thư mục hiện tại
set PROJECT_DIR=%~dp0
set EXE_PATH=%PROJECT_DIR%dist\MedVisionAI\MedVisionAI.exe
set ICO_PATH=%PROJECT_DIR%assets\logoNST.ico
set DESKTOP=%USERPROFILE%\Desktop
set SHORTCUT=%DESKTOP%\MedVision AI.lnk

REM Kiểm tra file exe tồn tại
if not exist "%EXE_PATH%" (
    echo [LOI] Chua tim thay file exe tai:
    echo       %EXE_PATH%
    echo.
    echo Hay chay build.bat truoc!
    pause
    exit /b 1
)

REM Tạo shortcut bằng PowerShell
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%EXE_PATH%'; $s.WorkingDirectory = '%PROJECT_DIR%dist\MedVisionAI'; $s.IconLocation = '%ICO_PATH%'; $s.Description = 'MedVision AI - He thong phan tich hinh anh y te'; $s.WindowStyle = 1; $s.Save()"

if exist "%SHORTCUT%" (
    echo.
    echo THANH CONG! Shortcut da duoc tao tai:
    echo %SHORTCUT%
    echo.
    echo Click doi vao shortcut tren Desktop de chay MedVision AI!
) else (
    echo.
    echo That bai khi tao shortcut.
    echo Ban co the tao thu cong:
    echo   1. Vao thu muc: dist\MedVisionAI\
    echo   2. Chuot phai vao MedVisionAI.exe
    echo   3. Chon "Send to" - "Desktop (create shortcut)"
)

echo.
pause
