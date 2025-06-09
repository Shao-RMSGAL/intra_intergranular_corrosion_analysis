@echo off
echo ================================================
echo Python Project Installation Script
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

echo Pip found:
pip --version
echo.

echo Installing required packages...
echo ================================================

REM Note: tkinter comes with Python by default, so we don't need to install it
echo Installing numba...
pip install numba
if %errorlevel% neq 0 (
    echo ERROR: Failed to install numba
    pause
    exit /b 1
)

echo Installing opencv-python...
pip install opencv-python
if %errorlevel% neq 0 (
    echo ERROR: Failed to install opencv-python
    pause
    exit /b 1
)

echo Installing matplotlib...
pip install matplotlib
if %errorlevel% neq 0 (
    echo ERROR: Failed to install matplotlib
    pause
    exit /b 1
)

echo Installing pandas...
pip install pandas
if %errorlevel% neq 0 (
    echo ERROR: Failed to install pandas
    pause
    exit /b 1
)

echo Installing pillow...
pip install pillow
if %errorlevel% neq 0 (
    echo ERROR: Failed to install pillow
    pause
    exit /b 1
)

echo.
echo ================================================
echo Installation completed successfully!
echo ================================================
echo.
echo All required packages have been installed:
echo - tkinter (built-in with Python)
echo - numba
echo - opencv-python
echo - matplotlib
echo - pandas
echo - pillow
echo.
echo You can now run your program using run.bat
echo.
pause
