@echo off
echo ================================================
echo Python Project Launcher
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please run install.bat first to set up the environment
    pause
    exit /b 1
)

REM Change to the script directory (in case the script is run from elsewhere)
cd /d "%~dp0"

REM Check if main.py exists (replace with your actual main file name)
if not exist "main.py" (
    echo ERROR: main.py not found in current directory
    echo Please make sure your main Python file is named 'main.py'
    echo Or modify this script to use the correct filename
    echo.
    echo Current directory: %CD%
    echo Available Python files:
    dir *.py /b 2>nul
    if %errorlevel% neq 0 (
        echo No Python files found in current directory
    )
    echo.
    pause
    exit /b 1
)

echo Starting Python program...
echo.

REM Run the main Python script
python main.py

REM Check if the program exited with an error
if %errorlevel% neq 0 (
    echo.
    echo ================================================
    echo Program exited with error code: %errorlevel%
    echo ================================================
) else (
    echo.
    echo ================================================
    echo Program completed successfully
    echo ================================================
)

echo.
pause
