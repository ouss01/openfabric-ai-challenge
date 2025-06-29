@echo off
REM 🚀 AI Creative Pipeline - Windows Start Script
REM This script sets up the environment and starts the AI Creative Pipeline

echo 🚀 Starting AI Creative Pipeline...
echo ==================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if Poetry is installed
poetry --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Poetry is not installed. Please install Poetry first.
    echo Install with: curl -sSL https://install.python-poetry.org ^| python3 -
    pause
    exit /b 1
)

REM Install dependencies
echo 📦 Installing dependencies...
poetry install

REM Create output directories
echo 📁 Creating output directories...
if not exist "outputs\images" mkdir "outputs\images"
if not exist "outputs\models" mkdir "outputs\models"

REM Check for local LLM model (optional)
if exist "llama-2-7b-chat.gguf" (
    echo 🧠 Local LLM model found - enhanced prompt expansion will be available
) else if exist "..\llama-2-7b-chat.gguf" (
    echo 🧠 Local LLM model found - enhanced prompt expansion will be available
) else (
    echo ℹ️  No local LLM model found - using template-based prompt expansion
    echo    To enable LLM enhancement, download a GGUF model and place it in the project root
)

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%
set LOG_LEVEL=INFO

REM Load Hugging Face API token if config file exists
if exist "config.env" (
    echo 🔑 Loading Hugging Face API token from config.env...
    for /f "tokens=1,2 delims==" %%a in (config.env) do set %%a=%%b
    echo ✅ Hugging Face API token loaded
) else (
    echo ⚠️  No config.env file found - Hugging Face integration will use free tier
)

echo ✅ Environment setup complete!
echo.
echo 🎯 Next steps:
echo 1. The FastAPI server will start automatically
echo 2. Access the Swagger UI at: http://localhost:8888/docs
echo 3. Configure your Openfabric app IDs in the Swagger UI
echo 4. Start creating amazing content!
echo.
echo 📝 Example app IDs:
echo    Text-to-Image: f0997a01-d6d3-a5fe-53d8-561300318557
echo    Image-to-3D: 69543f29-4d41-4afc-7f29-3d51591f11eb
echo.
echo 🚀 Starting FastAPI server...

REM Start the FastAPI server
poetry run uvicorn server:app --host 0.0.0.0 --port 8888 --reload

pause 