@echo off
REM ============================================================================
REM One-click launcher: opens OLV server, Discord bot, and GPT-SoVITS TTS,
REM each in its own Windows Terminal tab.
REM
REM SETUP: copy this file to start_all.bat (gitignored) and edit the vars below.
REM   copy start_all.example.bat start_all.bat
REM Keep start_all.bat in the project root so %~dp0 resolves to the OLV dir.
REM ============================================================================

REM === EDIT THESE ===
set "CONDA_ENV="
REM Path to your local GPT-SoVITS project (the TTS backend). If you don't use
REM GPT-SoVITS, delete the " ; new-tab --title TTS ..." part at the end.
set "TTS_DIR=D:\path\to\GPT-SoVITS"
REM ==================

REM OLV project dir = wherever this script lives.
set "OLV_DIR=%~dp0"
if "%OLV_DIR:~-1%"=="\" set "OLV_DIR=%OLV_DIR:~0,-1%"

wt new-tab --title OLV -d "%OLV_DIR%" cmd /k "call conda activate %CONDA_ENV% && python run_server.py" ; new-tab --title Discord -d "%OLV_DIR%" cmd /k "timeout /t 5 && call conda activate %CONDA_ENV% && python scripts\run_discord_bot.py" ; new-tab --title TTS -d "%TTS_DIR%" cmd /k "runtime\python.exe api_v2.py"
