@echo off
REM ============================================================================
REM RESUME launcher: same as start_all, but OLV continues the PREVIOUS
REM conversation instead of starting a new session.
REM
REM Use this for frequent test restarts: it skips the diary/fact backfill of the
REM session you were just in, and keeps the conversation context — so the chat
REM picks up where it left off (web + Discord both adopt the continued session).
REM
REM SETUP: copy this file to start_all_resume.bat (gitignored) and edit the vars.
REM   copy start_all_resume.example.bat start_all_resume.bat
REM Keep it in the project root so %~dp0 resolves to the OLV dir.
REM ============================================================================

REM === EDIT THESE (same as start_all.bat) ===
set "CONDA_ENV="
set "TTS_DIR=D:\path\to\GPT-SoVITS"
REM ==========================================

REM OLV project dir = wherever this script lives.
set "OLV_DIR=%~dp0"
if "%OLV_DIR:~-1%"=="\" set "OLV_DIR=%OLV_DIR:~0,-1%"

REM Only OLV needs the resume flag (--resume); Discord + TTS launch normally and
REM adopt the continued session automatically when they connect.
wt new-tab --title OLV -d "%OLV_DIR%" cmd /k "call conda activate %CONDA_ENV% && python run_server.py --resume" ; new-tab --title Discord -d "%OLV_DIR%" cmd /k "timeout /t 5 && call conda activate %CONDA_ENV% && python scripts\run_discord_bot.py" ; new-tab --title TTS -d "%TTS_DIR%" cmd /k "runtime\python.exe api_v2.py"
