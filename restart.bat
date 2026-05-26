@echo off
REM ============================================================================
REM Auto-restart script triggered by Discord bot /restart command.
REM
REM Reads PID files from pids/, kills OLV + Discord bot, pulls the latest
REM code from the current branch, then re-launches both services in new
REM Windows Terminal tabs. TTS (which runs out of a separate project) is
REM intentionally NOT touched.
REM
REM EDIT THE LINE BELOW to match your conda environment name.
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM === EDIT THIS ===
set "CONDA_ENV=openllmvtuber"
REM =================

echo Waiting for services to exit cleanly...
timeout /t 3 /nobreak >nul

if exist pids\olv.pid (
    set /p OLV_PID=<pids\olv.pid
    if defined OLV_PID (
        echo Killing OLV PID !OLV_PID!...
        taskkill /F /PID !OLV_PID! 2>nul
    )
    del /f /q pids\olv.pid 2>nul
)

if exist pids\discord.pid (
    set /p BOT_PID=<pids\discord.pid
    if defined BOT_PID (
        echo Killing Discord bot PID !BOT_PID!...
        taskkill /F /PID !BOT_PID! 2>nul
    )
    del /f /q pids\discord.pid 2>nul
)

echo Pulling latest code...
REM -c core.editor=true ensures git never blocks waiting for a merge commit
REM message (replaces VS Code / Vim with a no-op).
REM --no-edit takes the default merge commit message silently if a merge happens.
git -c core.editor=true pull --no-edit
if errorlevel 1 (
    echo.
    echo *** Git pull failed. Aborting any in-progress merge. ***
    git merge --abort 2>nul
    echo *** Restarting services with the existing code anyway. ***
    echo.
)

echo Re-launching OLV and Discord bot in new wt tabs...
wt new-tab --title OLV -d "%~dp0" cmd /k "call conda activate %CONDA_ENV% && python run_server.py" ; new-tab --title Discord -d "%~dp0" cmd /k "timeout /t 5 && call conda activate %CONDA_ENV% && python scripts\run_discord_bot.py"

endlocal
exit /b 0
