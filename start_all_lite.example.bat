@echo off
REM KEEP THIS FILE PURE ASCII: cmd decodes batch files in the console codepage; non-ASCII bytes desync the parser into executing comment fragments (diagnosed 08-28).
REM ============================================================================
REM Lite launcher: OLV server + Discord bot only -- no TTS, no frontend.
REM For lightweight operation where the heavy processes are dead weight,
REM e.g. the scheduled wake-for-alarm task that boots the stack before an
REM alarm fires (music playback runs in-process; no TTS/frontend needed).
REM
REM Uses Windows Terminal tabs. wt is an MSIX app and can fail to activate in
REM a LOCKED session -- if your machine shows a lock screen on timer wake
REM (power setting CONSOLELOCK=1, the default), either disable it
REM (powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_NONE CONSOLELOCK 0) or swap
REM the wt line below for plain `start "..." cmd /k ...` windows. A locked
REM session also mutes audio started inside it -- a wake-up alarm needs the
REM lock screen gone anyway.
REM
REM SETUP: copy this file to start_all_lite.bat (gitignored) and set CONDA_ENV.
REM   copy start_all_lite.example.bat start_all_lite.bat
REM Keep start_all_lite.bat in the project root so %~dp0 resolves to the OLV dir.
REM ============================================================================

REM === EDIT THESE ===
set "CONDA_ENV="
REM ==================

REM %~dp0 ends with a backslash. Inside -d "...\" that \" ESCAPES the closing
REM quote and wt swallows the rest of the line into the directory argument
REM (error 0x8007010b). The strip guard below exists for exactly this -- keep it.
set "OLV_DIR=%~dp0"
if "%OLV_DIR:~-1%"=="\" set "OLV_DIR=%OLV_DIR:~0,-1%"
cd /d "%OLV_DIR%"
wt new-tab --title OLV -d "%OLV_DIR%" cmd /k "call conda activate %CONDA_ENV% && python run_server.py" ; new-tab --title Discord -d "%OLV_DIR%" cmd /k "timeout /t 15 && call conda activate %CONDA_ENV% && python scripts\run_discord_bot.py"
