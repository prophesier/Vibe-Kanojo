# Wake-for-alarm task manager. Single implementation for both entry points:
# the Discord /wake command shells out to this script, and it can be run
# by hand in any terminal when the server stack is down.
#
#   wake_task.ps1 arm [-LeadMinutes 10]   register the one-shot wake task for
#                                         the earliest pending wake=true alarm
#                                         firing within 24h
#   wake_task.ps1 status                  show the registered task, if any
#   wake_task.ps1 cancel                  delete the task
#   wake_task.ps1 run ...                 (internal) the wake-time action the
#                                         scheduled task executes
#
# Design (settled 2026-08-08): single fixed task name (one slot, re-arm
# overwrites, self-deletes on firing -> zero residue); schtasks CLI cannot
# set WakeToRun so registration goes through Register-ScheduledTask; the run
# action holds SetThreadExecutionState(ES_SYSTEM_REQUIRED) from wake until the
# FIRST physical input (GetLastInputInfo), because timer wakes are unattended
# and re-sleep after ~120s otherwise.
#
# Hard-won constraints (first flight 08-09 04:50 failed on the third one):
# - Keep this file pure ASCII: PowerShell 5.1 reads BOM-less files as ANSI.
# - The ES_* flag math lives INSIDE the C# class: a PS 5.1 hex literal like
#   0x80000001 parses as a NEGATIVE Int32 and dies converting to uint at the
#   P/Invoke boundary ("-2147483647 too small for UInt32"), killing the whole
#   run under ErrorActionPreference=Stop before the hold was ever acquired.
# - Everything in `run` is try/catch'd into the run log - the scheduled task
#   has no console and no operational log by default, so an unlogged death is
#   invisible until the alarm fails to ring.
# - AUDIO GATE (case closed 08-11): after a timer wake, Windows renders NO
#   audio until a keyboard/mouse input arrives (mouse MOVEMENT is not enough
#   - click or key only). The 08-11 05:00 alarm played ffplay for 15 minutes
#   into a live endpoint in total silence. Fix: inject one synthetic F15
#   keystroke (a key that exists on no physical keyboard - nothing reacts to
#   it) right after booting the stack. That injection also counts as "input",
#   so the hold-until-input wait compares against a BASELINE last-input tick
#   taken after the nudge, releasing only when the tick changes again (a real
#   human). The old IdleMs<15s check would have released on our own nudge.
param(
    [Parameter(Position = 0)]
    [ValidateSet("arm", "status", "cancel", "run")]
    [string]$Action = "status",
    [int]$LeadMinutes = 10,
    # Default: newest alarms.json under the repo's chat_history (script lives
    # in <repo>\scripts). Override for tests or exotic layouts.
    [string]$AlarmsPath = "",
    # Repo-root personal copy, gitignored - same convention as restart.bat
    # (copy start_all_lite.example.bat there and edit). Empty = resolved
    # below: $PSScriptRoot is NOT populated yet while param defaults are
    # evaluated in PS 5.1, so the Join-Path cannot live here.
    [string]$StartScript = "",
    # Internal, passed by the registered task so `run` can log context.
    [string]$AlarmTimeLocal = "",
    # Test hook: cap the hold-until-input wait (0 = wait for input, no cap).
    [int]$MaxHoldSeconds = 0
)

$ErrorActionPreference = "Stop"
# Emit UTF-8 so the Discord bot's pipe capture decodes cleanly.
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}

if (-not $StartScript) {
    $StartScript = Join-Path $PSScriptRoot "..\start_all_lite.bat"
}

$TaskName = "VibeKanojo-Wake"
$RunLog = Join-Path $env:TEMP "vibekanojo_wake_run.log"

function Write-RunLog([string]$line) {
    "$(Get-Date -Format 'HH:mm:ss') $line" | Out-File -Encoding utf8 -Append $RunLog
}

function Resolve-AlarmsPath {
    if ($AlarmsPath) { return $AlarmsPath }
    $candidates = Get-ChildItem -Path (Join-Path $PSScriptRoot "..\chat_history\*\alarms.json") -ErrorAction SilentlyContinue
    if (-not $candidates) { throw "no alarms.json found under chat_history (use -AlarmsPath)" }
    return ($candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}

function Get-EligibleAlarm {
    $path = Resolve-AlarmsPath
    $now = Get-Date
    $best = $null
    $eligible = 0
    $records = Get-Content -Raw -Encoding UTF8 $path | ConvertFrom-Json
    foreach ($a in $records) {
        if ($a.status -ne "pending") { continue }
        if (-not $a.wake) { continue }
        try {
            $fire = [DateTimeOffset]::Parse(
                $a.fire_at_utc, [Globalization.CultureInfo]::InvariantCulture
            ).ToLocalTime().DateTime
        } catch { continue }
        if ($fire -le $now) { continue }
        if (($fire - $now).TotalHours -gt 24) { continue }
        $eligible += 1
        if (-not $best -or $fire -lt $best.fire) {
            $best = @{ fire = $fire; id = $a.id; note = $a.note }
        }
    }
    if ($best) { $best.eligible = $eligible }
    return $best
}

function Invoke-Arm {
    $alarm = Get-EligibleAlarm
    if (-not $alarm) {
        Write-Output "arm: no pending wake=true alarm within the next 24h - nothing registered."
        exit 1
    }
    if (-not (Test-Path $StartScript)) {
        Write-Output "arm: start script not found: $StartScript"
        Write-Output "     copy start_all_lite.example.bat to the repo root as start_all_lite.bat first."
        exit 1
    }
    $wakeAt = $alarm.fire.AddMinutes(-$LeadMinutes)
    $floor = (Get-Date).AddMinutes(1)
    if ($wakeAt -lt $floor) { $wakeAt = $floor }
    $fireStr = $alarm.fire.ToString("yyyy-MM-dd HH:mm:ss")
    $wakeStr = $wakeAt.ToString("yyyy-MM-dd HH:mm:ss")

    $argStr = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" run " +
        "-AlarmTimeLocal `"$fireStr`" -StartScript `"$StartScript`""
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argStr
    $trigger = New-ScheduledTaskTrigger -Once -At $wakeAt
    $settings = New-ScheduledTaskSettingsSet -WakeToRun `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours 26)
    $desc = "alarm $fireStr id=$($alarm.id) note=$($alarm.note) | wake at $wakeStr (lead ${LeadMinutes}m)"
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Description $desc -Force | Out-Null
    Write-Output "arm: OK"
    Write-Output "  alarm : $fireStr  (id $($alarm.id))  note: $($alarm.note)"
    Write-Output "  wake  : $wakeStr  (lead $LeadMinutes min)"
    if ($alarm.eligible -gt 1) {
        Write-Output "  note  : $($alarm.eligible) eligible wake alarms in 24h; armed the earliest."
    }
    Write-Output "  task  : $TaskName (single slot; re-arm overwrites, self-deletes on firing)"
}

function Invoke-Status {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Output "status: no wake task registered."
        return
    }
    $info = $task | Get-ScheduledTaskInfo
    Write-Output "status: task '$TaskName' registered  (state: $($task.State))"
    Write-Output "  next run : $($info.NextRunTime)"
    Write-Output "  detail   : $($task.Description)"
}

function Invoke-Cancel {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Output "cancel: no wake task registered - nothing to do."
        return
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "cancel: task '$TaskName' deleted."
}

function Invoke-Run {
    "=== wake run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') alarm=$AlarmTimeLocal ===" |
        Out-File -Encoding utf8 -Append $RunLog
    try {
        # 1. Self-delete FIRST - the trigger already fired, so the slot is
        #    spent no matter what happens below (first flight died mid-run
        #    and left the task behind).
        try {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
            Write-RunLog "task self-deleted"
        } catch {
            Write-RunLog "self-delete failed: $($_.Exception.Message)"
        }

        # ES_* flags stay inside C# - see header. PS hex literals are Int32.
        Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WakeHold {
    [DllImport("kernel32.dll")]
    static extern uint SetThreadExecutionState(uint esFlags);
    const uint ES_CONTINUOUS = 0x80000000;
    const uint ES_SYSTEM_REQUIRED = 0x00000001;
    public static bool Hold() {
        return SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0;
    }
    public static bool Release() {
        return SetThreadExecutionState(ES_CONTINUOUS) != 0;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
    [DllImport("user32.dll")]
    public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
    public static uint LastInputTick() {
        LASTINPUTINFO lii = new LASTINPUTINFO();
        lii.cbSize = (uint)Marshal.SizeOf(typeof(LASTINPUTINFO));
        GetLastInputInfo(ref lii);
        return lii.dwTime;
    }
    [DllImport("user32.dll")]
    static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
    const byte VK_F15 = 0x7E;
    const uint KEYEVENTF_KEYUP = 0x0002;
    public static void Nudge() {
        keybd_event(VK_F15, 0, 0, UIntPtr.Zero);
        keybd_event(VK_F15, 0, KEYEVENTF_KEYUP, UIntPtr.Zero);
    }
}
"@
        Write-RunLog "types compiled"

        # 2. Hold the machine awake: timer wakes are unattended and re-sleep
        #    after ~120s of no input (UNATTENDSLEEP) without this.
        if ([WakeHold]::Hold()) {
            Write-RunLog "hold acquired"
        } else {
            Write-RunLog "HOLD FAILED - SetThreadExecutionState returned 0"
        }

        # 3. Boot the stack (lite: OLV + Discord bot) - unless it is already
        #    up (armed but never slept): a second bot on the same token would
        #    double-reply on Discord.
        $alive = $false
        $pidFile = Join-Path $PSScriptRoot "..\pids\olv.pid"
        if (Test-Path $pidFile) {
            try {
                $olvPid = [int](Get-Content $pidFile -First 1).Trim()
                $p = Get-Process -Id $olvPid -ErrorAction SilentlyContinue
                # Name check guards against a stale pid file (hard kill leaves
                # it) whose PID got recycled by an unrelated process - a false
                # "alive" would silently skip the start and mute the alarm.
                if ($p -and $p.ProcessName -match "python") { $alive = $true }
            } catch {}
        }
        if ($alive) {
            Write-RunLog "stack already running (olv.pid alive) - skipping start"
        } elseif (Test-Path $StartScript) {
            Start-Process cmd -ArgumentList "/c", "`"$StartScript`""
            Write-RunLog "started $StartScript"
        } else {
            Write-RunLog "START SCRIPT MISSING: $StartScript"
        }

        # 3.5 Open the audio gate: no keyboard/mouse input after a timer wake
        #     means NO sound at all (see header). F15 is a no-op key to every
        #     application but a real input to the system.
        try {
            [WakeHold]::Nudge()
            Write-RunLog "F15 nudge injected (audio gate)"
        } catch {
            Write-RunLog "NUDGE FAILED: $($_.Exception.Message)"
        }
        Start-Sleep -Milliseconds 500

        # 4. Hold until the FIRST physical input AFTER our nudge. Baseline
        #    comparison, not idle time: the nudge itself is "input", so an
        #    idle-based check would release immediately. No input ever -> the
        #    26h task ExecutionTimeLimit is the outer bound (or
        #    -MaxHoldSeconds when testing).
        $baseline = [WakeHold]::LastInputTick()
        $deadline = $null
        if ($MaxHoldSeconds -gt 0) { $deadline = (Get-Date).AddSeconds($MaxHoldSeconds) }
        while ([WakeHold]::LastInputTick() -eq $baseline) {
            if ($deadline -and (Get-Date) -gt $deadline) {
                Write-RunLog "max hold ($MaxHoldSeconds s) reached"
                break
            }
            Start-Sleep -Seconds 5
        }
        Write-RunLog "hold released (input after nudge, or cap reached)"
    } catch {
        Write-RunLog "RUN DIED: $($_.Exception.Message)"
        Write-RunLog ($_ | Out-String)
    } finally {
        try { [WakeHold]::Release() | Out-Null } catch {}
    }
}

switch ($Action) {
    "arm" { Invoke-Arm }
    "status" { Invoke-Status }
    "cancel" { Invoke-Cancel }
    "run" { Invoke-Run }
}
