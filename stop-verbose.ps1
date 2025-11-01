# === stop.ps1 (ASCII, UTF-8, VERBOSE) ===
$ErrorActionPreference = "SilentlyContinue"

function LogInfo($m){ Write-Host $m -ForegroundColor Cyan }
function LogOk($m){ Write-Host $m -ForegroundColor Green }
function LogWarn($m){ Write-Host $m -ForegroundColor Yellow }
function LogErr($m){ Write-Host $m -ForegroundColor Red }

function Get-PidsByPort([int]$port){
  $pids = @()
  try{
    $lines = netstat -ano -p tcp | Select-String (":$port\s")
    foreach($ln in $lines){
      $txt = $ln.ToString().Trim()
      if(-not $txt){ continue }
      $parts = $txt -split '\s+'
      if($parts.Length -ge 5){
        $pid = $parts[-1]
        if($pid -match '^\d+$' -and -not ($pids -contains [int]$pid)){
          $pids += [int]$pid
        }
      }
    }
  }catch{}
  return $pids
}

function Describe-PID([int]$pid){
  try{
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$pid"
    if($null -eq $p){ return "PID $pid (no details)" }
    $name = $p.Name
    $cmd  = $p.CommandLine
    if($cmd -and $cmd.Length -gt 160){ $cmd = $cmd.Substring(0,160) + " ..." }
    return "PID $pid  Name=$name  Cmd=$cmd"
  }catch{
    return "PID $pid (no details)"
  }
}

function Kill-ByPort([int]$port){
  $pids = Get-PidsByPort -port $port
  if(-not $pids -or $pids.Count -eq 0){
    Write-Output ("Port {0} not in use." -f $port)
    return
  }
  Write-Output ("Found {0} PID(s) on port {1}:" -f $pids.Count, $port)
  foreach($pid in $pids){
    Write-Output ("  - " + (Describe-PID $pid))
  }
  foreach($pid in $pids){
    try{
      taskkill /PID $pid /F | Out-Null
      Write-Output ("Killed PID {0} (port {1})" -f $pid, $port)
    }catch{
      Write-Output ("Could not kill PID {0} (port {1}): {2}" -f $pid, $port, $_.Exception.Message)
    }
  }
}

function Kill-ByName([string]$nameLike){
  try{
    $procs = Get-CimInstance Win32_Process | Where-Object {
      $_.Name -like $nameLike -or $_.CommandLine -match $nameLike
    }
    if(-not $procs){ return }
    Write-Output ("Found {0} process(es) by name '{1}':" -f $procs.Count, $nameLike)
    foreach($p in $procs){
      $cmd = $p.CommandLine
      if($cmd -and $cmd.Length -gt 160){ $cmd = $cmd.Substring(0,160) + " ..." }
      Write-Output ("  - PID {0}  Name={1}  Cmd={2}" -f $p.ProcessId, $p.Name, $cmd)
    }
    foreach($p in $procs){
      try{
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Output ("Killed process {0} (PID {1})" -f $p.Name, $p.ProcessId)
      }catch{
        Write-Output ("Could not kill PID {0}: {1}" -f $p.ProcessId, $_.Exception.Message)
      }
    }
  }catch{}
}

function Recheck-Port([int]$port){
  $pids = Get-PidsByPort -port $port
  if(-not $pids -or $pids.Count -eq 0){
    Write-Host ("-> Port {0} is free." -f $port) -ForegroundColor Green
  }else{
    Write-Host ("-> Port {0} still used by: {1}" -f $port, ($pids -join ", ")) -ForegroundColor Yellow
    foreach($pid in $pids){ Write-Host ("   " + (Describe-PID $pid)) -ForegroundColor Yellow }
  }
}

# ------------------------------
# MAIN
# ------------------------------
$Root = $PSScriptRoot
Set-Location $Root

Write-Host ""
Write-Host "----------------------------------------------" -ForegroundColor White
Write-Host "STOPPING SERVICES (VERBOSE): AI Smart Credit   " -ForegroundColor White
Write-Host "----------------------------------------------" -ForegroundColor White

# 1) Streamlit (UI) - port 8501
LogInfo "Stopping Streamlit (port 8501)..."
Kill-ByPort 8501
# bonus: if streamlit moved port or zombie process remains
Kill-ByName "streamlit"
Kill-ByName "python.*streamlit_app.py"
Recheck-Port 8501
Write-Host ""

# 2) Backend (Uvicorn/FastAPI) - port 18000
LogInfo "Stopping backend Uvicorn/FastAPI (port 18000)..."
Kill-ByPort 18000
Kill-ByName "uvicorn"
Kill-ByName "python.*server:app"
Recheck-Port 18000
Write-Host ""

# 3) Ollama (LLM) - port 11434
LogInfo "Stopping Ollama (port 11434)..."
Kill-ByPort 11434
Kill-ByName "ollama"
Recheck-Port 11434
Write-Host ""

# 4) Summary
Write-Host "Summary (final check):" -ForegroundColor White
Recheck-Port 8501
Recheck-Port 18000
Recheck-Port 11434

Write-Host ""
LogOk "All services stopped. To start again: .\start.ps1"
