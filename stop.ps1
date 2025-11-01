# === stop.ps1 (UTF-8) ===
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

function Kill-ByPort([int]$port){
  $pids = Get-PidsByPort -port $port
  if(-not $pids -or $pids.Count -eq 0){ Write-Output "Port $port not in use."; return }
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
    foreach($p in $procs){
      try{
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Output ("Killed process {0} (PID {1})" -f $p.Name, $p.ProcessId)
      }catch{}
    }
  }catch{}
}

# -------- MAIN --------
$Root = $PSScriptRoot
Set-Location $Root

Write-Host ""
Write-Host "----------------------------------------------" -ForegroundColor White
Write-Host "STOPPING SERVICES: AI Smart Credit Suite" -ForegroundColor White
Write-Host "----------------------------------------------" -ForegroundColor White

# Lire le port UI depuis .env (fallback 8501)
$UiPort = 8501
$EnvPath = Join-Path $Root ".env"
if(Test-Path $EnvPath){
  $line = (Get-Content $EnvPath) | Where-Object { $_ -match '^\s*WEB_PORT\s*=' }
  if($line){ $UiPort = [int](($line -split '=',2)[1].Trim()) }
}

# 1) Streamlit
LogInfo "Stopping Streamlit..."
Kill-ByPort $UiPort
Kill-ByPort 8501
Kill-ByPort 8502
Kill-ByName "streamlit"
Kill-ByName "python.*streamlit_app.py"

# 2) Backend (Uvicorn/FastAPI)
LogInfo "Stopping backend (Uvicorn/FastAPI)..."
Kill-ByPort 18000
Kill-ByName "uvicorn"
Kill-ByName "python.*server:app"

# 3) Ollama (si actif)
LogInfo "Stopping Ollama (if running)..."
Kill-ByPort 11434
Kill-ByName "ollama"

# 4) Résumé
Write-Host ""
LogInfo "Port check ($UiPort, 8501, 8502, 18000, 11434):"
foreach($p in @($UiPort, 8501, 8502, 18000, 11434)){
  $pids = Get-PidsByPort -port $p
  if(-not $pids -or $pids.Count -eq 0){ Write-Host ("Port {0} is free." -f $p) -ForegroundColor Green }
  else{ Write-Host ("Port {0} still used by: {1}" -f $p, ($pids -join ", ")) -ForegroundColor Yellow }
}

Write-Host ""
LogOk "All services stopped. To start again: .\start.ps1"
