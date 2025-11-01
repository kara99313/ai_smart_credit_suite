# === restart.ps1 (ASCII, UTF-8, intelligent + notification) ===
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

function Port-State([int]$port){
  $p = Get-PidsByPort -port $port
  if(-not $p -or $p.Count -eq 0){ return "free" } else { return "used: " + ($p -join ",") }
}

function Test-Health($url, $retries = 60, $delaySec = 0.5){
  for($i=0; $i -lt $retries; $i++){
    try{
      $r = Invoke-WebRequest -Uri $url -TimeoutSec 2
      if($r.StatusCode -ge 200 -and $r.StatusCode -lt 300){ return $true }
    } catch { Start-Sleep -Seconds $delaySec }
  }
  return $false
}

function Time-Action([scriptblock]$action){
  $t0 = Get-Date
  & $action
  $t1 = Get-Date
  return (New-TimeSpan -Start $t0 -End $t1).TotalSeconds
}

function Show-Notification($title, $message){
  # 1) Try BurntToast (modern Windows toast)
  try{
    if (Get-Module -ListAvailable -Name BurntToast) {
      Import-Module BurntToast -ErrorAction SilentlyContinue | Out-Null
      New-BurntToastNotification -Text $title, $message | Out-Null
      return
    }
  }catch{}
  # 2) Try .NET MessageBox
  try{
    Add-Type -AssemblyName System.Windows.Forms | Out-Null
    [System.Windows.Forms.MessageBox]::Show($message, $title, 'OK', 'Information') | Out-Null
    return
  }catch{}
  # 3) Fallback: console bell + line
  Write-Host "`a"
  Write-Host ("[NOTIFY] {0} - {1}" -f $title, $message) -ForegroundColor Green
}

# ------------------------------
# MAIN
# ------------------------------
$Root = $PSScriptRoot
Set-Location $Root

Write-Host ""
Write-Host "----------------------------------------------" -ForegroundColor White
Write-Host "RESTARTING (intelligent): AI Smart Credit Suite" -ForegroundColor White
Write-Host "----------------------------------------------" -ForegroundColor White
Write-Host ""

# Load .env (for URLs/ports)
$EnvPath = Join-Path $Root ".env"
if(Test-Path $EnvPath){
  Get-Content $EnvPath | Where-Object { $_ -match '^\s*[^#].+=.+$' } | ForEach-Object {
    $kv = $_ -split '=', 2
    if($kv.Length -eq 2){
      $k = $kv[0].Trim()
      $v = $kv[1].Trim()
      [Environment]::SetEnvironmentVariable($k, $v, "Process")
    }
  }
}

# Resolve endpoints
$HostIP   = $env:API_HOST;  if(-not $HostIP){ $HostIP = "127.0.0.1" }
$ApiPort  = $env:API_PORT;  if(-not $ApiPort){ $ApiPort = "18000" }
$UiPort   = $env:WEB_PORT;  if(-not $UiPort){ $UiPort = "8501" }
$Backend  = "http://$HostIP`:$ApiPort"
$Health   = "$Backend/health"

$OBase    = $env:OLLAMA_BASE_URL; if(-not $OBase){ $OBase = "http://127.0.0.1:11434" }
$OBase    = $OBase.TrimEnd('/')
$OTags    = "$OBase/api/tags"

LogInfo ("Using BACKEND   : {0}" -f $Backend)
LogInfo ("Using OLLAMA_URL: {0}" -f $OBase)
Write-Host ""

# Snapshot BEFORE
$before = @{
  "8501" = Port-State 8501
  "18000" = Port-State 18000
  "11434" = Port-State 11434
}

Write-Host "Before restart - port states:"
Write-Host ("  8501  -> {0}" -f $before["8501"])
Write-Host ("  18000 -> {0}" -f $before["18000"])
Write-Host ("  11434 -> {0}" -f $before["11434"])
Write-Host ""

# STOP
$stopSecs = Time-Action { 
  if(Test-Path "$Root\stop.ps1"){ & "$Root\stop.ps1" } else { LogWarn "stop.ps1 not found, skipping stop phase." }
}

# Small pause to ensure ports close
Start-Sleep -Seconds 2

# START
$startSecs = Time-Action { 
  if(Test-Path "$Root\start.ps1"){ & "$Root\start.ps1" } else { LogErr "start.ps1 not found. Cannot restart."; exit 1 }
}

# POST checks
Write-Host ""
LogInfo "Post-start checks..."
$backendOk = Test-Health -url $Health -retries 60 -delaySec 0.5
if($backendOk){ LogOk "Backend is healthy." } else { LogWarn "Backend health not responding." }

$ollamaOk = Test-Health -url $OTags -retries 10 -delaySec 0.5
if($ollamaOk){ LogOk "Ollama responds at /api/tags." } else { LogWarn "Ollama does not respond (maybe OpenAI fallback will be used)." }

# Snapshot AFTER
$after = @{
  "8501" = Port-State 8501
  "18000" = Port-State 18000
  "11434" = Port-State 11434
}

Write-Host ""
Write-Host "----------------------------------------------" -ForegroundColor White
Write-Host "REPORT" -ForegroundColor White
Write-Host "----------------------------------------------" -ForegroundColor White
Write-Host ("Stop phase time   : {0:n1} s" -f $stopSecs)
Write-Host ("Start phase time  : {0:n1} s" -f $startSecs)
Write-Host ""
Write-Host "Ports before  -> after"
Write-Host ("  8501  : {0} -> {1}" -f $before["8501"], $after["8501"])
Write-Host ("  18000 : {0} -> {1}" -f $before["18000"], $after["18000"])
Write-Host ("  11434 : {0} -> {1}" -f $before["11434"], $after["11434"])
Write-Host ""
Write-Host ("Backend health   : {0}" -f ($(if($backendOk){"OK"}else{"KO"})))
Write-Host ("Ollama /api/tags : {0}" -f ($(if($ollamaOk){"OK"}else{"KO"})))
Write-Host ""

# Desktop notification
$notifTitle = "AI Smart Credit Suite"
$notifMsg   = "Restart complete. UI on http://localhost:$UiPort  |  Backend: $($backendOk ? 'OK' : 'KO')  |  Ollama: $($ollamaOk ? 'OK' : 'KO')"
Show-Notification -title $notifTitle -message $notifMsg

LogOk "Restart complete. If UI is not open, browse: http://localhost:$UiPort"
