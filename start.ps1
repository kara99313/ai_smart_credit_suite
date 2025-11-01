# start.ps1
$ErrorActionPreference = "Stop"

function LogInfo($m){ Write-Host $m -ForegroundColor Cyan }
function LogOk($m){ Write-Host $m -ForegroundColor Green }
function LogWarn($m){ Write-Host $m -ForegroundColor Yellow }
function LogErr($m){ Write-Host $m -ForegroundColor Red }

function Test-Health($url, $retries = 60, $delaySec = 0.5){
  for($i=0; $i -lt $retries; $i++){
    try{
      $r = Invoke-WebRequest -Uri $url -TimeoutSec 2
      if($r.StatusCode -ge 200 -and $r.StatusCode -lt 300){ return $true }
    } catch {}
    Start-Sleep -Seconds $delaySec
  }
  return $false
}
function Wait-Http($url, $label, $retries = 60, $delaySec = 0.5){
  LogInfo ("Waiting for {0} at {1} ..." -f $label, $url)
  if (Test-Health -url $url -retries $retries -delaySec $delaySec){
    LogOk ("{0} is healthy." -f $label); return $true
  }
  LogErr ("{0} is not responding at {1}" -f $label, $url); return $false
}
function Show-Notification($title, $message){
  try{
    if (Get-Module -ListAvailable -Name BurntToast) {
      Import-Module BurntToast -ErrorAction SilentlyContinue | Out-Null
      New-BurntToastNotification -Text $title, $message | Out-Null
      return
    }
  }catch{}
  try{
    Add-Type -AssemblyName System.Windows.Forms | Out-Null
    [System.Windows.Forms.MessageBox]::Show($message, $title, 'OK', 'Information') | Out-Null
    return
  }catch{}
  Write-Host "`a"
  Write-Host ("[NOTIFY] {0} - {1}" -f $title, $message) -ForegroundColor Green
}

# 0) UTF-8 + PYTHONPATH (pour les imports relatifs des pages)
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Process")
[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "Process")
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::UTF8

# 1) Project root
$Root = $PSScriptRoot
Set-Location $Root
LogInfo ("Working directory: {0}" -f $Root)
# — important pour st.Page("app/xxx.py") —
[Environment]::SetEnvironmentVariable("PYTHONPATH", $Root, "Process")

# 2) Python from venv
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if(-not (Test-Path $VenvPy)){
  $VenvPy = Join-Path $Root "venv\Scripts\python.exe"
}
if(-not (Test-Path $VenvPy)){
  LogWarn "venv Python not found, fallback to system python"
  $VenvPy = "python"
}else{
  LogOk "Virtual env detected."
}

# 3) Load .env
$EnvPath = Join-Path $Root ".env"
if(Test-Path $EnvPath){
  LogInfo ".env found. Loading..."
  Get-Content $EnvPath | Where-Object { $_ -match '^\s*[^#].+=.+$' } | ForEach-Object {
    $kv = $_ -split '=', 2
    $k = $kv[0].Trim()
    $v = $kv[1].Trim()
    [Environment]::SetEnvironmentVariable($k, $v, "Process")
  }
}else{
  LogWarn ".env not found (not blocking)."
}

# 4) Resolve env vars (ports & backend url)
$HostIP  = $env:API_HOST;  if(-not $HostIP){ $HostIP = "127.0.0.1" }
$ApiPort = $env:API_PORT;  if(-not $ApiPort){ $ApiPort = "18000" }
# Par défaut on colle à Streamlit convention (8501) ; .env peut l’écraser
$UiPort  = $env:WEB_PORT;  if(-not $UiPort){ $UiPort = "8501" }

$BackendURL = "http://$HostIP`:$ApiPort"
[Environment]::SetEnvironmentVariable("BACKEND_URL", $BackendURL, "Process")

$LLMProv = $env:LLM_PROVIDER; if(-not $LLMProv){ $LLMProv = "ollama" }
$OModel  = $env:OLLAMA_MODEL; if(-not $OModel){ $OModel = "llama3.2:3b" }
$OBase   = $env:OLLAMA_BASE_URL; if(-not $OBase){ $OBase = "http://127.0.0.1:11434" }
$OBase   = $OBase.TrimEnd('/')

LogInfo ("LLM_PROVIDER = {0}" -f $LLMProv)
LogInfo ("OLLAMA_MODEL = {0}" -f $OModel)
LogInfo ("OLLAMA_BASE_URL = {0}" -f $OBase)
LogInfo ("BACKEND_URL  = {0}" -f $BackendURL)

# 5) (Optionnel) Vérif modèle
$ModelPath = $env:MODEL_PATH
if($ModelPath -and (Test-Path $ModelPath)){
  LogOk ("MODEL_PATH OK: {0}" -f (Resolve-Path $ModelPath))
}else{
  if($ModelPath){ LogWarn ("MODEL_PATH not found: {0}" -f $ModelPath) }
  else { LogWarn "MODEL_PATH not set (server.py default will be used)." }
}

# 6) (Optionnel) Démarrer Ollama si choisi
function Start-OllamaIfNeeded(){
  if($LLMProv.ToLower() -ne "ollama"){ return }
  $tagsUrl = "$OBase/api/tags"
  try{
    $r = Invoke-WebRequest -Uri $tagsUrl -TimeoutSec 2
    if($r.StatusCode -ge 200 -and $r.StatusCode -lt 300){ LogOk "Ollama running."; return }
  }catch{}
  $ollamaCmd = Get-Command "ollama" -ErrorAction SilentlyContinue
  if(-not $ollamaCmd){
    LogWarn "Ollama CLI not found. Skipping."
    return
  }
  LogInfo "Starting 'ollama serve'..."
  Start-Process -FilePath "ollama" -ArgumentList "serve" -WorkingDirectory $Root -WindowStyle Normal
}
Start-OllamaIfNeeded

# 7) Start backend (FastAPI on 18000)
LogInfo ("Starting backend on {0}:{1} ..." -f $HostIP, $ApiPort)
$BackendArgs = @("-m","uvicorn","server:app","--host",$HostIP,"--port",$ApiPort,"--reload")
Start-Process -FilePath $VenvPy -ArgumentList $BackendArgs -WorkingDirectory $Root -WindowStyle Normal

# 8) Wait /health
$HealthURL = "$BackendURL/health"
if(-not (Wait-Http -url $HealthURL -label "Backend" -retries 120 -delaySec 0.5)){
  LogErr ("Backend is not responding on {0}. Check the backend window for errors." -f $HealthURL)
}

# 9) Start Streamlit UI — EXACTEMENT comme ta commande OK
LogInfo ("Starting Streamlit UI on http://localhost:{0} ..." -f $UiPort)

# (A) Log de contrôle : version & chemin Python/Streamlit
& $VenvPy -c "import sys,streamlit,platform; print('STREAMLIT_VERSION=',streamlit.__version__); print('PYTHON_EXE=',sys.executable); print('OS=',platform.platform())"

# (B) Lancement identique à la commande manuelle qui marche
$UiArgs = @(
  "-m","streamlit","run","streamlit_app.py",
  "--server.address","127.0.0.1",           # forcé local (évite edge-cases)
  "--server.port",$UiPort,
  "--logger.level=debug"
)
Start-Process -FilePath $VenvPy -ArgumentList $UiArgs -WorkingDirectory $Root -WindowStyle Normal

# 10) Open browser
LogOk "Opening browser..."
Start-Process ("http://localhost:{0}" -f $UiPort)

# 11) Desktop notification
$notifTitle = "AI Smart Credit Suite"
$notifMsg   = "Startup complete. UI on http://localhost:$UiPort  |  Backend: $BackendURL  |  LLM: $LLMProv ($OBase)"
Show-Notification -title $notifTitle -message $notifMsg

LogOk "All set. To stop everything: .\stop.ps1"
