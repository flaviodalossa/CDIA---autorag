$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot 'config\ollama-local.json'
if (-not (Test-Path -LiteralPath $configPath)) { throw 'Execute 04_configurar_ollama.bat primeiro.' }
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$env:OLLAMA_MODELS = $config.modelos
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
$env:TEMP = Join-Path (Split-Path -Parent $config.modelos) 'tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP, (Join-Path $projectRoot 'logs') | Out-Null
try { Invoke-RestMethod -Uri ($config.endpoint + '/api/version') -TimeoutSec 2 | Out-Null; exit 0 } catch {}
Start-Process -FilePath $config.executavel -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectRoot 'logs\ollama.stdout.log') -RedirectStandardError (Join-Path $projectRoot 'logs\ollama.stderr.log') | Out-Null
Write-Host 'Ollama iniciado em segundo plano.'
