param([string]$InstallRoot = 'D:\Ollama', [string]$ModelsRoot = 'D:\Ollama\models', [switch]$SomenteLeve)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:OLLAMA_MODELS = $ModelsRoot
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
New-Item -ItemType Directory -Force -Path $InstallRoot, $ModelsRoot, (Join-Path $projectRoot 'logs'), (Join-Path $projectRoot 'downloads'), (Join-Path $projectRoot 'config') | Out-Null
$env:TEMP = Join-Path $InstallRoot 'tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ollamaExe = $null
$localConfigPath = Join-Path $projectRoot 'config\ollama-local.json'
if (Test-Path -LiteralPath $localConfigPath) {
    $localConfig = Get-Content -LiteralPath $localConfigPath -Raw | ConvertFrom-Json
    if (Test-Path -LiteralPath $localConfig.executavel) { $ollamaExe = $localConfig.executavel }
}
if (-not $ollamaExe) {
    $installed = Get-Command ollama -ErrorAction SilentlyContinue
    if ($installed) { $ollamaExe = $installed.Source }
}
if (-not $ollamaExe) {
    # CLI oficial portatil: nao depende de instalar binarios na unidade C:.
    $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/ollama/ollama/releases/latest' -Headers @{'User-Agent'='AutoRAG-Conceito'}
    $asset = $release.assets | Where-Object { $_.name -eq 'ollama-windows-amd64.zip' } | Select-Object -First 1
    if (-not $asset) { throw 'Pacote oficial Windows amd64 nao encontrado.' }
    $versionDir = Join-Path $InstallRoot ('versoes\' + $release.tag_name)
    $archivePath = Join-Path $projectRoot ('downloads\ollama-' + $release.tag_name + '.zip')
    Write-Host "Baixando Ollama $($release.tag_name) para a unidade D:..."
    Invoke-WebRequest -UseBasicParsing -Uri $asset.browser_download_url -OutFile $archivePath
    New-Item -ItemType Directory -Force -Path $versionDir | Out-Null
    Expand-Archive -LiteralPath $archivePath -DestinationPath $versionDir -Force
    $ollamaExe = Join-Path $versionDir 'ollama.exe'
    if (-not (Test-Path -LiteralPath $ollamaExe)) { throw 'Executavel ausente no pacote extraido.' }
}
@{executavel=$ollamaExe; modelos=$ModelsRoot; endpoint='http://127.0.0.1:11434'} | ConvertTo-Json | Set-Content -LiteralPath $localConfigPath -Encoding UTF8

$serverAlive = $false
try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 3 | Out-Null; $serverAlive = $true } catch {}
if (-not $serverAlive) {
    Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectRoot 'logs\ollama.stdout.log') -RedirectStandardError (Join-Path $projectRoot 'logs\ollama.stderr.log') | Out-Null
    for ($attempt=0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null; $serverAlive=$true; break } catch {}
    }
}
if (-not $serverAlive) { throw 'Ollama nao respondeu. Consulte logs\ollama.stderr.log.' }
if ($serverAlive) { Write-Host 'Servidor ativo. Se ele ja estava aberto, utiliza sua configuracao de pastas existente.' }
$models = @('embeddinggemma:300m-qat-q4_0')
if (-not $SomenteLeve) { $models += 'bge-m3' }
foreach ($model in $models) {
    Write-Host "Instalando modelo $model..."
    & $ollamaExe pull $model
    if ($LASTEXITCODE -ne 0) { throw "Falha ao baixar $model." }
}
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $python) {
    & $python (Join-Path $projectRoot 'autorag_cli.py') cadastrar --provedor ollama --modelo 'embeddinggemma:300m-qat-q4_0' --dimensoes 256
    if ($LASTEXITCODE -ne 0) { throw 'Falha no cadastro do perfil leve.' }
    if (-not $SomenteLeve) {
        & $python (Join-Path $projectRoot 'autorag_cli.py') cadastrar --provedor ollama --modelo bge-m3 --dimensoes 1024
        if ($LASTEXITCODE -ne 0) { throw 'Falha no cadastro do BGE-M3.' }
    }
    & $python (Join-Path $projectRoot 'diagnostico_modelos.py')
    if ($LASTEXITCODE -ne 0) { throw 'O teste de inferencia real falhou.' }
}
Write-Host 'Ollama pronto. Nenhum embedding da base foi gerado automaticamente.'
