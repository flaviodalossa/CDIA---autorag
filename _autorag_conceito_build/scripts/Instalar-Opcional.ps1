param([ValidateSet('local','analise','extracao')][string]$Grupo='analise')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Execute 01_configurar_ambiente.bat.' }
$env:PYTHONUTF8='1'
$env:PIP_CACHE_DIR=Join-Path $projectRoot '.cache\pip'
$env:TEMP=Join-Path $projectRoot '.cache\tmp'
$env:TMP=$env:TEMP
$env:HF_HOME=Join-Path $projectRoot '.cache\huggingface'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
& $python -m pip install -r (Join-Path $projectRoot "requirements-$Grupo.txt")
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar dependencias opcionais.' }
if ($Grupo -eq 'local') {
    & $python (Join-Path $projectRoot 'instalar_modelo_local.py')
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar modelo local.' }
}
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependencias incompativeis.' }
