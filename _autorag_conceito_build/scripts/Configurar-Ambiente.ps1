param([string]$PythonExe = "", [switch]$InstalarPython, [string]$PythonVersion = '3.13.16')
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
$env:TEMP = Join-Path $projectRoot '.cache\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

function Invoke-PythonChecked([string]$exePath, [string[]]$pythonArgs) {
    & $exePath @pythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Python terminou com codigo $LASTEXITCODE." }
}

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    if (-not $PythonExe) {
        $candidates = @()
        $py = Get-Command py -ErrorAction SilentlyContinue
        if ($py) {
            $resolved = & $py.Source -3 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0) { $candidates += $resolved }
        }
        $regular = Get-Command python -ErrorAction SilentlyContinue
        if ($regular -and $regular.Source -notmatch 'WindowsApps') { $candidates += $regular.Source }
        $bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $bundled) { $candidates += $bundled }
        $projectPython = Join-Path $projectRoot '.python\python.exe'
        if (Test-Path -LiteralPath $projectPython) { $candidates = @($projectPython) + $candidates }
        foreach ($candidate in $candidates) {
            & $candidate -c "import sys, tkinter, venv, ensurepip; assert sys.version_info >= (3,10); tkinter.Tcl().eval('info library')" 2>$null
            if ($LASTEXITCODE -eq 0) { $PythonExe = $candidate; break }
        }
    }
    if (-not $PythonExe -and $InstalarPython) {
        $downloadDir = Join-Path $projectRoot 'downloads'
        New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
        $installerPath = Join-Path $downloadDir "python-$PythonVersion-amd64.exe"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe" -OutFile $installerPath
        $pythonRoot = Join-Path $projectRoot '.python'
        $installerArgs = @('/quiet', 'InstallAllUsers=0', "TargetDir=`"$pythonRoot`"", 'Include_launcher=0', 'PrependPath=0', 'AssociateFiles=0', 'Shortcuts=0', 'Include_test=0', 'Include_doc=0', 'Include_tcltk=1', 'Include_pip=1')
        $installProcess = Start-Process -FilePath $installerPath -ArgumentList $installerArgs -WindowStyle Hidden -Wait -PassThru
        if ($installProcess.ExitCode -notin @(0,3010)) { throw "Instalador Python terminou com codigo $($installProcess.ExitCode)." }
        $PythonExe = Join-Path $pythonRoot 'python.exe'
    }
    if (-not $PythonExe) {
        throw 'Instale Python 3.10 ou superior com Tcl/Tk, ou use -PythonExe caminho\python.exe. Para instalar o Python oficial nesta pasta: 01_configurar_ambiente.bat -InstalarPython.'
    }
    Invoke-PythonChecked $PythonExe @('-c', "import sys, tkinter, venv, ensurepip; assert sys.version_info >= (3,10); tkinter.Tcl().eval('info library')")
    Write-Host "Criando ambiente com $PythonExe"
    Invoke-PythonChecked $PythonExe @('-m', 'venv', (Join-Path $projectRoot '.venv'))
}
Invoke-PythonChecked $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-PythonChecked $venvPython @('-m', 'pip', 'install', '-r', (Join-Path $projectRoot 'requirements.txt'))
Invoke-PythonChecked $venvPython @('-m', 'pip', 'check')
Invoke-PythonChecked $venvPython @('-c', "import numpy, tkinter; print('Ambiente OK; NumPy', numpy.__version__, 'Tcl', tkinter.Tcl().eval('info patchlevel'))")
Write-Host 'Ambiente pronto. Abra 02_abrir_aplicacao.bat.'
