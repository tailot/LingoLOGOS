# ==============================================================================
# LingoLOGOS WSL Bridge & Automatic Environment Setup
# ==============================================================================

param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectPath = $PSScriptRoot
)

$WslDistro = "Ubuntu"
$WinProjectPath = $ProjectPath

$WslProjectPath = $(wsl -d $WslDistro wslpath -u "$WinProjectPath").Trim()

Write-Host "[LingoLOGOS] Attivazione ambiente PowerShell..." -ForegroundColor Cyan
Write-Host "[LingoLOGOS] Collegamento al contesto WSL ($WslDistro) in $WslProjectPath..." -ForegroundColor Yellow

$WslInitCmd = @"
cd '$WslProjectPath'

if [ ! -d ".venv-wsl" ]; then
    echo "[WSL] Creazione venv Linux (.venv-wsl)..."
    python3 -m venv .venv-wsl
fi

source .venv-wsl/bin/activate

if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
fi

echo "[WSL] Ambiente LingoLOGOS pronto!"
exec bash --login
"@

wsl -d $WslDistro --cd "$WslProjectPath" bash -c "$WslInitCmd"
