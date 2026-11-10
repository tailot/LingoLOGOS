# ==============================================================================
# LingoLOGOS WSL Bridge & Automatic Environment Setup
# ==============================================================================

$WslDistro = "Ubuntu"
$WinProjectPath = "C:\Users\tailot\Desktop\progetti\LingoLOGOS"

# Converte il percorso Windows nel formato WSL (/mnt/c/...)
$WslProjectPath = $(wsl -d $WslDistro wslpath -u "$WinProjectPath").Trim()

Write-Host "[LingoLOGOS] Attivazione ambiente PowerShell..." -ForegroundColor Cyan
Write-Host "[LingoLOGOS] Collegamento al contesto WSL ($WslDistro) in $WslProjectPath..." -ForegroundColor Yellow

# Comando Bash per inizializzazione ed esecuzione in WSL
$WslInitCmd = @"
cd '$WslProjectPath'

# 1. Verifica/Creazione venv Linux isolato (.venv-wsl)
if [ ! -d ".venv-wsl" ]; then
    echo "[WSL] Creazione venv Linux (.venv-wsl)..."
    python3 -m venv .venv-wsl
fi

# 2. Attivazione venv Linux
source .venv-wsl/bin/activate

# 3. Sincronizzazione dipendenze se presente requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
fi

# 4. Avvio shell interattiva pronta per compilazione ed esecuzione lllayer
echo "[WSL] Ambiente LingoLOGOS pronto!"
exec bash --login
"@

# Esecuzione WSL in modalita interattiva sulla cartella del progetto
wsl -d $WslDistro --cd "$WslProjectPath" bash -c "$WslInitCmd"
