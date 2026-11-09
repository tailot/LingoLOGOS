<#
.SYNOPSIS
    One command from Windows for the OpenLane plus sky130A flow, setup included.

.DESCRIPTION
    Three things a shell inside WSL cannot do, done here: confirm Docker Desktop is
    installed, start it when the engine is not answering, and create the Ubuntu
    distribution when none exists. Then asic_chip/scripts/setup_openlane_flow.sh runs
    inside that distribution and does the rest: host packages, openlane and volare in
    a virtual environment, the upstream smoke test, the sky130A PDK, and the flow
    itself through asic_chip/scripts/build_gds.sh.

    The checkout that gets installed is the directory holding this file, found
    through the path of the file itself, so the current directory of whoever starts
    it does not matter. A shell inside WSL can start it too, but only with the path
    to this file quoted or written with forward slashes, because that shell removes
    backslashes. The file stops before copying anything when the directory it
    belongs to is not a checkout.

    Nothing large is downloaded by this file. The measured sizes on this machine are
    7.39 GB of docker disk for the openlane2 2.3.10 container image and 2.1 GiB for one
    sky130 version with the library set this flow reads, or 16 GiB for that same version
    with every library, against the 700 MiB to 1.5 GiB upstream quotes for a version. The
    work lands in the WSL filesystem rather than under /mnt/c, because the flow performs
    a very large number of small file operations.

.PARAMETER Distro
    The WSL distribution to run in. Created with wsl --install -d when absent.

.PARAMETER Copy
    Put the checkout inside the distribution first. By default the working tree is
    copied as it stands, uncommitted files included, through robocopy addressed to
    the distribution over the WSL file share. The copy is additive: it overwrites the
    files it brings and never deletes what is already inside the distribution, so the
    runs directory an earlier flow left under asic_chip/openlane/runs survives, while
    any edit made to a file there is replaced by the one in this directory.

.PARAMETER FromCommit
    With -Copy, take a snapshot of the named revision with git archive instead of
    the working tree. Anything that is only in the working tree is then absent from
    the copy, which is what makes the scripts of this very file missing on a fresh
    machine that has not committed them yet.

.PARAMETER RepoRoot
    Repository path inside the distribution, as the distribution sees it. Defaults to
    ~/LingoLOGOS, which the distribution expands.

.PARAMETER FlowOnly
    Skip the setup inside the distribution and go straight to the flow.

.PARAMETER SkipSmoke
    Skip the upstream smoke test, the small design the container route is checked with
    before the real one. Passes --skip-smoke to the script inside the distribution. Worth
    using when the container image and the PDK are already known good, which is the case
    on a machine that has already run the flow once.

.PARAMETER Spec
    Run the flow at the declared 4.0 ns budget instead of the 32.0 ns bring-up gate it
    uses by default. This passes --spec to the script inside the distribution, which
    exports LL_OPENLANE_CONFIG=config.json.

.PARAMETER NoPpa
    Refuse to add a third party apt repository inside the distribution. Ubuntu 26.04
    ships only python3.14 and has no older interpreter in its own archive, while the
    openlane dependency libparse has no wheel past cp312, so python3.12 comes from
    ppa:deadsnakes/ppa unless this switch is used. Without that repository the script
    inside the distribution fetches a standalone build of CPython 3.12 instead, and
    LL_PYTHON overrides both routes with an interpreter of your own.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File asic_chip\scripts\setup_openlane_flow.ps1 -Copy

.EXAMPLE
    powershell.exe -ExecutionPolicy Bypass -File 'C:\path\to\checkout\asic_chip\scripts\setup_openlane_flow.ps1' -Copy
    The form for a shell inside WSL, where the backslashes of the first form would be
    removed before the path reaches PowerShell.

.NOTES
    The engine check, the distribution creation, the copy into the distribution, the
    resolution of the checkout from the path of this file and the refusal of a
    directory that is not a checkout have run on this machine. Inside the distribution
    an attempt on the default interpreter stopped in pip, because no wheel of the
    openlane dependencies reaches it; the attempt on Python 3.12 got past that install
    and stopped one step later, at the PDK listing, where the volare of that install
    knows neither a --global option nor a search command. The POSIX script now uses the
    ls-remote, fetch and enable subcommands that version does have, and installs the tk
    module of the interpreter, which openlane imports before it reads a design. That
    listing, the PDK download and the flow have since run here: ~/pdk/sky130A exists, and
    the flow has now run three times. The first time it stopped at the custom I/O
    placement step, the 25th, on a pin order file written in the format of OpenLane 1,
    which this tree rewrote in the grammar the installed tool reads. The second time, as
    RUN_2026-11-09_18-43-18, all 78 steps of the OpenLane 2 Classic flow ran in 16 min 59
    s, three GDSII files were written, DRC reported zero violations and LVS reported that
    the circuits match uniquely, and the flow then exited non zero on Antenna Failed and
    on deferred setup and hold errors, which is the timing gate reporting that the layout
    did not close the 8.0 ns period it was constrained to; that measurement is why the
    bring-up gate became 30.0 ns. The third time, as RUN_2026-11-09_21-14-43 at that gate,
    the 78 steps ran in 12 min 12 s and the run reported Antenna Passed, LVS Passed and
    DRC Passed, with zero violations in both the Magic and the KLayout rule checks and no
    design rule error left by the detailed router, and it still exited non zero: setup was
    0.4906 ns short in max_ss_100C_1v60 over 35 register to register paths and hold was
    0.1540 ns short in the same corner over 10 paths that all start at an input port,
    because this flow repairs hold before detailed routing and never after it. That is why
    the gate is 32.0 ns now and why PL_RESIZER_HOLD_SLACK_MARGIN and
    GRT_RESIZER_HOLD_SLACK_MARGIN are set to 0.40 ns in both constraint sets. The smoke
    test has not been run on its own. All of these results are recorded in
    asic_chip/doc/verification_report.md, which also records the sizes the downloads
    really cost: 7.39 GB for the tool image and 2.1 GiB for the PDK library set this flow
    reads, against 16 GiB for every library of the same version.
#>
[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu',
    [string]$RepoRoot = '~/LingoLOGOS',
    [string]$FromCommit = '',
    [switch]$Copy,
    [switch]$FlowOnly,
    [switch]$SkipSmoke,
    [switch]$Spec,
    [switch]$NoPpa
)

$ErrorActionPreference = 'Stop'
# The tree to copy is the one this file sits in, not the one the caller happens to
# be standing in. The same file gets started from a Windows shell, from a shell
# inside WSL, and from inside the copy in the distribution, and each of those has
# a different current directory, so the current directory is not a usable answer.
# This file is two directories below the root of the checkout.
$WindowsRepo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Write-Host "[setup] checkout to install: $WindowsRepo"
if (-not (Test-Path (Join-Path $WindowsRepo 'asic_chip\config.json'))) {
    # The flow later runs the scripts of that same tree inside the distribution, so
    # a root that is not a checkout has to stop here rather than copy something
    # that is not the repository.
    Write-Host "[FAIL] '$WindowsRepo' is not the checkout this file belongs to, no asic_chip\config.json"
    Write-Host '[FAIL] run the setup_openlane_flow.ps1 of the checkout you want installed'
    exit 1
}

function Test-WslDistro {
    param([string]$Name)
    # wsl writes UTF-16, so the captured string arrives with nul bytes between the
    # characters. The nuls have to be removed before splitting, not trimmed.
    $raw = (wsl -l -q 2>$null) -join "`n"
    if (-not $raw) { return $false }
    $clean = @((($raw -replace "`0", '') -split "`r?`n") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    return $clean -contains $Name
}

# 1. The docker CLI alone is not enough. This machine has the client and no
# reachable engine, which is exactly the failure the troubleshooting table of
# asic_chip/doc/build_instructions.md describes.
$cli = Get-Command docker -ErrorAction SilentlyContinue
if (-not $cli) {
    Write-Host '[FAIL] the docker CLI is not on PATH. Install Docker Desktop for Windows.'
    exit 127
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    $desktop = @(
        (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),
        (Join-Path $env:LOCALAPPDATA 'Docker\Docker Desktop.exe')
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $desktop) {
        Write-Host '[FAIL] the docker CLI is present but Docker Desktop is not installed.'
        Write-Host '[FAIL] Install it with the WSL 2 engine option, then rerun this file.'
        exit 127
    }
    Write-Host '[setup] the engine did not answer, starting Docker Desktop'
    Start-Process $desktop
    $deadline = (Get-Date).AddSeconds(240)
    $ready = $false
    while (-not $ready) {
        Start-Sleep -Seconds 5
        docker info *> $null
        if ($LASTEXITCODE -eq 0) { $ready = $true }
        if (-not $ready -and (Get-Date) -gt $deadline) {
            Write-Host '[FAIL] the engine did not come up within four minutes.'
            Write-Host '[FAIL] Open Docker Desktop, wait for the engine to report running, and rerun.'
            exit 1
        }
    }
    Write-Host '[setup] engine is up'
}

# 2. A Linux distribution to work in. docker-desktop is not one you can build in.
if (-not (Test-WslDistro -Name $Distro)) {
    Write-Host "[setup] no '$Distro' distribution, installing it"
    wsl --install -d $Distro
    if (-not (Test-WslDistro -Name $Distro)) {
        Write-Host "[FAIL] '$Distro' is still absent. Run: wsl --install -d $Distro"
        Write-Host '[FAIL] create the user it asks for, then rerun this file.'
        exit 1
    }
}

$Marker = "$RepoRoot/asic_chip/scripts/setup_openlane_flow.sh"

# 3. The checkout belongs inside the distribution, not under /mnt/c, because a
# bind mount over the Windows file system makes every small file operation slow.
if ($Copy) {
    Write-Host "[setup] putting the checkout into $RepoRoot inside $Distro"
    # Absolute paths are resolved on the distribution side and used verbatim. A
    # Windows path with backslashes does not survive the PowerShell, cmd and sh
    # quoting layers intact, so wslpath is never handed one.
    $homeWsl = (wsl -d $Distro -- sh -c 'echo $HOME').Trim()
    $targetWsl = "$homeWsl/LingoLOGOS"
    $targetWin = '\\wsl.localhost\' + $Distro + ($homeWsl -replace '/', '\') + '\LingoLOGOS'
    wsl -d $Distro -- mkdir -p $targetWsl
    if ($FromCommit) {
        # A committed snapshot, with nothing that is only in the working tree.
        # Verified form: the archive is handed to tar on standard input, with cmd
        # doing the redirection because PowerShell corrupts a binary stream.
        $archive = Join-Path $env:TEMP 'lingologos_head.tar'
        Push-Location $WindowsRepo
        git archive --format=tar --output "$archive" $FromCommit
        $gitExit = $LASTEXITCODE
        Pop-Location
        if ($gitExit -ne 0) {
            Remove-Item $archive -Force -ErrorAction SilentlyContinue
            Write-Host "[FAIL] git archive $FromCommit failed, does that commit exist?"
            exit 1
        }
        cmd /c "wsl -d $Distro -- tar -xf - -C ""$targetWsl"" < ""$archive"""
        $tarExit = $LASTEXITCODE
        Remove-Item $archive -Force -ErrorAction SilentlyContinue
        if ($tarExit -ne 0) { Write-Host '[FAIL] the distribution could not unpack the archive'; exit 1 }
    } else {
        # The working tree as it stands, uncommitted files included, because that is
        # what a run of this repository needs. robocopy addressing the distribution
        # through the WSL file share was checked against this machine: it lands in
        # the distribution filesystem, not under /mnt/c, and exits below 8.
        $roboArgs = @($WindowsRepo, $targetWin, '/E', '/XD',
            'obj_dir', 'obj_dir_tb', 'obj_dir_vvp', '__pycache__', '.git',
            '/XF', '*.vcd', '*.vvp', '*.log', '/NFL', '/NDL', '/NJH', '/NJS', '/NP')
        robocopy @roboArgs
        if ($LASTEXITCODE -ge 8) { Write-Host '[FAIL] robocopy could not copy the checkout'; exit 1 }
    }
}

$present = (wsl -d $Distro -- sh -c "test -f $Marker && echo yes || echo no").Trim()
if ($present -ne 'yes') {
    Write-Host "[FAIL] $Marker is not present inside '$Distro'."
    Write-Host '[FAIL] clone the repository inside WSL, or rerun this file with -Copy.'
    exit 1
}

# 4. Everything else is the POSIX script, which is also the route a plain Linux
# user takes from a shell on their own machine.
$flags = ''
if ($FlowOnly) { $flags += ' --flow-only' }
if ($SkipSmoke) { $flags += ' --skip-smoke' }
if ($Spec) { $flags += ' --spec' }
if ($NoPpa) { $flags += ' --no-ppa' }
Write-Host "[setup] running setup_openlane_flow.sh inside $Distro"
wsl -d $Distro -- sh -c "sh $Marker$flags"
exit $LASTEXITCODE
