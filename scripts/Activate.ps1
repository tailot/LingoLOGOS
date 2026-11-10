<#
.SYNOPSIS
    LingoLOGOS developer entry point: Windows PowerShell to WSL bridge.

.DESCRIPTION
    The bridge enters the Linux environment of the checkout and leaves an
    interactive shell in it. It is written so that any developer can run it from
    any account and from any location of the file system:

      1. The checkout is found by walking up from the position of this file until
         the repository fingerprint appears, so there is no hard coded path and no
         assumption about where a person clones the repository.
      2. The distribution is chosen from the -Distro parameter, then from the
         LINGOLOGOS_WSL_DISTRO environment variable, then from the Ubuntu
         distribution registered on that machine, then from the default
         distribution of WSL, then from the only registered one.
      3. The Linux side lives in scripts/wsl_bootstrap.sh, which does the same
         thing for a developer working from a shell inside WSL or Linux.

    The virtual environment, the dependency installation and the dependency
    synchronisation are idempotent: rerunning the file only repairs what is missing.

.PARAMETER Distro
    WSL distribution to work in. Optional, see the resolution order above.

.PARAMETER ProjectRoot
    Checkout to enter. Optional, by default derived from this file, or from
    LINGOLOGOS_PROJECT_ROOT when that variable is set.

.PARAMETER NoSync
    Attach the existing Linux virtual environment without synchronising
    requirements.txt, which is what a developer offline or on a metered link wants.

.EXAMPLE
    .\scripts\Activate.ps1

.EXAMPLE
    .\scripts\Activate.ps1 -Distro Ubuntu-24.04 -NoSync
#>
[CmdletBinding()]
param(
    [string]$Distro = $env:LINGOLOGOS_WSL_DISTRO,
    [string]$ProjectRoot = $env:LINGOLOGOS_PROJECT_ROOT,
    [switch]$NoSync
)

$ErrorActionPreference = 'Stop'

function ConvertFrom-WslOutput {
    # wsl writes UTF-16, so what PowerShell captures arrives with nul bytes between
    # the characters. Those bytes have to be removed before anything is split or
    # compared, not trimmed. The answer is the list of non empty lines.
    param([AllowEmptyString()][string]$Raw = '')
    return @((($Raw -replace "`0", '') -split "`r?`n") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Write-Host '[FAIL] wsl.exe is not on PATH, so this machine has no Windows Subsystem for Linux.'
    Write-Host '[FAIL] install it with: wsl --install'
    Write-Host '[FAIL] or work from a Linux host and run scripts/wsl_bootstrap.sh there.'
    exit 1
}

# 1. The checkout. Two copies of this entry point exist at different depths, one in
# scripts and one in the Scripts directory of the Windows training environment, so
# counting parent directories would be a guess that breaks when a third copy
# appears. The fingerprint of the repository is what makes the answer independent of
# the layout, and the deepest match wins because a clone nested inside another
# checkout has to use its own tree.
if (-not $ProjectRoot) {
    $candidate = $PSScriptRoot
    while ($candidate) {
        $requirements = Join-Path $candidate 'requirements.txt'
        $manifest = Join-Path $candidate 'pyproject.toml'
        $source = Join-Path $candidate 'src'
        if ((Test-Path -LiteralPath $requirements -PathType Leaf) -and
            ((Test-Path -LiteralPath $manifest -PathType Leaf) -or
             (Test-Path -LiteralPath $source -PathType Container))) {
            $ProjectRoot = $candidate
            break
        }
        # -Path and not -LiteralPath here: Windows PowerShell 5.1 refuses
        # Split-Path -LiteralPath together with -Parent as an ambiguous parameter
        # set, so the working form has to be the one that stays in the file.
        $parent = Split-Path -Path $candidate -Parent
        if (-not $parent -or $parent -eq $candidate) {
            break
        }
        $candidate = $parent
    }
}
if (-not $ProjectRoot) {
    Write-Host '[FAIL] the checkout was not found above this file and no -ProjectRoot was given.'
    Write-Host '[FAIL] run the Activate.ps1 of the checkout you want, or pass -ProjectRoot.'
    exit 1
}
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    Write-Host "[FAIL] '$ProjectRoot' is not a directory."
    exit 1
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'requirements.txt') -PathType Leaf)) {
    Write-Host "[FAIL] '$ProjectRoot' has no requirements.txt, it is not the LingoLOGOS checkout."
    exit 1
}

# 2. The distribution. The default of WSL is not necessarily a distribution a build
# can run in, because a machine with Docker Desktop also has docker-desktop and that
# one becomes the default. A distribution named Ubuntu is therefore preferred over
# the default, and an explicit choice always wins.
$registered = ''
try {
    $registered = (& wsl.exe -l -v 2>$null) -join "`n"
} catch {
    Write-Host '[FAIL] wsl did not answer, so there is no Linux environment to enter.'
    Write-Host '[FAIL] install WSL 2 with a distribution: wsl --install -d Ubuntu'
    exit 1
}
$names = @()
$defaultDistro = $null
foreach ($line in (ConvertFrom-WslOutput $registered)) {
    $isDefault = $line.StartsWith('*')
    $body = $line
    if ($isDefault) {
        $body = $line.Substring(1).Trim()
    }
    $fields = @($body -split '\s{2,}')
    # The version column is the last field and it is always a number, which is how
    # the header line is dropped whatever language the WSL output is in.
    if ($fields.Count -lt 3) {
        continue
    }
    if ($fields[-1] -notmatch '^[0-9]+$') {
        continue
    }
    $name = ($fields[0..($fields.Count - 3)] -join '  ')
    $names += $name
    if ($isDefault) {
        $defaultDistro = $name
    }
}
if ($names.Count -eq 0) {
    Write-Host '[FAIL] WSL is present but no distribution is registered.'
    Write-Host '[FAIL] add one: wsl --install -d Ubuntu'
    exit 1
}
if ($Distro) {
    if ($names -notcontains $Distro) {
        Write-Host "[FAIL] '$Distro' is not a registered WSL distribution."
        Write-Host ("[FAIL] registered: " + ($names -join ', '))
        Write-Host '[FAIL] change the name, or register it with: wsl --install -d <name>'
        exit 1
    }
} else {
    foreach ($name in $names) {
        if ($name -like 'Ubuntu*') {
            $Distro = $name
            break
        }
    }
    if (-not $Distro) {
        $Distro = $defaultDistro
    }
    if (-not $Distro -and $names.Count -eq 1) {
        $Distro = $names[0]
    }
    if (-not $Distro) {
        Write-Host '[FAIL] no obvious distribution to work in and WSL has no default one.'
        Write-Host ("[FAIL] registered: " + ($names -join ', '))
        Write-Host '[FAIL] pass -Distro with the name you want, or set LINGOLOGOS_WSL_DISTRO.'
        exit 1
    }
}

# 3. Prove that the distribution can see the checkout before anything is created in
# it. A Windows drive that is not automounted, or a path that only exists for this
# account, fails here with the reason instead of in the middle of a pip run.
# A Windows path is handed to the Linux side with forward slashes only. As written
# with backslashes it does not survive the command line layers between PowerShell,
# wsl.exe and the Linux process: wslpath was seen receiving C:UserstailotDesktop...,
# every separator eaten, and it answered that the path does not exist.
$WindowsPath = $ProjectRoot -replace '\\', '/'
$converted = ''
try {
    $converted = (& wsl.exe -d $Distro wslpath -u $WindowsPath 2>$null) -join "`n"
} catch {
    $converted = ''
}
$convertedLines = ConvertFrom-WslOutput $converted
if ($convertedLines.Count -lt 1) {
    Write-Host "[FAIL] '$Distro' cannot see '$ProjectRoot'."
    Write-Host '[FAIL] Windows drives are reached under /mnt/<letter>; check that automount is'
    Write-Host '[FAIL] enabled in /etc/wsl.conf of that distribution.'
    exit 1
}
$WslPath = $convertedLines[0]

$bootstrap = Join-Path $ProjectRoot 'scripts\wsl_bootstrap.sh'
if (-not (Test-Path -LiteralPath $bootstrap -PathType Leaf)) {
    Write-Host "[FAIL] scripts\wsl_bootstrap.sh is missing from '$ProjectRoot'."
    Write-Host '[FAIL] restore it from git, the Linux steps live in that file.'
    exit 1
}
# bash reads a CRLF shell script as a script full of trailing carriage returns, and
# reports it as a syntax problem far from the cause. core.autocrlf is on in this
# checkout, so the attribute of .gitattributes is what keeps this file on LF, and
# the check catches a local editor that undid that.
$bootstrapBytes = [System.IO.File]::ReadAllBytes($bootstrap)
if ($bootstrapBytes -contains 13) {
    Write-Host '[FAIL] scripts\wsl_bootstrap.sh has CRLF line endings, bash cannot read it.'
    Write-Host '[FAIL] write it with LF endings, then: git add --renormalize scripts/wsl_bootstrap.sh'
    exit 1
}

Write-Host '[LingoLOGOS] activating the PowerShell environment' -ForegroundColor Cyan
Write-Host "[LingoLOGOS] checkout      : $ProjectRoot"
Write-Host "[LingoLOGOS] distribution  : $Distro"
Write-Host "[LingoLOGOS] wsl path      : $WslPath"
if ($NoSync) {
    Write-Host '[LingoLOGOS] dependency synchronisation is off'
}

# 4. Hand the work to the POSIX script of the same checkout. wsl.exe enters the
# checkout itself through --cd, and the script is then addressed by a path relative
# to it, so the only path on that command line is the one wsl.exe reads on its own
# and the quoting layers have nothing to eat.
$wslArgs = @('-d', $Distro, '--cd', $WindowsPath, '--', 'sh', 'scripts/wsl_bootstrap.sh')
if ($NoSync) {
    $wslArgs += '--skip-pip'
}
& wsl.exe @wslArgs
exit $LASTEXITCODE
