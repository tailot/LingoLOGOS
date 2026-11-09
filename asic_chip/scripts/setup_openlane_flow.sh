#!/bin/sh
# -----------------------------------------------------------------------------
# One command for the whole OpenLane plus sky130A route, setup included.
#
# Run it inside the Ubuntu distribution that WSL 2 provides, from anywhere:
#
#   sh ~/LingoLOGOS/asic_chip/scripts/setup_openlane_flow.sh
#
# What it does, in order:
#   1. checks that the Docker engine answers, and names the fix when it does not;
#   2. installs the host packages the flow expects, only if they are missing, and
#      picks an interpreter, 3.12 at best, whose version has prebuilt wheels for
#      both compiled dependencies of openlane and can import tkinter, which openlane
#      needs before it reads a design: the distribution default first, then the
#      archive Ubuntu defers to for older interpreters, then a standalone build of
#      3.12 fetched under the user account;
#   3. creates ~/openlane-venv on that interpreter and installs openlane plus
#      volare into it;
#   4. installs the sky130A PDK that volare publishes, at $PDK_ROOT, default
#      ~/pdk, and exposes the variant under the name the flow asks for;
#   5. runs the upstream smoke test, which pulls the OpenLane container image and
#      proves the container route produces a GDSII for a small design;
#   6. hands the design to asic_chip/scripts/build_gds.sh, which is the declared
#      flow entry point and still refuses to claim a layout it did not produce.
#
# It is idempotent: a second run skips what is already on disk and goes straight
# to the flow. What it does not do, because a shell inside WSL cannot do it, is
# install WSL, install Docker Desktop, or start the daemon. Run
# asic_chip/scripts/setup_openlane_flow.ps1 from PowerShell for those three.
#
# Flags:
#   --flow-only    skip the setup and run the flow; fails if the tools are absent
#   --skip-smoke   skip the smoke test, for when the image and the PDK are known good
#   --spec         run the flow at the declared 4.0 ns budget instead of the default
#                  32.0 ns bring-up gate, by exporting LL_OPENLANE_CONFIG=config.json
#   --no-ppa       never add a third party apt repository; needed for python3.12 on
#                  Ubuntu 26.04, so with this flag give the interpreter yourself
#                  through LL_PYTHON
#
# The downloads are large and slow the first time, and the sizes measured here are
# the honest ones: the openlane2 2.3.10 image takes 7.39 GB of docker disk, one
# sky130 version with the libraries this flow reads takes 2.1 GiB, and the same
# version with -l all takes 16 GiB. Upstream says 700 MiB to 1.5 GiB per version,
# which is in the range for the default library set and well under the large one.
# Every step of this file except the smoke test has run on this machine: the docker
# check, the apt step, the interpreter choice that landed on Python 3.12.16, the
# install of openlane and volare into ~/openlane-venv, and the volare ls-remote, fetch
# and enable that created ~/pdk/sky130A. The smoke test has not been run on its own.
# The flow of the last step has run three times. The first time it stopped at the custom
# I/O placement step, the 25th, on a pin order file this repository then rewrote in the
# grammar that step reads. The second time, as RUN_2026-11-09_18-43-18, it went through
# all 78 steps of the OpenLane 2 Classic flow in 16 min 59 s, wrote three GDSII files and
# reported Antenna Failed with 54 violating pins, then exited non zero on deferred setup
# and hold errors at the 8.0 ns gate of that run, which is when the bring-up period became
# 30.0 ns. The third time, as RUN_2026-11-09_21-14-43 at that 30.0 ns gate, it ran all 78
# steps in 12 min 12 s and reported Antenna Passed, LVS Passed and DRC Passed with zero
# violations, and still exited non zero on deferred setup errors in one corner and hold
# errors in seven, which is when the period became 32.0 ns and the resizer hold margins
# became 0.40 ns. doc/verification_report.md keeps the measurements.
# -----------------------------------------------------------------------------

set -eu

ASIC_CHIP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LL_REPO_ROOT="$(cd "$ASIC_CHIP_DIR/.." && pwd)"
VENV="$HOME/openlane-venv"
export PDK_ROOT="${PDK_ROOT:-$HOME/pdk}"
PDK=sky130A
STD_CELL_LIBRARY=sky130_fd_sc_hd
export PDK PDK_ROOT STD_CELL_LIBRARY

RUN_SETUP=1
RUN_SMOKE=1
USE_PPA=1
for arg in "$@"; do
    case "$arg" in
        --flow-only)  RUN_SETUP=0 ;;
        --skip-smoke) RUN_SMOKE=0 ;;
        --no-ppa)     USE_PPA=0 ;;
        --spec)       LL_OPENLANE_CONFIG=config.json; export LL_OPENLANE_CONFIG ;;
        *) echo "[FAIL] unknown flag: $arg" >&2; exit 2 ;;
    esac
done

say() { echo "[setup] $*"; }
die() { echo "[FAIL] $*" >&2; exit "${2:-1}"; }

# 1. The engine has to answer before anything can run. This is the failure mode
# that the checkout has never escaped: a docker client is present, a daemon is not.
if ! docker info >/dev/null 2>&1; then
    echo "[FAIL] the Docker engine did not answer. Start Docker Desktop from Windows" >&2
    echo "[FAIL] and enable its WSL integration for this distribution, then rerun." >&2
    exit 127
fi
say "docker engine: $(docker --version)"

# 2. Host packages, installed only when absent so a rerun costs nothing.
NEED_APT=0
for prog in python3 make git curl; do
    command -v "$prog" >/dev/null 2>&1 || NEED_APT=1
done
# The interpreter the virtual environment is built on, selected rather than
# trusted. openlane declares python >=3.8,<4 and pins klayout >=0.29.0,<0.30.0
# and libparse >=0.3.1,<1. The wheel sets of the two releases those pins resolve
# to were read from the Python package index: klayout 0.29.12 publishes manylinux
# wheels for cp38 through cp313, libparse 0.3.1 only through cp312. The narrower
# set decides, so 3.12 is the newest interpreter this flow installs from wheels,
# and any newer one sends pip to the source distributions. That path was walked
# here by accident and failed twice: klayout on zlib.h, curl/curl.h, png.h and
# expat.h missing and then on -lcurl, -lexpat and -lpng at link time, libparse on
# "make: *** No rule to make target 'patch'", all recorded in
# doc/verification_report.md. Ubuntu 26.04.1 LTS, the distribution this machine
# created, ships python3 at 3.14.4 and carries no older interpreter in apt, so
# 3.12 has to come from ppa:deadsnakes/ppa or from LL_PYTHON. Prefer 3.12, then
# 3.11, 3.10 and 3.9, and let LL_PYTHON name another interpreter outright.
select_python() {
    for candidate in "${LL_PYTHON:-}" python3.12 python3.11 python3.10 python3.9 python3; do
        if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1; then
            PYBIN="$candidate"
            return 0
        fi
    done
    return 1
}
too_new() {
    # True when the interpreter that would be chosen has no prebuilt libparse.
    [ -n "${PYBIN:-}" ] &&
        "$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)' 2>/dev/null
}
usable_python() {
    select_python && ! too_new
}
STANDALONE_PYTHON_URL="${LL_PYTHON_URL:-https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.12.16%2B20261026-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz}"
STANDALONE_PYTHON_DIR="$HOME/.local/share/lingologos/python"
fetch_python312() {
    # Last resort, and the one route that needs no sudo and no third party
    # archive: python-build-standalone publishes a self-contained CPython 3.12
    # for x86_64 Linux that runs from any directory. The URL is pinned to one
    # release rather than resolved, so a 404 means the pin aged and
    # LL_PYTHON_URL overrides it. That exact file ran here on 2026-11-09 and
    # installed openlane 2.3.10 with klayout 0.29.12 and libparse 0.3.1 from
    # wheels, recorded in doc/verification_report.md.
    command -v curl >/dev/null 2>&1 || return 1
    if [ ! -x "$STANDALONE_PYTHON_DIR/bin/python3" ]; then
        say "no python 3.12 from the package manager, fetching a standalone build"
        say "from $STANDALONE_PYTHON_URL"
        rm -rf "$STANDALONE_PYTHON_DIR" "$STANDALONE_PYTHON_DIR.partial"
        mkdir -p "$STANDALONE_PYTHON_DIR.partial" || return 1
        curl -sfL "$STANDALONE_PYTHON_URL" -o "$STANDALONE_PYTHON_DIR.partial/cpython.tar.gz" || return 1
        tar -C "$STANDALONE_PYTHON_DIR.partial" -xzf "$STANDALONE_PYTHON_DIR.partial/cpython.tar.gz" || return 1
        # The archive unpacks to one python directory at its top level.
        [ -d "$STANDALONE_PYTHON_DIR.partial/python" ] || return 1
        mv "$STANDALONE_PYTHON_DIR.partial/python" "$STANDALONE_PYTHON_DIR" || return 1
        rm -rf "$STANDALONE_PYTHON_DIR.partial"
    fi
    [ -x "$STANDALONE_PYTHON_DIR/bin/python3" ] || return 1
    LL_PYTHON="$STANDALONE_PYTHON_DIR/bin/python3"
    export LL_PYTHON
    say "interpreter obtained by this script: $("$LL_PYTHON" -V 2>&1) at $LL_PYTHON"
    return 0
}
select_python || NEED_APT=1
if [ "$NEED_APT" -eq 1 ] && [ "$RUN_SETUP" -eq 1 ]; then
    say "installing the build tools through apt"
    sudo apt-get update
    sudo apt-get install -y build-essential python3 python3-venv python3-pip python3-tk git curl make
fi
if [ "$RUN_SETUP" -eq 1 ] && [ -z "${LL_PYTHON:-}" ] && ! usable_python; then
    # Nothing with prebuilt wheels is installed. Ask apt for 3.12 first, one
    # command so a name the release does not carry does not cancel the rest, and
    # then ask the archive Ubuntu defers to for older interpreters. That PPA is
    # third party, and --no-ppa skips it: the standalone build below still runs.
    say "no python 3.12 or older found, the default here is $(python3 -V 2>&1)"
    sudo apt-get install -y python3.12 python3.12-venv python3.12-tk || true
    if [ "$USE_PPA" -eq 1 ] && ! usable_python; then
        say "adding ppa:deadsnakes/ppa, third party, for python3.12 and python3.12-venv"
        sudo apt-get install -y software-properties-common || true
        sudo add-apt-repository -y ppa:deadsnakes/ppa || true
        sudo apt-get update || true
        sudo apt-get install -y python3.12 python3.12-venv python3.12-tk || true
    fi
    usable_python || true
fi
if [ "$RUN_SETUP" -eq 1 ] && [ -z "${LL_PYTHON:-}" ] && ! usable_python; then
    # Neither the distribution nor the archive it defers to produced an
    # interpreter with wheels. Fetch one instead of stopping, and note what was
    # fetched: this is a download from outside the package manager.
    fetch_python312 || true
    select_python || true
fi
if [ "$RUN_SETUP" -eq 0 ]; then
    select_python || true
fi
[ -n "${PYBIN:-}" ] || die "no python interpreter usable for the virtual environment" 127
if too_new; then
    # Compiling both of them was tried on this machine and both failed, so a
    # newer interpreter is not a slower route, it is a dead end.
    echo "[FAIL] $("$PYBIN" -V 2>&1) has no prebuilt wheel for libparse 0.3.1, and its" >&2
    echo "[FAIL] source distribution does not build here. Install 3.12, then rerun:" >&2
    echo "[FAIL]   sudo apt-get install -y python3.12 python3.12-venv" >&2
    echo "[FAIL] Ubuntu 26.04 has no such package, so add the archive that does:" >&2
    echo "[FAIL]   sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update" >&2
    echo "[FAIL]   sudo apt-get install -y python3.12 python3.12-venv" >&2
    echo "[FAIL] Or point the script at an interpreter you obtained yourself:" >&2
    echo "[FAIL]   LL_PYTHON=/path/to/python3.12 sh asic_chip/scripts/setup_openlane_flow.sh" >&2
    echo "[FAIL] A standalone build of 3.12 is fetched when the archives fail; it failed" >&2
    echo "[FAIL] here too, and LL_PYTHON_URL replaces the pinned archive it downloads." >&2
    exit 127
fi
if ! "$PYBIN" -c 'import tkinter' 2>/dev/null; then
    # openlane 2.3.10 imports tkinter while starting up, in openlane/common/tcl.py,
    # and that import was seen failing on this machine with ModuleNotFoundError.
    # Debian and Ubuntu ship the module apart from the interpreter, so an
    # interpreter can be the right version and still miss it.
    PYVER="$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo '')"
    say "$("$PYBIN" -V 2>&1) has no tkinter, which openlane imports before it reads a design"
    if [ "$RUN_SETUP" -eq 1 ]; then
        sudo apt-get install -y "python${PYVER}-tk" python3-tk || true
    fi
    "$PYBIN" -c 'import tkinter' 2>/dev/null || die "still no tkinter for $("$PYBIN" -V 2>&1), install the tk package of that interpreter: python${PYVER:-3}-tk on Ubuntu" 127
fi
say "interpreter for the virtual environment: $("$PYBIN" -V 2>&1) ($PYBIN)"

# 3. openlane and volare live in their own virtual environment, so the system
# interpreter is never touched. One can sit on disk from an attempt on an
# interpreter pip cannot install into, and a virtual environment holds nothing
# but these packages, so such an environment is replaced rather than reused.
if [ "$RUN_SETUP" -eq 1 ] && [ -x "$VENV/bin/python3" ]; then
    if "$VENV/bin/python3" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)' 2>/dev/null; then
        say "replacing $VENV, built on $("$VENV/bin/python3" -V 2>&1), which pip cannot install openlane on"
        rm -rf "$VENV"
    fi
fi
if [ ! -x "$VENV/bin/openlane" ] && [ "$RUN_SETUP" -eq 1 ]; then
    say "creating $VENV and installing openlane plus volare"
    "$PYBIN" -m venv "$VENV"
    "$VENV/bin/python3" -m pip install --upgrade pip
    "$VENV/bin/python3" -m pip install openlane volare
fi
[ -x "$VENV/bin/openlane" ] || die "$VENV has no openlane, rerun without --flow-only" 127
PATH="$VENV/bin:$PATH"

# 4. A PDK version is a commit of open_pdks. Rather than hardcode one, ask volare
# for the versions it publishes and take the first one it lists, then install that
# one, so the download never guesses at a commit. The layout was read from the
# volare package installed here: versions live under
# $PDK_ROOT/volare/sky130/versions/<hash>, and enabling one symlinks each variant
# of the family, sky130A and sky130B, straight under $PDK_ROOT and records the
# version in $PDK_ROOT/volare/sky130/current. This flow names the variant sky130A,
# so enabling is enough, and the loop below only covers an older layout where the
# family was exposed as $PDK_ROOT/sky130 and no link for the variant existed.
if [ ! -e "$PDK_ROOT/$PDK" ]; then
    say "no $PDK under $PDK_ROOT, resolving the newest version volare lists"
    mkdir -p "$PDK_ROOT"
    # ls-remote is a subcommand of this volare, and the family is named with --pdk.
    # The --global search spelling older guides show is not a thing in volare 0.20:
    # it exits with "No such option '--global'", which a discarded stderr used to
    # turn into a message blaming the network. The error of the tool is left visible.
    VERSIONS="$("$VENV/bin/volare" ls-remote --pdk sky130 || true)"
    NEWEST="$(printf '%s\n' "$VERSIONS" | grep -E -o '[0-9a-f]{40}' | head -n 1)"
    [ -n "$NEWEST" ] || die "volare listed no sky130 version. What it printed above says why: a network that is not reachable, or a GitHub limit." 1
    # One PDK version is a set of libraries built from one open_pdks commit. The
    # default set, which is all the place and route of this design reads, measured
    # 2.1 GiB on disk here; -l all, which also carries the magic, netgen, klayout,
    # vs and xdrc variants of the same commit, measured 16 GiB for the version this
    # script installed. Nothing in this flow needs those, so the default is volare's
    # own and LL_PDK_LIBRARIES=all asks for the large set on purpose.
    PDK_FETCH_ARGS=""
    if [ -n "${LL_PDK_LIBRARIES:-}" ]; then
        PDK_FETCH_ARGS="-l $LL_PDK_LIBRARIES"
    fi
    say "downloading sky130 $NEWEST, about 2.1 GiB for the libraries this flow reads"
    "$VENV/bin/volare" fetch --pdk-root "$PDK_ROOT" --pdk sky130 $PDK_FETCH_ARGS "$NEWEST"
    "$VENV/bin/volare" enable --pdk-root "$PDK_ROOT" --pdk sky130 "$NEWEST" || true
fi
if [ ! -e "$PDK_ROOT/$PDK" ]; then
    for link_target in "$PDK_ROOT/sky130/$PDK" "$PDK_ROOT/volare/sky130/versions/${NEWEST:-}/$PDK"; do
        if [ -d "$link_target" ]; then
            say "exposing $PDK at $PDK_ROOT/$PDK"
            ln -s "$link_target" "$PDK_ROOT/$PDK"
            break
        fi
    done
fi
[ -e "$PDK_ROOT/$PDK" ] || die "$PDK_ROOT/$PDK is still missing. List $PDK_ROOT and link the variant it did create." 1
say "PDK ready: $PDK_ROOT/$PDK"

# 5. The smoke test is the upstream check that the container route works end to
# end, and it is the step that pulls the OpenLane image. It runs once per setup.
if [ "$RUN_SMOKE" -eq 1 ] && [ "$RUN_SETUP" -eq 1 ]; then
    say "smoke test, this is the step that downloads the OpenLane image"
    "$VENV/bin/python3" -m openlane --dockerized --smoke-test
fi

# 6. The flow itself. build_gds.sh keeps its own gate on the generated RTL, which
# is the check that stops a stale rtl/generated directory from being routed. The
# entry point looks for openlane on PATH, and this script never activates the
# virtual environment for the caller, so where the package was put is handed over
# by name instead.
if [ -x "$VENV/bin/openlane" ]; then
    export LL_OPENLANE_BIN="$VENV/bin/openlane"
fi
say "handing the design to build_gds.sh"
cd "$LL_REPO_ROOT"
sh asic_chip/scripts/build_gds.sh

say "done. The GDSII and the reports are under $ASIC_CHIP_DIR/openlane/runs/"
say "the OpenLane 2 run directory is named by the tool, not production_build."
