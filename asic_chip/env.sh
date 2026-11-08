# Sourcing this file puts the LingoLOGOS ASIC tools on the path and sets the
# variables the OpenLane flow expects. It is a POSIX shell script, meant for the
# Linux container described in asic_chip/doc/build_instructions.md; on Windows run
# the same commands inside WSL or a Docker container.
#
#   source asic_chip/env.sh
#
# Nothing here installs anything. It only records the versions the flow was written
# against, so a build that fails does so against known variables rather than guessing.

ASIC_CHIP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ASIC_CHIP_DIR

# OpenLane. The flow configuration uses OpenLane 1 variable names (flow.tcl). OpenLane 2
# keeps those names in its Classic flow but has a different command line, and
# asic_chip/scripts/build_gds.sh detects which one is present. The image names below are the
# ones published on the registries at the time of writing, looked up over the network and
# never run from this checkout: efabless/openlane:superstable and efabless/openlane:1.0.0
# for OpenLane 1, ghcr.io/efabless/openlane2:2.0.4-amd64 for OpenLane 2.
export OPENLANE_ROOT="${OPENLANE_ROOT:-$HOME/openlane}"
export OPENLANE_VERSION="${OPENLANE_VERSION:-superstable}"

# SkyWater PDK. The design targets sky130A with the high density standard cells.
export PDK_ROOT="${PDK_ROOT:-$HOME/pdk}"
export PDK=sky130A
export STD_CELL_LIBRARY=sky130_fd_sc_hd

# Simulation tools, with the versions the harness expects.
#   Verilator >= 5.006  (the harness does not call the removed final() method)
#   Icarus Verilog >= 11  (SystemVerilog 2012 for the testbench)
#   Yosys >= 0.38       (SystemVerilog read only memory inference)
export VERILATOR_THREADS="${VERILATOR_THREADS:-4}"

# The generated RTL and the vector tables are addressed from the repository root,
# so every command in the Makefile is written as if the current directory were the
# checkout root.
export LL_REPO_ROOT="$(cd "$ASIC_CHIP_DIR/.." && pwd)"

alias ll-gen="python3 $LL_REPO_ROOT/asic_chip/scripts/lingologos_compiler.py"
alias ll-check="python3 $LL_REPO_ROOT/asic_chip/scripts/lingologos_compiler.py --check"

echo "LingoLOGOS ASIC environment"
echo "  repo root:    $LL_REPO_ROOT"
echo "  asic_chip:    $ASIC_CHIP_DIR"
echo "  PDK_ROOT:     $PDK_ROOT  (PDK=$PDK, library=$STD_CELL_LIBRARY)"
echo "  OPENLANE_ROOT: $OPENLANE_ROOT"
echo "Run: make -f asic_chip/Makefile gen && make -f asic_chip/Makefile sim-verilator"
