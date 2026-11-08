#!/bin/sh
# -----------------------------------------------------------------------------
# Place and route entry point for the LingoLOGOS ASIC on SkyWater sky130A.
#
# This script is the declared flow. It has never been run in this checkout: the
# machine the RTL was written on has no OpenLane, no Yosys and no network access to
# fetch the PDK, and this script reports exactly that instead of pretending. The
# numbers in asic_chip/doc/datasheet_lingologos.md are therefore not post-layout
# numbers, and asic_chip/doc/verification_report.md says so in its own words.
#
# What running it really takes:
#   1. Linux, Docker, and roughly 12 GB of free disk.
#   2. OpenLane 1:   git clone -b 1.0.644 https://github.com/efabless/openlane \
#                    then make openlane and make openlane-pdk, or the equivalent
#                    OpenLane 2 install (openlane2 + sky130A via volare).
#   3. From the repository root:  sh asic_chip/scripts/build_gds.sh
#   4. Result lands in asic_chip/openlane/runs/production_build/, with the final
#      GDSII at 6_final.gds and the timing report at 6_final/reports/report.csv.
# -----------------------------------------------------------------------------

set -eu

ASIC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DESIGN_DIR="$ASIC_DIR/openlane"
TAG="production_build"

echo "LingoLOGOS OpenLane entry point"
echo "  design directory: $DESIGN_DIR"
echo "  PDK: sky130A, standard cell library: sky130_fd_sc_hd"
echo "  clock budget: 250 MHz, T_clk 4.0 ns, declared in openlane/lingologos.sdc"

if [ ! -f "$DESIGN_DIR/config.json" ]; then
    echo "[FAIL] missing $DESIGN_DIR/config.json"
    exit 1
fi

# The generated blocks must exist and must be the current ones before a layout is
# attempted. The --check mode of the generator recomputes every file and compares the
# bytes, so a stale rtl/generated directory stops the flow here.
python3 "$ASIC_DIR/scripts/lingologos_compiler.py" --check

if command -v flow.tcl >/dev/null 2>&1; then
    echo "[INFO] OpenLane 1 detected: $TAG"
    flow.tcl -design "$DESIGN_DIR" -tag "$TAG" -overwrite
elif [ -n "${OPENLANE_ROOT:-}" ] && [ -f "$OPENLANE_ROOT/flow.tcl" ]; then
    echo "[INFO] OpenLane 1 detected at $OPENLANE_ROOT/flow.tcl"
    "$OPENLANE_ROOT/flow.tcl" -design "$DESIGN_DIR" -tag "$TAG" -overwrite
elif command -v openlane >/dev/null 2>&1; then
    # OpenLane 2 takes the configuration as an argument and renames the step outputs.
    # The keys in config.json are OpenLane 1 names; the OpenLane 2 migration maps the
    # ones this design uses (design, verilog_files, clock port and period, sdc,
    # pin_order_cfg, fp_core_util, pl_target_density) onto their OpenLane 2 spellings.
    echo "[INFO] OpenLane 2 detected"
    openlane "$DESIGN_DIR/config.json"
else
    echo "[FAIL] neither flow.tcl nor openlane was found on PATH."
    echo "[FAIL] No place and route has been run for this design in this checkout."
    echo "[FAIL] Install OpenLane as described in asic_chip/doc/build_instructions.md,"
    echo "[FAIL] then re-run this script. Nothing about the layout is asserted until then."
    exit 127
fi

echo "[INFO] done. Inspect:"
echo "  $DESIGN_DIR/runs/$TAG/final/*.gds"
echo "  $DESIGN_DIR/runs/$TAG/reports/report.csv"
echo "  $DESIGN_DIR/runs/$TAG/logs/"
