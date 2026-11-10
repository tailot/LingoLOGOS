#!/bin/sh
# -----------------------------------------------------------------------------
# Place and route entry point for the LingoLOGOS ASIC on SkyWater sky130A.
#
# This script is the declared flow. It has run in this checkout, on 2026-11-09, inside
# Ubuntu on WSL 2. The first run stopped at the 25th step of the OpenLane 2 Classic flow,
# on the pin order file; after that file was rewritten in the grammar the tool actually
# reads, the second run went through all 78 steps in 16 min 59 s, wrote a GDSII, passed
# DRC with zero violations and LVS with "Circuits match uniquely", and then exited non
# zero on deferred setup and hold errors, which is the timing gate doing its job at a
# period the layout did not close. What that run measured, and what was changed in the
# constraint sets because of it, is written in asic_chip/doc/verification_report.md and
# asic_chip/doc/build_instructions.md; the layout numbers are in
# asic_chip/doc/datasheet_lingologos.md.
#
# What running it really takes:
#   1. Linux or WSL 2 with Docker, and free disk for the tool image plus one PDK build.
#      The upstream documentation puts a PDK build at 700 MiB to 1.5 GiB per version.
#      Measured here: 7.39 GB of docker disk for the openlane2 2.3.10 image, 2.1 GiB for
#      one sky130 version with the library set this flow reads, and 16 GiB for that same
#      version with -l all.
#   2. OpenLane 2 through pip inside a virtual environment, or an OpenLane 1 checkout
#      whose flow.tcl is on PATH. openlane itself is looked for in $LL_OPENLANE_BIN
#      first, then in $HOME/openlane-venv/bin, then on PATH, because a package in a
#      virtual environment is not on PATH unless the caller activated that
#      environment. One found in such an environment is invoked with --dockerized: the
#      package brings no tools, the container image does, and that is the image the
#      smoke test of the setup downloads. PDK_ROOT names the PDK, and $HOME/pdk, the
#      location the setup uses, is the fallback when nothing was named. The one
#      command that installs all of that is asic_chip/scripts/setup_openlane_flow.sh,
#      documented in asic_chip/doc/build_instructions.md.
#   3. From the repository root:  sh asic_chip/scripts/build_gds.sh
#   4. With OpenLane 1 the result lands in asic_chip/openlane/runs/production_build/,
#      because this script passes -tag production_build. OpenLane 2 has no such tag: the
#      runs observed here created asic_chip/openlane/runs/RUN_2026-11-09_16-15-02,
#      asic_chip/openlane/runs/RUN_2026-11-09_18-43-18 and
#      asic_chip/openlane/runs/RUN_2026-11-09_21-14-43, directories of their own
#      name under runs/ next to the configuration file they were given. OpenLane 2 writes
#      no results directory either: every output sits in the directory of the step that
#      made it, and the step numbering moves between runs because the number of enabled
#      steps changes. The 8.0 ns run wrote its GDSII files in 56-magic-streamout and
#      57-klayout-streamout and its signoff summary in 74-misc-reportmanufacturability;
#      the 30.0 ns run wrote them in 59-magic-streamout, 60-klayout-streamout and
#      77-misc-reportmanufacturability. The script therefore searches for the GDSII files
#      and prints the newest run directory under that runs directory, or under the
#      working directory, together with the GDSII files inside it.
# -----------------------------------------------------------------------------

set -eu

ASIC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DESIGN_DIR="$ASIC_DIR/openlane"
TAG="production_build"
FLOW_STATUS=0

# Constraint set selection. config.json carries the declared budget, 250 MHz at
# 4.0 ns. That budget has now been measured against twice. At the 8.0 ns gate this
# script used to ask for, the complete run was 4.4409 ns short in max_tt_025C_1v80 and
# 15.7205 ns short in max_ss_100C_1v60. At the 30.0 ns gate it was 0.4906 ns short in
# max_ss_100C_1v60 only, RUN_2026-11-09_21-14-43, so the bring-up period is now 32.0 ns,
# that slow-corner figure plus room for the rerouting a different period causes. A run at
# 4.0 ns is still expected to stop at the timing gate without a clean exit, and that is
# a result and not a crash: OpenLane 2 fails on the corners named by
# SETUP_VIOLATION_CORNERS and HOLD_VIOLATION_CORNERS, and both now name all corners. That
# same run passed Antenna, LVS and DRC and still exited non zero, on 10 [hold in-reg]
# paths whose worst slack was -0.1540 ns; PL_RESIZER_HOLD_SLACK_MARGIN and
# GRT_RESIZER_HOLD_SLACK_MARGIN are set to 0.40 ns in both constraint sets for that.
# The default is the bring-up set; LL_OPENLANE_CONFIG=config.json routes at the declared
# budget, and LL_BRINGUP_PERIOD overrides the bring-up period for the OpenLane 1
# override below.
FLOW_CONFIG="${LL_OPENLANE_CONFIG:-config.bringup.json}"
BRINGUP_PERIOD="${LL_BRINGUP_PERIOD:-32.0}"
case "$FLOW_CONFIG" in
    config.json)
        FLOW_CONFIG="config.json"
        MODE=spec
        ;;
    *)
        FLOW_CONFIG="config.bringup.json"
        MODE=bringup
        ;;
esac

if [ -z "${PDK_ROOT:-}" ] && [ -d "$HOME/pdk/sky130A" ]; then
    # The one command setup keeps the PDK there and exports the variable for the run
    # it starts itself. This script run on its own has no way of knowing that, so the
    # same location is the default when nothing was named.
    PDK_ROOT="$HOME/pdk"
    export PDK_ROOT
fi

echo "LingoLOGOS OpenLane entry point"
echo "  design directory: $DESIGN_DIR"
echo "  PDK: sky130A, standard cell library: sky130_fd_sc_hd"
echo "  PDK_ROOT: ${PDK_ROOT:-unset, and no $HOME/pdk to fall back on either}"
if [ "$MODE" = bringup ]; then
    echo "  constraint set: $FLOW_CONFIG, T_clk $BRINGUP_PERIOD ns, bring-up only"
    echo "  declared spec stays 250 MHz at 4.0 ns in config.json and lingologos.sdc"
    echo "  set LL_OPENLANE_CONFIG=config.json to route at the declared budget"
else
    echo "  constraint set: config.json, T_clk 4.0 ns, the declared 250 MHz budget"
fi

if [ ! -f "$DESIGN_DIR/config.json" ]; then
    echo "[FAIL] missing $DESIGN_DIR/config.json"
    exit 1
fi

if [ "$MODE" = bringup ] && { [ ! -f "$DESIGN_DIR/$FLOW_CONFIG" ] || [ ! -f "$DESIGN_DIR/lingologos.bringup.sdc" ]; }; then
    echo "[FAIL] the bring-up constraint set is incomplete: $FLOW_CONFIG and"
    echo "[FAIL] lingologos.bringup.sdc must both exist in $DESIGN_DIR"
    exit 1
fi

# The generated blocks must exist and must be the current ones before a layout is
# attempted. The --check mode of the generator recomputes every file and compares the
# bytes, so a stale rtl/generated directory stops the flow here.
python3 "$ASIC_DIR/scripts/lingologos_compiler.py" --check

run_openlane1() {
    # OpenLane 1 always reads the config.json of the design directory, so the
    # bring-up period cannot be selected by naming another file: it is passed as an
    # override of the two keys that differ, the period and the constraint file.
    if [ "$MODE" = bringup ]; then
        "$1" -design "$DESIGN_DIR" -tag "$TAG" -overwrite \
            -override_configs "CLOCK_PERIOD=$BRINGUP_PERIOD SDC=dir::lingologos.bringup.sdc"
    else
        "$1" -design "$DESIGN_DIR" -tag "$TAG" -overwrite
    fi
}

find_openlane2() {
    # The one command setup installs openlane as a package in a virtual environment
    # it does not activate, so PATH alone does not find it: it hands the location
    # over as LL_OPENLANE_BIN, and that environment is also searched by name in case
    # the variable was not set by the caller.
    for candidate in "${LL_OPENLANE_BIN:-}" "$HOME/openlane-venv/bin/openlane" "$(command -v openlane 2>/dev/null || true)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}
OPENLANE2_BIN="$(find_openlane2 || true)"

if command -v flow.tcl >/dev/null 2>&1; then
    echo "[INFO] OpenLane 1 detected: $TAG"
    FLOW_KIND=1
    run_openlane1 flow.tcl
elif [ -n "${OPENLANE_ROOT:-}" ] && [ -f "$OPENLANE_ROOT/flow.tcl" ]; then
    echo "[INFO] OpenLane 1 detected at $OPENLANE_ROOT/flow.tcl"
    FLOW_KIND=1
    run_openlane1 "$OPENLANE_ROOT/flow.tcl"
elif [ -n "$OPENLANE2_BIN" ]; then
    # OpenLane 2 takes the configuration as an argument and renames the step outputs.
    # The keys in the constraint files are OpenLane 1 names, which OpenLane 2 keeps in its
    # Classic flow. That mapping has now run end to end on this design: all 78 steps of
    # the Classic flow executed. Three things it taught are worth keeping in mind here.
    # A key OpenLane 2 no longer has is reported and ignored, so QUIT_ON_TIMING_VIOLATIONS
    # and RUN_CVC are gone from the constraint files rather than silently doing nothing.
    # The OpenLane 1 DIODE_INSERTION_STRATEGY is accepted and then overwrites the native
    # OpenLane 2 antenna knobs, so it is absent too. And set -eu means a run that fails a
    # signoff checker stops this script with the tool exit code, which is how the first
    # complete run reported its timing result.
    # A package installed by the one command setup carries no tools of its own: it runs
    # them from the container image, which is what --dockerized asks for and what the
    # smoke test of the setup pulled. An openlane found on PATH is left in the mode its
    # own installation chose.
    echo "[INFO] OpenLane 2 detected at $OPENLANE2_BIN, configuration $FLOW_CONFIG"
    FLOW_KIND=2
    # The status is captured instead of raised so that the run directory and its reports
    # are still printed when a signoff checker failed: the deferred errors of a checker
    # step are the result of the run, and the useful part of it is on disk either way.
    if [ "$OPENLANE2_BIN" = "${LL_OPENLANE_BIN:-}" ] || [ "$OPENLANE2_BIN" = "$HOME/openlane-venv/bin/openlane" ]; then
        "$OPENLANE2_BIN" --dockerized "$DESIGN_DIR/$FLOW_CONFIG" || FLOW_STATUS=$?
    else
        "$OPENLANE2_BIN" "$DESIGN_DIR/$FLOW_CONFIG" || FLOW_STATUS=$?
    fi
else
    echo "[FAIL] no flow.tcl, and no openlane in \$LL_OPENLANE_BIN, in \$HOME/openlane-venv/bin or on PATH."
    echo "[FAIL] A place and route has run for this design in this checkout, in the runs"
    echo "[FAIL] directory next to the configuration file, but no OpenLane is reachable now."
    echo "[FAIL] Install OpenLane as described in asic_chip/doc/build_instructions.md,"
    echo "[FAIL] then re-run this script. Nothing about a new layout is asserted until then."
    exit 127
fi

if [ "$FLOW_KIND" -eq 1 ]; then
    echo "[INFO] done. Inspect:"
    echo "  $DESIGN_DIR/runs/$TAG/final/*.gds"
    echo "  $DESIGN_DIR/runs/$TAG/reports/report.csv"
    echo "  $DESIGN_DIR/runs/$TAG/logs/"
else
    # OpenLane 2 names the run directory itself. The runs observed here wrote it under
    # runs/ next to the configuration file, which is where their own logs said every step
    # was, and a runs directory under the working directory is the other place it can
    # appear. Newest first is how the tool numbers them. OpenLane 2 writes no results
    # directory: the GDSII files sit in the directories of the steps that made them, the
    # magic and klayout streamout steps, and the pass or fail summary of the signoff
    # checkers is the manufacturability report of the last step.
    newest=""
    for runs_in in "$DESIGN_DIR/runs" "$PWD/runs"; do
        newest="$(ls -1dt "$runs_in"/*/ 2>/dev/null | head -n 1)"
        if [ -n "$newest" ]; then
            break
        fi
    done
    echo "[INFO] done, flow exit status $FLOW_STATUS. OpenLane 2 chose its own run directory:"
    echo "  ${newest:-no runs directory under $DESIGN_DIR or $PWD yet}"
    if [ -n "$newest" ]; then
        echo "[INFO] GDSII files under it, if the flow reached the streamout steps:"
        find "$newest" -maxdepth 5 -name '*.gds' 2>/dev/null | head -n 20
        echo "[INFO] summaries worth opening:"
        for summary in manufacturability.rpt error.log flow.log; do
            found="$(find "$newest" -maxdepth 2 -name "$summary" 2>/dev/null | head -n 1)"
            if [ -n "$found" ]; then
                echo "  $found"
            fi
        done
    fi
fi

# -----------------------------------------------------------------------------
# Artifact export, the step the board level artifact specification asks for after
# the flow: the factory tree (GDSII, LEF, DEF, SDC, the signoff reports) and the
# board tree (gate level netlists, the node netlist and the board wrapper) are
# written to export_artifacts/, outside every versioned source and ignored by
# .gitignore. The exporter looks for the newest run under the runs directory next
# to the configuration file, which is the directory named above, so it needs no
# path from here. When the flow did not exit cleanly the export is asked to
# tolerate missing artifacts: a run stopped by a signoff checker never reached the
# streamout steps, and the reports it did write are exactly the part worth keeping.
EXPORT_STATUS=0
PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "[WARN] no python3 on PATH, so the artifact export did not run."
    echo "[WARN] Rerun it by hand when Python is available:"
    echo "[WARN]   python3 asic_chip/scripts/export_artifacts.py"
elif [ "$FLOW_KIND" -eq 1 ] && [ ! -d "$DESIGN_DIR/runs/$TAG" ]; then
    echo "[WARN] no run directory at $DESIGN_DIR/runs/$TAG, so the artifact export did not run."
else
    if [ "$FLOW_STATUS" -ne 0 ]; then
        echo "[INFO] exporting artifacts of the newest run, tolerating the outputs a stopped flow never wrote"
        "$PYTHON_BIN" "$ASIC_DIR/scripts/export_artifacts.py" --allow-missing || EXPORT_STATUS=$?
    else
        echo "[INFO] exporting artifacts of the newest run"
        "$PYTHON_BIN" "$ASIC_DIR/scripts/export_artifacts.py" || EXPORT_STATUS=$?
    fi
fi

if [ "$FLOW_STATUS" -ne 0 ]; then
    echo "[INFO] the flow did not exit cleanly, status $FLOW_STATUS. A deferred error from a"
    echo "[INFO] signoff checker is the reported result of the run, not a crash: read the"
    echo "[INFO] manufacturability report named above and the corner tables in the step"
    echo "[INFO] directory of the static timing analysis after place and route."
fi
if [ "$FLOW_STATUS" -eq 0 ] && [ "$EXPORT_STATUS" -ne 0 ]; then
    echo "[FAIL] the flow exited cleanly but the artifact export did not, status"
    echo "[FAIL] $EXPORT_STATUS: an artifact the export specification requires was not found in the"
    echo "[FAIL] run directory, so the export tree is incomplete. The manifest written next to it"
    echo "[FAIL] names every entry and every miss, and the exporter has no other reason to fail."
    exit "$EXPORT_STATUS"
fi
exit "$FLOW_STATUS"

