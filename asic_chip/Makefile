# -----------------------------------------------------------------------------
# LingoLOGOS ASIC build flow.
#
# Run from the repository root, because the paths in the vector tables and in this
# file are written relative to it:
#
#   make -f asic_chip/Makefile gen        regenerate the RTL and the vectors
#   make -f asic_chip/Makefile check      prove the checked in files are reproducible
#   make -f asic_chip/Makefile sim-verilator   C++ harness, Verilator, VCD dump
#   make -f asic_chip/Makefile sim-tb          SystemVerilog testbench, Verilator
#   make -f asic_chip/Makefile sim-topology    SystemVerilog testbench of the topology switch
#   make -f asic_chip/Makefile sim-iverilog    same checks in Icarus Verilog
#   make -f asic_chip/Makefile lint            Verilator lint, no simulation
#   make -f asic_chip/Makefile lint-topology   Verilator lint of the topology switch alone
#   make -f asic_chip/Makefile pytest          the Python gate suite
#   make -f asic_chip/Makefile openlane        place and route at the 32.0 ns bring-up gate
#   make -f asic_chip/Makefile openlane-spec   the same flow at the declared 4.0 ns budget
#   make -f asic_chip/Makefile setup-openlane  one command: OpenLane plus sky130A setup, then the flow
#   make -f asic_chip/Makefile clean
#
# Every target that needs a tool reports it plainly if the tool is missing. On
# the machine this RTL was written on, Python and pytest are installed while
# Verilator, Icarus, Yosys and OpenLane are not, so lint, sim-tb, sim-verilator,
# lint-topology and sim-topology were run inside the container image
# verilator/verilator:latest (Verilator 5.054, which carries make and a C++
# compiler) while sim-iverilog and openlane were never run at all. The topology
# targets were run there on 2026-11-10 and their output is recorded in section
# 2.4.1 of asic_chip/doc/verification_report.md, together with the commands that
# produced it.
# -----------------------------------------------------------------------------

PYTHON ?= python3
ASIC := asic_chip
OBJDIR := $(ASIC)/sim/obj_dir
VCD := $(ASIC)/sim/lingologos.vcd

RTL_GENERATED := $(ASIC)/rtl/generated/weight_rom_mesh.sv \
                 $(ASIC)/rtl/generated/gf_automaton_1cycle.sv \
                 $(ASIC)/rtl/generated/smt_evaluator.sv
RTL_CORE := $(ASIC)/rtl/core/rv32e_regfile.sv $(ASIC)/rtl/core/rv32e_core.sv
RTL_TOP := $(ASIC)/rtl/top/lingologos_top.sv
ALL_RTL := $(RTL_GENERATED) $(RTL_CORE) $(RTL_TOP)

# The topology switch stands on its own: it is not instantiated in the top level and it
# is not in the source list of the chip configuration, so it is linted and simulated by
# its own targets and no synthesis run sees it.
RTL_TOPOLOGY := $(ASIC)/rtl/core/lingologos_topology_router.sv
TB_TOPOLOGY := $(ASIC)/sim/topology_testbench.sv

# LL_DEBUG_PORTS exists only for simulation. Synthesis and OpenLane never see it,
# so the dbg_ group never becomes a pad.
VDEF := -DLL_DEBUG_PORTS
VOPT := -O2 --timescale 1ns/1ps -Wall -Wno-UNSIGNED -Wno-UNUSED -Wno-WIDTHTRUNC

.DEFAULT_GOAL := help

.PHONY: help gen check sim-verilator sim-tb sim-topology sim-iverilog lint lint-topology pytest openlane openlane-spec setup-openlane clean

help:
	@echo "LingoLOGOS ASIC targets: gen check sim-verilator sim-tb sim-topology sim-iverilog lint lint-topology pytest openlane openlane-spec setup-openlane clean"

# The generator is the only place the datapath is described. It rebuilds the three
# generated modules, the golden vector table and the core self-test table from the
# artifacts, and it gates its own output.
gen:
	$(PYTHON) $(ASIC)/scripts/lingologos_compiler.py

check:
	$(PYTHON) $(ASIC)/scripts/lingologos_compiler.py --check

# C++ harness required by section 8 of the specification. The model is built with
# --exe and --build, then run from the repository root so the relative paths in
# sim_main.cpp resolve.
sim-verilator: gen
	@command -v verilator >/dev/null 2>&1 || { echo "verilator not found, see asic_chip/doc/build_instructions.md"; exit 127; }
	verilator --cc --exe --build $(VOPT) $(VDEF) --trace \
	  --top-module lingologos_top -Mdir $(OBJDIR) \
	  $(ALL_RTL) $(ASIC)/sim/sim_main.cpp
	$(OBJDIR)/Vlingologos_top $(ASIC)/sim/golden_vectors.mem $(ASIC)/sim/core_self_test.mem $(VCD)

# The SystemVerilog testbench does the same checks without C++, which is the path to
# use when only a simulator binary is wanted.
sim-tb: gen
	@command -v verilator >/dev/null 2>&1 || { echo "verilator not found, see asic_chip/doc/build_instructions.md"; exit 127; }
	verilator --binary --timing $(VOPT) $(VDEF) \
	  --top-module lingologos_testbench -Mdir $(OBJDIR)_tb \
	  $(ALL_RTL) $(ASIC)/sim/testbench.sv
	$(OBJDIR)_tb/Vlingologos_testbench

# The topology switch needs no generated RTL, so this target has no gen prerequisite:
# the four dies it wires together are the switch itself, four times.
sim-topology:
	@command -v verilator >/dev/null 2>&1 || { echo "verilator not found, see asic_chip/doc/build_instructions.md"; exit 127; }
	verilator --binary --timing $(VOPT) $(VDEF) \
	  --top-module lingologos_topology_testbench -Mdir $(OBJDIR)_topology \
	  $(RTL_TOPOLOGY) $(TB_TOPOLOGY)
	$(OBJDIR)_topology/Vlingologos_topology_testbench

sim-iverilog: gen
	@command -v iverilog >/dev/null 2>&1 || { echo "iverilog not found, see asic_chip/doc/build_instructions.md"; exit 127; }
	mkdir -p $(OBJDIR)_vvp
	iverilog -g2012 -D LL_DEBUG_PORTS -s lingologos_testbench \
	  -o $(OBJDIR)_vvp/lingologos.vvp $(ALL_RTL) $(ASIC)/sim/testbench.sv
	vvp $(OBJDIR)_vvp/lingologos.vvp

lint: lint-topology
	@command -v verilator >/dev/null 2>&1 || { echo "verilator not found, see asic_chip/doc/build_instructions.md"; exit 127; }
	verilator --lint-only $(VOPT) $(VDEF) --top-module lingologos_top $(ALL_RTL)
	@echo "lint clean"

# The switch alone, module and testbench, because the lint of the top level above never
# pulls it in: the module is not instantiated in lingologos_top.sv.
lint-topology:
	@command -v verilator >/dev/null 2>&1 || { echo "verilator not found, see asic_chip/doc/build_instructions.md"; exit 127; }
	verilator --lint-only $(VOPT) $(VDEF) --top-module lingologos_topology_router $(RTL_TOPOLOGY)
	verilator --lint-only $(VOPT) $(VDEF) --top-module lingologos_topology_testbench $(RTL_TOPOLOGY) $(TB_TOPOLOGY)
	@echo "topology switch lint clean"

pytest:
	$(PYTHON) -m pytest -v tests/

# The script behind this target has run in this checkout, on 2026-11-09, in WSL Ubuntu with
# openlane 2.3.10 and the sky130A PDK, though not through this target: three times, the last
# as RUN_2026-11-09_21-14-43, which went through all 78 steps of the Classic flow in 12 min
# 12 s and exited non zero on deferred setup and hold violations after reporting Antenna
# Passed, LVS Passed and DRC Passed. build_gds.sh detects OpenLane 1 or 2, reports which one
# it found, and stops if neither is reachable. It runs the 32.0 ns bring-up constraint set by
# default, a period taken from the slack those runs measured, because the declared 4.0 ns
# budget has never been routed.
openlane: gen
	sh $(ASIC)/scripts/build_gds.sh

# The same flow at the declared 250 MHz budget, 4.0 ns. Never run: at the 8.0 ns gate of the
# run of 2026-11-09 the same layout missed setup by 4.4409 ns in the worst typical corner and
# by 15.7205 ns in the worst slow one, so expect this to stop at the timing gate until the
# critical path has been reworked, not rerouted.
openlane-spec: gen
	LL_OPENLANE_CONFIG=config.json sh $(ASIC)/scripts/build_gds.sh

# One command that installs what the flow needs and then runs it. This is the POSIX
# route, valid on Linux and inside WSL: the distribution itself, Docker Desktop and
# its WSL integration cannot be installed or started from inside the distribution,
# so from Windows drive the same sequence with the PowerShell twin,
# asic_chip/scripts/setup_openlane_flow.ps1, run from the Windows shell.
setup-openlane:
	sh $(ASIC)/scripts/setup_openlane_flow.sh

clean:
	rm -rf $(OBJDIR) $(OBJDIR)_tb $(OBJDIR)_topology $(OBJDIR)_vvp
	rm -f $(VCD) $(ASIC)/sim/*.vvp
	rm -rf $(ASIC)/openlane/runs $(ASIC)/openlane/*.log
	@echo "removed simulation and OpenLane build products"
