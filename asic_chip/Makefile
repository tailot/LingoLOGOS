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
#   make -f asic_chip/Makefile sim-iverilog    same checks in Icarus Verilog
#   make -f asic_chip/Makefile lint            Verilator lint, no simulation
#   make -f asic_chip/Makefile pytest          the Python gate suite
#   make -f asic_chip/Makefile openlane        declared place and route run
#   make -f asic_chip/Makefile clean
#
# Every target that needs a tool reports it plainly if the tool is missing. On
# the machine this RTL was written on, Python and pytest are installed while
# Verilator, Icarus, Yosys and OpenLane are not, so lint, sim-tb and
# sim-verilator were run inside the container image verilator/verilator:latest
# (Verilator 5.054, which carries make and a C++ compiler) and sim-iverilog and
# openlane were never run at all. Section 2.4 of asic_chip/doc/
# verification_report.md records those commands and their output.
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

# LL_DEBUG_PORTS exists only for simulation. Synthesis and OpenLane never see it,
# so the dbg_ group never becomes a pad.
VDEF := -DLL_DEBUG_PORTS
VOPT := -O2 --timescale 1ns/1ps -Wall -Wno-UNSIGNED -Wno-UNUSED -Wno-WIDTHTRUNC

.DEFAULT_GOAL := help

.PHONY: help gen check sim-verilator sim-tb sim-iverilog lint pytest openlane clean

help:
	@echo "LingoLOGOS ASIC targets: gen check sim-verilator sim-tb sim-iverilog lint pytest openlane clean"

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

sim-iverilog: gen
	@command -v iverilog >/dev/null 2>&1 || { echo "iverilog not found, see asic_chip/doc/build_instructions.md"; exit 127; }
	mkdir -p $(OBJDIR)_vvp
	iverilog -g2012 -D LL_DEBUG_PORTS -s lingologos_testbench \
	  -o $(OBJDIR)_vvp/lingologos.vvp $(ALL_RTL) $(ASIC)/sim/testbench.sv
	vvp $(OBJDIR)_vvp/lingologos.vvp

lint:
	@command -v verilator >/dev/null 2>&1 || { echo "verilator not found, see asic_chip/doc/build_instructions.md"; exit 127; }
	verilator --lint-only $(VOPT) $(VDEF) --top-module lingologos_top $(ALL_RTL)
	@echo "lint clean"

pytest:
	$(PYTHON) -m pytest -v tests/

# Declared, not executed in this checkout. build_gds.sh detects OpenLane 1 or 2,
# reports which one it found, and stops if neither is installed.
openlane: gen
	sh $(ASIC)/scripts/build_gds.sh

clean:
	rm -rf $(OBJDIR) $(OBJDIR)_tb $(OBJDIR)_vvp
	rm -f $(VCD) $(ASIC)/sim/*.vvp
	rm -rf $(ASIC)/openlane/runs $(ASIC)/openlane/*.log
	@echo "removed simulation and OpenLane build products"
