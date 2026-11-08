# -----------------------------------------------------------------------------
# Timing constraints for the LingoLOGOS ASIC, sky130A, sky130_fd_sc_hd.
#
# The specification clock budget is 250 MHz, so T_clk is 4.0 ns. This file states
# that budget and the input and output budgets that go with it. It does not claim
# that the design meets it: no static timing analysis has been run in this checkout,
# and 250 MHz on sky130_fd_sc_hd is optimistic for this RTL. See
# asic_chip/doc/verification_report.md for what is and is not measured.
# -----------------------------------------------------------------------------

# Clock: 250 MHz period, single source at the pad.
create_clock -name clk -period 4.0 [get_ports clk]

# Setup and hold uncertainty. Held conservative because no clock tree has been
# characterized for this floorplan yet.
set_clock_uncertainty -setup 0.15 [get_clocks clk]
set_clock_uncertainty -hold 0.05 [get_clocks clk]

# Disabled clock transitions inside the power-down domain, kept as a placeholder for
# the production power intent; nothing in this RTL is power gated.
set_clock_transition 0.25 [get_clocks clk]

# Input budget: 20 percent of the period is available for external logic, and the
# pad ring owns the rest.
set_input_delay  -clock clk -max 0.80 [all_inputs]
set_input_delay  -clock clk -min 0.30 [all_inputs]

# Output budget: 20 percent of the period, measured at the pad boundary.
set_output_delay -clock clk -max 0.80 [all_outputs]
set_output_delay -clock clk -min 0.30 [all_outputs]

# Drive and load assumptions for the pad interface.
set_driving_cell -lib_cell sky130_fd_sc_hd__buf_4 -pin X [all_inputs]
set_load 0.030 [all_outputs]

# Internal loading on the mesh fanout: each of the 16 mesh cells multiplies an 8 bit
# input by a constant, so the input cones are wide.
set_load 0.025 [get_ports {in_vector_*}]

# The GF transition table is one level of a 1020 entry lookup; it is the shortest path
# in the design and gets no exception, it is checked like everything else.

# Asynchronous reset: released asynchronously, recovered synchronously by the core.
# Declared as a false path, which is what the RTL assumes: the reset is asserted for
# many cycles and is never a path that has to close inside one period.
set_false_path -from [get_ports rst_n]

# core_start is a level that is held stable for many cycles, so it is not a
# critical path and needs no exception either.

# No multicycle paths are declared. Every path in the design is single cycle by
# construction: the mesh cone, the GF lookup, the clause network and the core.
