# 10 ns is an analysis/fitting constraint, not a measured timing result.
set CORELET_CLOCK_PERIOD_NS 10.000
create_clock -name clk -period $CORELET_CLOCK_PERIOD_NS [get_ports {clk}]
derive_clock_uncertainty

# External SRAM/controller timing and board pin assignments are not supplied.
# Add approved min/max input/output delays before claiming full I/O timing closure.
# Do not hide these unconstrained interfaces with blanket false-path assignments.
