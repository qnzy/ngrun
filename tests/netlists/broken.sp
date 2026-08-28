* broken - deliberately unsimulatable netlist
*
* Every corner must still appear in the CSV, tagged SIM_ERROR rather than
* being silently dropped.

** ngr_param rr 1k 2k 4k
** ngr_out f3db

.param rr=1k

V1 in 0 DC 0 AC 1
R1 in out {rr}
Qbogus out in 0 no_such_model
C1 out 0 1n

.ac dec 10 1 1e6
.meas ac f3db when vdb(out) = -3.0103 cross=1

.end
