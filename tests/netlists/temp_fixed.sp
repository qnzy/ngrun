* temp_fixed - the netlist owns its .temp, and there is no ngr_temp directive
*
* Regression guard: ngrun must not rewrite or inject .temp when no
* temperature sweep was requested.  The measured f3db pins down the
* temperature actually used - at 125 C the tempco makes R 9.8% larger than
* at 27 C, which is far outside measurement tolerance.

** ngr_param cc 1n
** ngr_out f3db

.temp 125

.param cc=1n
V1 in 0 DC 0 AC 1
R1 in out 1k tc1=1e-3
C1 out 0 {cc}

.ac dec 500 1 1e9
.meas ac f3db when v(out) = 0.5 cross=1

.end
