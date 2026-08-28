* ctrl - .control block, derived measurement, and deliberate name shadowing
*
* The .control block prints 'f3db = 0' BEFORE the real measurement is taken.
* ngr_out must report the measured value, not the decoy: this is the
* regression guard for last-match extraction.  gain_ratio is derived from a
* measurement via let/print, and the bare 'write' exercises the injected
* 'set rawfile'.

** ngr_param rr 1k 2k
** ngr_out f3db gain_ratio

.param rr=1k cc=1n

V1 in 0 DC 0 AC 1
R1 in out {rr}
C1 out 0 {cc}

.control
let f3db = 0
print f3db
ac dec 200 1 1e9
meas ac f3db when vdb(out) = -3.0103 cross=1
let gain_ratio = f3db / 1000
print gain_ratio
write
quit
.endc

.end
