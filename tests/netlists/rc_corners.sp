* rc_corners - parameter/library/temperature sweep with an analytic answer
*
* Single-pole RC lowpass.  R has a linear tempco, so the -3 dB frequency is a
* function of the swept parameter, the library corner and the temperature:
*
*   f3db = 1 / (2*pi * rr*kfac*(1 + tc1*(T - 27)) * cc)
*
* Exercises multi-assignment .param, a '+' continuation .param, a library
* sweep with an explicit key, a temperature sweep, and extraction from a
* plain .meas dot card.
*
* Note on the measurement: ngspice 42's dot-card .meas parser rejects vdb()
* and vm(), and compares v() on the REAL part.  For a single pole the real
* part of the transfer function is exactly 0.5 at the pole frequency, so
* 'when v(out) = 0.5' is an exact and parser-safe way to find f3db.

** ngr_param cc 1n 10n
** ngr_param rr 1k 4k
** ngr_lib models.lib(tt) tt ff ss
** ngr_temp -40 27 125
** ngr_out f3db

.lib @TESTDIR@/models.lib tt

.param vsup=5.0 cc=1n
+ rr=1k

V1 in 0 DC 0 AC 1
R1 in out {rr*kfac} tc1=1e-3
C1 out 0 {cc}

.ac dec 500 1 1e9
.meas ac f3db when v(out) = 0.5 cross=1

.end
