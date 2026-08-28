# ngrun test suite

    make test      everything (sim layer auto-skipped if ngspice is missing)
    make unit      helper-level tests, no simulator
    make gen       netlist-generation tests, no simulator
    make sim       full ngspice runs against closed-form answers
    make verbose   everything, with per-test detail
    make clean

Drop this directory next to `ngrun.py`:

    ngrun.py
    tests/
      Makefile
      run_tests.py
      netlists/

## Layers

**unit** calls ngrun's helpers directly: parameter substitution, `.param`
statement state tracking, measurement extraction, error-row construction,
probe-spec parsing.

**gen** runs ngrun with `-n -k` and inspects the generated corner netlists —
this is where "did the sweep actually sweep" is decided, independently of
whether the simulator agrees.

**sim** runs ngspice for real and checks results against closed-form
expressions. Nothing is compared to a golden file; every expected number is
computed from theory in `run_tests.py`, so the tests do not need updating when
ngspice's formatting changes.

## Netlists

| file | what it exercises |
| --- | --- |
| `rc_corners.sp` | multi-assignment `.param`, `+` continuation `.param`, `ngr_lib` with a key, `ngr_temp`, `ngr_out` from a `.meas` dot card. 36 corners with an analytic `f3db`. |
| `temp_fixed.sp` | netlist owns its `.temp`, no `ngr_temp` — the measured `f3db` proves which temperature was used |
| `loop3.sp` | Tian probe on a three-pole loop with closed-form `a0`/UGF/PM/GM; two probes break the same loop at different points (dot form and colon form) |
| `loop3_hier.sp` | same loop with the amplifier one level deeper, forcing subckt cloning and a hierarchical `.control` block |
| `ctrl.sp` | `.control` block, `let`/`print` derived measurement, injected `set rawfile`, and a decoy `f3db = 0` printed before the real measurement |
| `broken.sp` | unsimulatable netlist — every corner must still appear in the CSV |
| `models.lib` | three library sections (`tt`/`ff`/`ss`) scaling the RC time constant |

`@TESTDIR@` in a netlist is expanded to an absolute path before the run. This
is necessary because ngrun writes corner netlists into a temp directory, so a
relative `.lib` or `.include` path in the source netlist no longer resolves.

## Reference values

`loop3.sp` is three ideal-buffered RC sections behind a VCVS, so

    T(s) = a0 / ((1+s/w1)(1+s/w2)(1+s/w3))

with a0 = 1000, f = 100 Hz / 100 kHz / 1 MHz. `run_tests.py` solves this for
UGF, PM, GM freq and GM numerically and compares. Because it is a single loop,
the loop gain is independent of where the loop is broken — the two probes in
`loop3.sp` must agree, and the hierarchical probe in `loop3_hier.sp` must agree
with both.

`rc_corners.sp` expects `f3db = 1/(2*pi*R*C)` with
`R = rr * kfac * (1 + tc1*(T - 27))`.

## Two ngspice notes worth knowing

ngspice 42's **dot-card** `.meas` parser rejects `vdb()` and `vm()` (it warns
"can't parse" and then aborts the analysis), and it compares `v()` on the
**real part**, not the magnitude. The RC netlists exploit the fact that the
real part of a single-pole response is exactly 0.5 at the pole, so
`when v(out) = 0.5` is both exact and parser-safe. Inside a `.control` block
the full measure implementation is available and `vdb()` works — that is what
`ctrl.sp` uses.

## Mutation status

Each of the five behaviours below was reverted in `ngrun.py` and the suite
re-run, to confirm the tests actually catch them rather than merely passing:

| reverted behaviour | tests that fail |
| --- | --- |
| `.temp` left alone when no `ngr_temp` | 2 |
| multi-assignment `.param` substitution | 9 |
| last-match `ngr_out` extraction | 2 |
| comment terminates a `.param` statement | 3 |
| error row for a dead worker | 2 |
| directive token validation | 2 |
| `ngr_lib` checked against `.lib` statements | 2 |
