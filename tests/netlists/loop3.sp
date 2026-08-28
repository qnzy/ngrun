* loop3 - three-pole feedback loop with analytically known loop gain
*
* Ideal buffers isolate each RC section, so the loop gain is exactly
*
*   T(s) = a0 / ((1+s/w1)(1+s/w2)(1+s/w3)),   wi = 2*pi*fi
*
* which gives a closed-form a0_db, UGF, phase margin and gain margin.
* Two probes break the same loop at different points (dot form at the amp
* output, colon form at the amp input); a single-loop system has the same
* loop gain at every break point, so both probes must agree.

** ngr_stb amp.out   fstart=0.01 fstop=1e9 pts=200
** ngr_stb amp:2     fstart=0.01 fstop=1e9 pts=200

.param a0=1000
.param f1=100 f2=1e5 f3=1e6
.param twopi=6.283185307179586
.param cp1={1/(twopi*f1)} cp2={1/(twopi*f2)} cp3={1/(twopi*f3)}

Xamp n1 fb amp

R1 n1 n2 1
C1 n2 0 {cp1}
Eb1 n3 0 n2 0 1

R2 n3 n4 1
C2 n4 0 {cp2}
Eb2 n5 0 n4 0 1

R3 n5 fb 1
C3 fb 0 {cp3}

.subckt amp out inn
Eamp out 0 0 inn {a0}
.ends amp

.ac dec 200 0.01 1e9

.end
