* loop3_hier - same loop as loop3.sp, amplifier buried one level deeper
*
* The probe path 'core.amp.out' forces ngrun to clone the intermediate
* subcircuit and to emit a hierarchical .control block.  The extracted
* margins must match the flat loop3.sp exactly.

** ngr_stb core.amp.out fstart=0.01 fstop=1e9 pts=200

.param a0=1000
.param f1=100 f2=1e5 f3=1e6
.param twopi=6.283185307179586
.param cp1={1/(twopi*f1)} cp2={1/(twopi*f2)} cp3={1/(twopi*f3)}

Xcore n1 fb core

R1 n1 n2 1
C1 n2 0 {cp1}
Eb1 n3 0 n2 0 1

R2 n3 n4 1
C2 n4 0 {cp2}
Eb2 n5 0 n4 0 1

R3 n5 fb 1
C3 fb 0 {cp3}

.subckt core out inn
Xamp out inn amp
.ends core

.subckt amp out inn
Eamp out 0 0 inn {a0}
.ends amp

.ac dec 200 0.01 1e9

.end
