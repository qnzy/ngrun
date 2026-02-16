# ngrun

NGSpice Corner + Stability Simulation Tool

Automates corner sweeps and/or Tian closed-loop stability analysis on ngspice
netlists. Configuration is embedded as comment directives directly in the
netlist — no separate config file needed.

---

## Installation

No installation required. Copy `ngrun.py` anywhere and run it with Python 3.6+.
ngspice must be on your `PATH`.

```bash
python3 ngrun.py [options] <netlist>
```

Optionally make it executable:

```bash
chmod +x ngrun.py
./ngrun.py my_circuit.sp
```

---

## Quick start

Add `ngr_` directives as comments in your netlist, then run:

```spice
* My LDO regulator
** ngr_param vdd_p 3.0 3.3 3.6
** ngr_lib models.lib(tt) tt ff ss
** ngr_temp -40 27 125
** ngr_out vout_dc iq_ua
** ngr_stb rfb.1

.lib /path/to/models.lib tt
.param vdd_p=3.3
...
```

```bash
python3 ngrun.py ldo.sp
```

This runs 27 corners (3 voltages × 3 process corners × 3 temperatures),
extracts `vout_dc` and `iq_ua` from a normal simulation and PM/GM from a
Tian stability simulation at each corner, and writes `ldo_results.csv`.

---

## Netlist directives

All directives are embedded as SPICE comments (lines starting with `*`).
They are ignored by ngspice and processed only by ngrun.

### `ngr_param` — parameter sweep

```spice
** ngr_param <name> <value1> [value2 ...]
```

Sweeps a `.param` statement across the given values.
The name must exactly match the `.param` name in the netlist.

```spice
** ngr_param vdd_p 2.7 3.0 3.3
** ngr_param cload 10p 100p
```

Multiple `ngr_param` directives produce a full cartesian product.

---

### `ngr_lib` — library corner sweep

```spice
** ngr_lib <libfile>[(<key>)] <corner1> [corner2 ...]
```

Replaces the keyword in `.lib` statements that reference the named file.
An optional `(key)` restricts substitution to `.lib` lines with a matching
keyword — useful when a single library file has multiple `.lib` calls with
different keys.

```spice
** ngr_lib models.lib tt ff ss
** ngr_lib models.lib(tt) tt ff ss
** ngr_lib models.lib(res_nom) res_nom res_fast res_slow
```

The tool matches `.lib /any/path/models.lib <key>` and replaces `<key>`.

---

### `ngr_temp` — temperature sweep

```spice
** ngr_temp <T1> [T2 ...]
```

Temperatures in degrees Celsius. Injects a `.temp` statement into each
corner netlist before the first analysis command.

```spice
** ngr_temp -40 27 125
```

If `ngr_temp` is absent, a single simulation is run at the simulator default
temperature (usually 27°C, unless `.temp` is already in the netlist).

---

### `ngr_out` — output measures

```spice
** ngr_out <measure1> [measure2 ...]
```

Names of `.measure` results to extract from the normal simulation.
Each name must match a `.measure` statement in the netlist.

```spice
** ngr_out trise tfall power_avg vout_dc
```

Output values appear as columns in the CSV. If a measure is not found in
ngspice output, the field is set to `N/A`.

---

### `ngr_stb` — Tian stability analysis

```spice
** ngr_stb <probe> [fstart=<f>] [fstop=<f>] [pts=<n>]
```

Instruments the netlist with a Tian probe at the specified pin and runs a
separate two-sweep AC simulation to extract loop gain, phase margin, and gain
margin.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fstart`  | `1`     | Start frequency, Hz |
| `fstop`   | `1e9`   | Stop frequency, Hz |
| `pts`     | `100`   | Points per decade |

```spice
** ngr_stb rfb.1
** ngr_stb ota.out fstart=1 fstop=1e9 pts=100
```

Only one `ngr_stb` directive is allowed per netlist. If multiple are present,
the first is used and a warning is printed.

#### Probe specification

Two forms are supported, and they can be mixed in a hierarchical path:

**`inst.pinname`** — resolve by pin name. Requires the subcircuit definition
to be present in the netlist or an included file. Works for all devices whose
model is defined in the netlist.

**`inst:N`** — resolve by 1-based pin position as written on the instance
line. Does not require the subcircuit definition. Use this for library-only
models (PDK devices, etc.) whose subcircuit is not in the netlist.

Both forms support hierarchy — intermediate instances always use dot
separation, only the leaf uses the colon:

```
instance.pin            dot form, top-level leaf
instance:N              colon form, top-level leaf
inst1.inst2.pin         dot form, inst2 inside inst1
inst1.inst2:N           colon form, inst2 inside inst1
```

Instance names do not require the `X` prefix — it is added automatically.

**Pin names for the dot form:**

| Device | Prefix | Pin names |
|--------|--------|-----------|
| Subcircuit | X | use the subcircuit's port names |
| Resistor, Capacitor, Inductor | R, C, L | `1`, `2` |
| MOSFET | M | `d`, `g`, `s`, `b`  (or `1`–`4`) |
| BJT | Q | `c`, `b`, `e`, `s`  (or `1`–`4`) |
| Diode | D | `a`, `k`  (or `1`, `2`) |
| Voltage/Current source | V, I | `p`, `n`  (or `1`, `2`) |
| Controlled sources | E, G | `1`–`4` |

**Examples:**

```spice
** ngr_stb ota.out             * Xota subcircuit port 'out' (dot form)
** ngr_stb xr1:1               * Xr1 pin 1 by position (library-only model)
** ngr_stb ldo.erramp.out      * Xerramp inside Xldo, port 'out'
** ngr_stb amp.m1.d            * MOSFET M1 drain inside Xamp
** ngr_stb ldo.xrfb:1         * library resistor inside Xldo, pin 1
```

#### How the Tian probe works

The probe is inserted by breaking the signal path at the chosen pin and
inserting three elements:

```
  a ──[Vi_stb]── x ──[Vnodebuffer_stb]── b
                 │
              [Ii_stb]
                 │
                GND
```

Two AC sweeps are run: one with `Vi` active and `Ii` off, one with `Ii`
active and `Vi` off. The loop gain is computed as:

```
T = D / (1 − D),   D = 2(I₁V₂ − V₁I₂) + V₁ + I₂
```

This gives an accurate result regardless of the loop's source/load impedances.

Reference: M. Tian et al., *"Striving for Small-Signal Stability,"*
IEEE Circuits & Devices, vol. 17, no. 1, pp. 31–41, Jan. 2001.

**Phase unwrapping:** ngspice's `vp()` returns phase in (−180°, +180°]. The gain-margin calculation requires detecting the −180° crossing, which can fail when the phase wraps. ngrun applies `unwrap()` to the phase vector before all measurements to avoid this. Requires ngspice 37 or later.

#### Stability output columns

| Column | Description |
|--------|-------------|
| `a0_db` | DC loop gain in dB (gain at `fstart`) |
| `ugf_freq` | Unity-gain frequency in Hz |
| `pm` | Phase margin in degrees |
| `gm_freq` | Gain-margin frequency in Hz (−180° crossing) |
| `gm_db` | Gain margin in dB |

Interpretation guide:

| PM | Assessment |
|----|-----------|
| < 0° | Unstable |
| 0°–30° | Likely to ring or oscillate |
| 30°–45° | Marginally stable, caution |
| > 45° | Acceptable for most applications |

| GM | Assessment |
|----|-----------|
| < 0 dB | Unstable |
| 0–6 dB | Marginal |
| > 6 dB | Acceptable |

If the loop never crosses 0 dB (UGF not found) or −180° (GM not found),
the corresponding fields are `N/A`.

**Raw waveform files** (`av`, `av_dB`, `av_ph`, unwrapped phase) are written to the same temporary directory as the generated corner netlists. Use `-k` to keep the temp directory and inspect the waveforms after the run.

---

## Simulation modes

### Corner sweep (default)

Runs the full cartesian product of all `ngr_param`, `ngr_lib`, and `ngr_temp`
values. If none are defined, a single simulation is run with the netlist
exactly as written.

### Typical (`--typ`)

Runs a single simulation with the netlist as written — no `.param`, `.lib`,
or `.temp` substitution is performed. Useful for a quick sanity check or
when you only want to run the stability analysis on the nominal design.

### Simulation runs per corner

| `ngr_out` | `ngr_stb` | Runs per corner |
|----------|----------|-----------------|
| yes | no | 1 (normal) |
| no | yes | 1 (Tian) |
| yes | yes | 2 (normal + Tian, results merged) |
| no | no | 1 (normal, nothing extracted — warning) |

When both are present, the normal simulation uses the original corner netlist.
The Tian simulation uses a separately instrumented copy. The two sets of
results are merged into one CSV row per corner.

---

## Command line reference

```
python3 ngrun.py [options] <netlist>
```

| Option | Description |
|--------|-------------|
| `--typ` | Single typical run, netlist as-is |
| `-k`, `--keep` | Keep generated temporary netlists |
| `-j N`, `--parallel N` | Run N corners in parallel (default: 1) |
| `-o FILE`, `--output FILE` | Output CSV filename (default: `<netlist>_results.csv`) |
| `-n`, `--no-run` | Generate netlists only, do not simulate (combine with `-k`) |

---

## Output CSV

Results are written to `<netlist>_results.csv` by default.

**Column order:**

```
corner_id | temperature | param_<name>... | lib_<name>... | <ngr_out measures>... | a0_db | ugf_freq | pm | gm_freq | gm_db
```

- `corner_id`: `c0001`, `c0002`, ... for corner sweeps; `typ` for `--typ` mode
- `temperature`: in °C; `typ` if no temperature substitution was performed
- `param_*`: one column per swept parameter
- `lib_*`: one column per swept library (name includes the key if specified)
- Stability columns are only present if `ngr_stb` is defined

**Example rows:**

```csv
corner_id,temperature,param_vdd_p,lib_models.lib_tt,trise,tfall,a0_db,ugf_freq,pm,gm_freq,gm_db
c0001,-40,2.7,tt,1.23e-9,2.45e-9,78.3,1.45e6,62.1,3.2e7,18.4
c0002,-40,2.7,ff,1.18e-9,2.31e-9,79.1,1.52e6,59.8,3.4e7,19.1
...
```

---

## Examples

### Corner sweep, measures only

```spice
* Inverter chain timing
** ngr_param vdd 1.62 1.8 1.98
** ngr_lib process.lib tt ff ss
** ngr_temp -40 27 125
** ngr_out tphl tplh

.lib /pdk/process.lib tt
.param vdd=1.8
...
```

```bash
python3 ngrun.py inv_chain.sp
# Produces: inv_chain_results.csv, 27 corners
```

---

### Corner sweep with stability

```spice
* LDO regulator
** ngr_param vdd_p 3.0 3.3 3.6
** ngr_lib models.lib(tt) tt ff ss
** ngr_temp -40 27 125
** ngr_out vout_dc psrr_db iq_ua
** ngr_stb rfb.1 fstart=1 fstop=100e6 pts=50

.lib /pdk/models.lib tt
.param vdd_p=3.3
...
```

```bash
python3 ngrun.py ldo.sp -j 4
# 27 corners × 2 sims each, 4 parallel workers
# Produces: ldo_results.csv with vout_dc, psrr_db, iq_ua, a0_db, ugf_freq, pm, gm_freq, gm_db
```

---

### Typical run only

```bash
python3 ngrun.py ldo.sp --typ
# Single simulation, netlist as-is
# If ngr_stb present: prints Tian summary to terminal + writes single-row CSV
```

---

### Inspect generated netlists without running

```bash
python3 ngrun.py ldo.sp -k -n
# Generates all corner netlists into a temp directory and exits
# Useful for debugging substitutions before committing to a full run
```

---

## Notes and limitations

**Hierarchical probing** creates a clone of each subcircuit along the probe
path (named with a `_stb` suffix). This ensures that only the targeted
instance is modified; other instantiations of the same subcircuit are
unaffected. The original netlist file is never modified.

**AC source zeroing:** before inserting the Tian probe, all `AC` magnitudes
on existing V/I sources are set to zero. The probe's `Vi_stb` and `Ii_stb`
sources are then the only active AC stimulus. This is required for correct
loop gain extraction.

**Parallel execution** (`-j N`) parallelizes at the corner level. Both
simulations for a given corner (normal + Tian) run sequentially within one
worker process. Different corners run in parallel.

**Stability + measures incompatibility:** ngr_stb and ngr_out simulations
cannot share a netlist. The Tian-instrumented netlist has modified AC sources
and extra probe elements that would corrupt `.measure` results. ngrun handles
this automatically by running them as two separate simulations.

**`.lib` path matching** requires the library filename to appear at the end
of the path component in the `.lib` statement:
`.lib /path/to/models.lib key` — the tool matches on `models.lib` only,
not the full path. The full path is preserved in substitution.

**ngspice version requirement for stability analysis:** the `unwrap()` command used for phase unwrapping was introduced in ngspice 37. Earlier versions will fail when `ngr_stb` is active. Check your version with `ngspice --version`.

**Timeout:** each ngspice subprocess has a 10-minute timeout. Adjust
`timeout=600` in `run_ngspice()` if your simulations are longer.

---

## References

- M. Tian, B. Visvanathan, J. Hantgan, K. Kundert, *"Striving for
  Small-Signal Stability,"* IEEE Circuits & Devices Magazine, vol. 17, no. 1,
  pp. 31–41, January 2001.
- R. Turnbull, ngspice Tian probe feature request:
  sourceforge.net/p/ngspice/feature-requests/34/
- SLICE Semiconductor open source tool examples:
  github.com/SLICESemiconductor/OpenSourceTool_Examples
