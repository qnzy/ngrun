#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                           NGRUN - User Guide                              ║
║              NGSpice Corner + Stability Simulation Tool v1.0              ║
╚═══════════════════════════════════════════════════════════════════════════╝

OVERVIEW
--------
ngrun automates corner simulations and/or Tian closed-loop stability analysis
on ngspice netlists. Configuration is embedded as comment directives (ng_*)
directly in the netlist.

NETLIST CONFIGURATION DIRECTIVES
---------------------------------
All directives are embedded as comments (lines starting with '*' or '**').
They must start with 'ngr_' to be recognised.

1. ngr_param - Parameter variations
   Syntax: ** ngr_param <name> <val1> <val2> ... <valN>
   Example:
     ** ngr_param vdd 2.7 3.0 3.3
     ** ngr_param cload 10p 100p

2. ngr_lib - Library corner variations
   Syntax: ** ngr_lib <libfile>[(<key>)] <corner1> ... <cornerN>
   Examples:
     ** ngr_lib process.lib tt ff ss
     ** ngr_lib models.lib(mos_typ) tt ff ss
   Matches .lib statements by filename (and optional key), substitutes corner.

3. ngr_temp - Temperature sweep
   Syntax: ** ngr_temp <T1> <T2> ... <TN>
   Example:
     ** ngr_temp -40 27 125
   Temperatures in Celsius.

4. ngr_out - Measurements to extract (from .measure statements)
   Syntax: ** ngr_out <meas1> <meas2> ... <measN>
   Example:
     ** ngr_out trise tfall power_avg
   Results written to CSV. Requires a normal simulation (not Tian-only).

5. ngr_stb - Tian stability probe (one probe per netlist for now)
   Syntax: ** ngr_stb <probe> [fstart=<f>] [fstop=<f>] [pts=<n>]
   Examples:
     ** ngr_stb ota.out
     ** ngr_stb reg.erramp.out fstart=0.1 fstop=1e9 pts=50
     ** ngr_stb rfb.1
   Probe specification (hierarchical dot notation):
     inst.pin                - pin on a top-level instance
     inst1.inst2.pin         - pin on inst2 inside inst1's subcircuit
   Pin names for primitives: 1/2 (R/C/L/V/I), d/g/s/b (M), a/k (D), c/b/e (Q)
   Stability columns in CSV: a0_db, ugf_freq, pm, gm_freq, gm_db

SIMULATION MODES
----------------
By default, ngrun runs all corner combinations (cartesian product of all
ngr_param, ngr_lib, ngr_temp values). Use --typ to skip corner generation and
run a single simulation with the netlist exactly as written.

  ngr_out only       -> 1 simulation per corner  (normal)
  ngr_stb only       -> 1 simulation per corner  (Tian-instrumented)
  ngr_out + ngr_stb   -> 2 simulations per corner (normal, then Tian)
  neither           -> 1 simulation per corner  (warn: nothing extracted)

COMMAND LINE
------------
  ngrun <netlist>              Run all corners
  ngrun <netlist> --typ        Run single typical simulation (no substitution)
  ngrun <netlist> -k           Keep generated netlists in temp dir
  ngrun <netlist> -j 4         Run 4 corners in parallel
  ngrun <netlist> -o out.csv   Custom output CSV filename
  ngrun <netlist> -n           Generate netlists only, do not run

CSV COLUMN ORDER
----------------
  corner_id, temperature, param_*, lib_*, <ngr_out measures>,
  a0_db, ugf_freq, pm, gm_freq, gm_db  (only when ngr_stb is present)

REFERENCES
----------
  M. Tian et al., "Striving for Small-Signal Stability,"
  IEEE Circuits & Devices, vol. 17, no. 1, pp. 31-41, Jan. 2001.
  R. Turnbull, ngspice Tian probe, sourceforge.net/p/ngspice/feature-requests/34/
"""

import argparse
import copy
import csv
import itertools
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# SPICE netlist data structures
# ---------------------------------------------------------------------------

PRIMITIVE_PIN_MAPS = {
    "r": {"1": 0, "2": 1, "p": 0, "n": 1},
    "c": {"1": 0, "2": 1, "p": 0, "n": 1},
    "l": {"1": 0, "2": 1, "p": 0, "n": 1},
    "d": {"1": 0, "2": 1, "a": 0, "k": 1},
    "m": {"1": 0, "2": 1, "3": 2, "4": 3,
          "d": 0, "g": 1, "s": 2, "b": 3},
    "q": {"1": 0, "2": 1, "3": 2, "4": 3,
          "c": 0, "b": 1, "e": 2, "s": 3},
    "j": {"1": 0, "2": 1, "3": 2,
          "d": 0, "g": 1, "s": 2},
    "v": {"1": 0, "2": 1, "p": 0, "n": 1},
    "i": {"1": 0, "2": 1, "p": 0, "n": 1},
    "e": {"1": 0, "2": 1, "3": 2, "4": 3},
    "f": {"1": 0, "2": 1},
    "g": {"1": 0, "2": 1, "3": 2, "4": 3},
    "h": {"1": 0, "2": 1},
}


def is_primitive(name):
    return bool(name) and name[0].lower() in PRIMITIVE_PIN_MAPS


def is_subckt_instance(name):
    return bool(name) and name[0].lower() == "x"


class SpiceLine:
    """One logical SPICE line (continuation lines merged)."""
    def __init__(self, text, line_numbers=None):
        self.text = text
        self.line_numbers = line_numbers or []

    def __repr__(self):
        return f"SpiceLine({self.text!r})"


class SubcktDef:
    """Parsed .subckt definition."""
    def __init__(self, name, pins, body_lines, params=None):
        self.name = name
        self.pins = pins
        self.body_lines = body_lines
        self.params = params or []

    def __repr__(self):
        return f"SubcktDef({self.name}, pins={self.pins})"

    def find_instance(self, inst_name):
        target = inst_name.lower()
        for i, sl in enumerate(self.body_lines):
            parts = sl.text.split()
            if parts and parts[0].lower() == target:
                return i, parts
        return None, None

    def find_all_instances(self):
        return [sl.text.split()[0] for sl in self.body_lines
                if sl.text.strip() and not sl.text.strip().startswith(".")]

    def to_lines(self):
        header = f".subckt {self.name} {' '.join(self.pins)}"
        if self.params:
            header += " " + " ".join(self.params)
        out = [header + "\n"]
        for sl in self.body_lines:
            out.append(sl.text + "\n")
        out.append(f".ends {self.name}\n")
        return out


class SpiceNetlist:
    """Parsed SPICE netlist."""

    def __init__(self):
        self.title = ""
        self.top_lines = []
        self.subckt_defs = {}
        self.include_paths = []

    @classmethod
    def parse(cls, lines, base_dir="."):
        nl = cls()
        merged = cls._merge_continuations(lines)
        nl._parse_structure(merged)
        for inc_path in nl.include_paths:
            full = inc_path if os.path.isabs(inc_path) else os.path.join(base_dir, inc_path)
            if os.path.isfile(full):
                try:
                    with open(full) as f:
                        nl._parse_included_subckts(cls._merge_continuations(f.readlines()))
                except IOError:
                    pass
        return nl

    @staticmethod
    def _merge_continuations(lines):
        merged = []
        for i, line in enumerate(lines):
            stripped = line.rstrip("\n").rstrip("\r")
            clean = stripped.lstrip()
            if clean.startswith("+") and merged:
                merged[-1].text += " " + clean[1:].strip()
                merged[-1].line_numbers.append(i + 1)
            else:
                merged.append(SpiceLine(stripped, [i + 1]))
        return merged

    def _parse_structure(self, merged):
        i = 0
        if merged:
            self.title = merged[0].text
            i = 1
        while i < len(merged):
            sl = merged[i]
            lower = sl.text.strip().lower()
            if lower.startswith(".include") or lower.startswith(".lib"):
                parts = sl.text.strip().split(None, 1)
                if len(parts) >= 2:
                    self.include_paths.append(parts[1].strip().strip('"\''))
                self.top_lines.append(sl)
                i += 1
                continue
            if lower.startswith(".subckt"):
                subckt, end_i = self._parse_subckt_block(merged, i)
                self.subckt_defs[subckt.name.lower()] = subckt
                i = end_i + 1
                continue
            if lower == ".end":
                i += 1
                continue
            if lower.startswith(".control"):
                while i < len(merged):
                    if merged[i].text.strip().lower().startswith(".endc"):
                        i += 1
                        break
                    i += 1
                continue
            self.top_lines.append(sl)
            i += 1

    def _parse_subckt_block(self, merged, start_i):
        parts = merged[start_i].text.strip().split()
        name = parts[1]
        pins, params = [], []
        for p in parts[2:]:
            (params if "=" in p else pins).append(p)
        body = []
        i = start_i + 1
        while i < len(merged):
            lower = merged[i].text.strip().lower()
            if lower.startswith(".ends"):
                break
            if lower.startswith(".subckt"):
                nested, end_i = self._parse_subckt_block(merged, i)
                self.subckt_defs[nested.name.lower()] = nested
                i = end_i + 1
                continue
            body.append(merged[i])
            i += 1
        return SubcktDef(name, pins, body, params), i

    def _parse_included_subckts(self, merged):
        i = 0
        while i < len(merged):
            if merged[i].text.strip().lower().startswith(".subckt"):
                subckt, end_i = self._parse_subckt_block(merged, i)
                self.subckt_defs.setdefault(subckt.name.lower(), subckt)
                i = end_i + 1
            else:
                i += 1

    def get_subckt(self, name):
        return self.subckt_defs.get(name.lower())

    def find_top_instance(self, inst_name):
        target = inst_name.lower()
        for i, sl in enumerate(self.top_lines):
            parts = sl.text.split()
            if parts and parts[0].lower() == target:
                return i, parts
        return None, None

    def add_subckt(self, subckt_def):
        self.subckt_defs[subckt_def.name.lower()] = subckt_def

    def zero_ac_sources(self):
        ac_pat = re.compile(r"(\bac\s+)(\S+)", re.IGNORECASE)
        for sl in self.top_lines:
            if sl.text and sl.text[0].lower() in ("v", "i"):
                if re.search(r"\bac\b", sl.text, re.IGNORECASE):
                    sl.text = ac_pat.sub(r"\g<1>0", sl.text)

    def to_text(self):
        out = [self.title + "\n"]
        for sl in self.top_lines:
            out.append(sl.text + "\n")
        out.append("\n")
        for sdef in self.subckt_defs.values():
            out.extend(sdef.to_lines())
            out.append("\n")
        out.append(".end\n")
        return "".join(out)


# ---------------------------------------------------------------------------
# Tian probe: hierarchical resolution and subcircuit cloning
# ---------------------------------------------------------------------------

def _resolve_instance_subckt(netlist, inst_parts):
    nets_and_subckt = []
    for p in inst_parts[1:]:
        if "=" in p:
            break
        nets_and_subckt.append(p)
    if len(nets_and_subckt) < 2:
        raise ValueError(f"Instance line too short: {' '.join(inst_parts)}")
    return nets_and_subckt[-1], nets_and_subckt[:-1]


def _resolve_pin_index(device_name, pin_spec, subckt_pins=None):
    prefix = device_name[0].lower()
    if prefix == "x":
        if subckt_pins is None:
            raise ValueError(f"No subcircuit pin list for '{device_name}'")
        for i, p in enumerate(subckt_pins):
            if p.lower() == pin_spec.lower():
                return i
        if pin_spec.isdigit():
            idx = int(pin_spec) - 1
            if 0 <= idx < len(subckt_pins):
                return idx
        raise ValueError(f"Pin '{pin_spec}' not found on '{device_name}'. "
                         f"Available: {subckt_pins}")
    pin_map = PRIMITIVE_PIN_MAPS.get(prefix)
    if pin_map is None:
        raise ValueError(f"Unknown device prefix '{prefix}' for '{device_name}'")
    if pin_spec.lower() in pin_map:
        return pin_map[pin_spec.lower()]
    raise ValueError(f"Pin '{pin_spec}' not valid for '{device_name}'. "
                     f"Valid: {list(pin_map.keys())}")


def _clone_subckt(original, new_name):
    body = [SpiceLine(sl.text, list(sl.line_numbers)) for sl in original.body_lines]
    return SubcktDef(new_name, list(original.pins), body, list(original.params))


def resolve_hierarchical_probe(netlist, path_parts):
    """
    Walk the hierarchy, clone subcircuits as needed, and return
    (original_net, new_net, stb_x_node, leaf_scope).
    Modifies netlist in place.
    """
    if len(path_parts) < 2:
        raise ValueError(f"Probe must be at least instance.pin, got: {'.'.join(path_parts)}")

    instances = path_parts[:-1]
    pin_name = path_parts[-1]
    current_scope = None
    clone_suffix = "_stb"

    for depth, inst_name_raw in enumerate(instances):
        is_leaf = (depth == len(instances) - 1)

        # Normalise instance name
        inst_name = inst_name_raw
        if not inst_name[0:1].lower() in ("x",) and inst_name[0].lower() not in PRIMITIVE_PIN_MAPS:
            inst_name = "x" + inst_name

        # Find instance in current scope
        find = (netlist.find_top_instance if current_scope is None
                else current_scope.find_instance)
        idx, inst_parts = find(inst_name)
        if idx is None:
            idx, inst_parts = find("x" + inst_name)
            if idx is None:
                scope_name = "top level" if current_scope is None else current_scope.name
                raise ValueError(f"Instance '{inst_name}' not found in {scope_name}.")
            inst_name = "x" + inst_name
            if current_scope is None:
                inst_parts = netlist.top_lines[idx].text.split()
            else:
                inst_parts = current_scope.body_lines[idx].text.split()

        if is_leaf:
            if is_subckt_instance(inst_name):
                subckt_name, _ = _resolve_instance_subckt(netlist, inst_parts)
                subckt_def = netlist.get_subckt(subckt_name)
                if subckt_def is None:
                    raise ValueError(f"Subcircuit '{subckt_name}' not found.")
                pin_idx = _resolve_pin_index(inst_name, pin_name, subckt_def.pins)
            else:
                pin_idx = _resolve_pin_index(inst_name, pin_name)

            net_idx = 1 + pin_idx
            if net_idx >= len(inst_parts):
                raise ValueError(f"Pin index out of range for: {' '.join(inst_parts)}")
            original_net = inst_parts[net_idx]
            new_net = original_net + clone_suffix
            stb_x = original_net + "_stb_x"

            inst_parts[net_idx] = new_net
            new_line = " ".join(inst_parts)
            if current_scope is None:
                netlist.top_lines[idx].text = new_line
            else:
                current_scope.body_lines[idx].text = new_line

            return original_net, new_net, stb_x, current_scope

        else:
            if not is_subckt_instance(inst_name):
                raise ValueError(
                    f"Cannot descend into primitive '{inst_name}'. "
                    f"Only X instances support hierarchy.")
            subckt_name, _ = _resolve_instance_subckt(netlist, inst_parts)
            subckt_def = netlist.get_subckt(subckt_name)
            if subckt_def is None:
                raise ValueError(f"Subcircuit '{subckt_name}' not found.")

            clone_name = subckt_name + clone_suffix + (str(depth) if depth > 0 else "")
            cloned = _clone_subckt(subckt_def, clone_name)
            netlist.add_subckt(cloned)

            # Point the instance to the clone
            for i in range(len(inst_parts) - 1, 0, -1):
                if "=" not in inst_parts[i] and inst_parts[i].lower() == subckt_name.lower():
                    inst_parts[i] = clone_name
                    break
            new_line = " ".join(inst_parts)
            if current_scope is None:
                netlist.top_lines[idx].text = new_line
            else:
                current_scope.body_lines[idx].text = new_line

            current_scope = cloned


# ---------------------------------------------------------------------------
# Tian probe elements and control block
# ---------------------------------------------------------------------------

def _build_probe_elements(net_a, net_b, stb_x):
    return [SpiceLine(l) for l in [
        "* --- Tian Probe (ngrun) ---",
        f"Ii_stb 0 {stb_x} DC 0 AC 0",
        f"Vi_stb {stb_x} {net_a} DC 0 AC 1",
        f"Vnodebuffer_stb {net_b} {stb_x} 0",
        "* --- End Tian Probe ---",
    ]]


def _build_tian_control(stb_x, raw_file, fstart, fstop, pts,
                         inside_subckt, subckt_inst_path):
    if inside_subckt and subckt_inst_path:
        hier = ".".join(subckt_inst_path) + "."
        v_node  = f"{hier}{stb_x}"
        i_src   = f"v.{hier}vi_stb"
        alt_ii  = f"i.{hier}ii_stb"
        alt_vi  = f"v.{hier}vi_stb"
    else:
        v_node  = stb_x
        i_src   = "Vi_stb"
        alt_ii  = "Ii_stb"
        alt_vi  = "Vi_stb"

    alter = (f"alter {alt_ii} acmag=1\nalter {alt_vi} acmag=0"
             if inside_subckt else
             f"alter @{alt_ii}[acmag] = 1\nalter @{alt_vi}[acmag] = 0")

    return "\n".join([
        ".control",
        "",
        "* === Tian Stability Analysis (ngrun) ===",
        "* Run 1: Voltage injection (Vi AC=1, Ii AC=0)",
        f"ac dec {pts} {fstart} {fstop}",
        "",
        "* Run 2: Current injection",
        alter,
        f"ac dec {pts} {fstart} {fstop}",
        "",
        f"let V1 = ac1.v({v_node})",
        f"let I1 = ac1.i({i_src})",
        f"let V2 = v({v_node})",
        f"let I2 = i({i_src})",
        "",
        "let D = 2*(I1*V2 - V1*I2) + V1 + I2",
        "let av = D / (1 - D)",
        "let av_dB = vdb(av)",
        "let av_ph_rad = vp(av)",
        "let av_ph = 180/pi * vp(av)",
        "",
        f"meas ac a0_db    find av_dB    at = {fstart:.6e}",
        "meas ac ugf_freq  when av_dB    = 0",
        "meas ac ugf_ph_rad find av_ph_rad when av_dB = 0 cross=1",
        "let ugf_ph_deg = ugf_ph_rad * 180 / pi",
        "let pm = 180 + ugf_ph_deg",
        "print pm",
        "",
        "let n180_rad = -pi",
        "meas ac gm_freq  when av_ph_rad = n180_rad cross=1",
        "meas ac gm_gain  find av_dB     when av_ph_rad = n180_rad cross=1",
        "let gm_db = -gm_gain",
        "print gm_db",
        "",
        f"write {raw_file} av av_dB av_ph frequency",
        "",
        "quit",
        ".endc",
    ]) + "\n"


def instrument_netlist_tian(netlist_text, probe_spec, fstart, fstop, pts,
                             raw_file, base_dir="."):
    """
    Instrument a netlist string with a Tian probe.
    Returns modified netlist text.
    Raises ValueError on probe resolution errors.
    """
    lines = netlist_text.splitlines(keepends=True)
    netlist = SpiceNetlist.parse(lines, base_dir)
    path_parts = probe_spec.split(".")
    if len(path_parts) < 2:
        raise ValueError(f"Probe must be 'instance.pin', got '{probe_spec}'")

    original_net, new_net, stb_x, scope = resolve_hierarchical_probe(netlist, path_parts)

    # Zero existing AC sources before inserting probe
    netlist.zero_ac_sources()
    if scope is not None:
        ac_pat = re.compile(r"(\bac\s+)(\S+)", re.IGNORECASE)
        for sl in scope.body_lines:
            if sl.text and sl.text[0].lower() in ("v", "i"):
                if re.search(r"\bac\b", sl.text, re.IGNORECASE):
                    sl.text = ac_pat.sub(r"\g<1>0", sl.text)

    probe_lines = _build_probe_elements(new_net, original_net, stb_x)
    if scope is None:
        netlist.top_lines.extend(probe_lines)
    else:
        scope.body_lines.extend(probe_lines)

    inside_subckt = (scope is not None)
    subckt_inst_path = None
    if inside_subckt:
        subckt_inst_path = []
        for p in path_parts[:-2]:
            name = p.lower()
            if not name.startswith("x"):
                name = "x" + name
            subckt_inst_path.append(name)

    ctrl = _build_tian_control(stb_x, raw_file, fstart, fstop, pts,
                                inside_subckt, subckt_inst_path)
    out = netlist.to_text()
    out = out.replace(".end\n", ctrl + "\n.end\n")
    return out


# ---------------------------------------------------------------------------
# ng_ directive parsing
# ---------------------------------------------------------------------------

class NgConfig:
    """Configuration from ng_ directives."""

    def __init__(self):
        self.params: Dict[str, List[str]] = {}
        self.libs: Dict[Tuple[str, Optional[str]], List[str]] = {}
        self.temps: List[str] = []
        self.outputs: List[str] = []
        self.stb: Optional[Dict] = None  # {probe, fstart, fstop, pts}

    @property
    def has_corners(self):
        return bool(self.params or self.libs or self.temps)

    @property
    def has_out(self):
        return bool(self.outputs)

    @property
    def has_stb(self):
        return self.stb is not None


def parse_ng_directives(lines: List[str]) -> NgConfig:
    config = NgConfig()
    for line in lines:
        s = line.strip()
        if not s.startswith("*"):
            continue
        content = s.lstrip("*").strip()
        if not content.startswith("ngr_"):
            continue
        parts = content.split()
        if len(parts) < 2:
            continue
        cmd, args = parts[0], parts[1:]

        if cmd == "ngr_param":
            if len(args) < 2:
                print(f"Warning: ngr_param requires name + value(s): {s}")
                continue
            config.params[args[0]] = args[1:]

        elif cmd == "ngr_lib":
            if len(args) < 2:
                print(f"Warning: ngr_lib requires libfile + corner(s): {s}")
                continue
            m = re.match(r'^([^()]+)(?:\(([^)]+)\))?$', args[0])
            if m:
                config.libs[(m.group(1), m.group(2))] = args[1:]
            else:
                print(f"Warning: invalid ngr_lib spec: {args[0]}")

        elif cmd == "ngr_temp":
            config.temps = args

        elif cmd == "ngr_out":
            config.outputs = args

        elif cmd == "ngr_stb":
            if config.stb is not None:
                print("Warning: multiple ngr_stb directives; only the first is used.")
                continue
            if not args:
                print(f"Warning: ngr_stb requires a probe spec: {s}")
                continue
            stb = {"probe": args[0], "fstart": 1.0, "fstop": 1e9, "pts": 100}
            for kv in args[1:]:
                m = re.match(r'^(fstart|fstop|pts)=(\S+)$', kv, re.IGNORECASE)
                if m:
                    k = m.group(1).lower()
                    try:
                        stb[k] = int(m.group(2)) if k == "pts" else float(m.group(2))
                    except ValueError:
                        print(f"Warning: invalid value for ngr_stb {k}: {m.group(2)}")
                else:
                    print(f"Warning: unknown ngr_stb option: {kv}")
            config.stb = stb

    return config


# ---------------------------------------------------------------------------
# Corner generation and netlist creation
# ---------------------------------------------------------------------------

class CornerGenerator:
    def __init__(self, lines: List[str], config: NgConfig, base_netlist: str):
        self.lines = lines
        self.config = config
        self.base_netlist = base_netlist

    def generate_corners(self) -> List[Dict]:
        param_names = sorted(self.config.params)
        param_vals  = [self.config.params[n] for n in param_names]
        lib_keys    = sorted(self.config.libs)
        lib_vals    = [self.config.libs[k] for k in lib_keys]
        temps       = self.config.temps or ["25"]

        corners = []
        cid = 0
        for temp in temps:
            for pcomb in (itertools.product(*param_vals) if param_vals else [()]):
                for lcomb in (itertools.product(*lib_vals) if lib_vals else [()]):
                    cid += 1
                    corners.append({
                        "id":          f"c{cid:04d}",
                        "temperature": temp,
                        "params":      dict(zip(param_names, pcomb)) if param_names else {},
                        "libs":        dict(zip(lib_keys, lcomb))    if lib_keys    else {},
                    })
        return corners

    def build_corner_text(self, corner: Dict) -> str:
        modified = []
        temp_inserted = False
        for line in self.lines:
            s = line.strip()
            if s.lstrip("*").strip().startswith("ngr_"):
                modified.append(line)
                continue
            ml = line
            for pname, pval in corner["params"].items():
                pat = r'^(\s*\.param\s+' + re.escape(pname) + r'\s*=\s*)(\S+)(.*)'
                m = re.match(pat, ml, re.IGNORECASE)
                if m:
                    ml = f"{m.group(1)}{pval}{m.group(3)}\n"
            for (libfile, key), cval in corner["libs"].items():
                pat = r'^(\s*\.lib\s+)(.*)/' + re.escape(libfile) + r'(\s+)(\S+)(.*)'
                m = re.match(pat, ml, re.IGNORECASE)
                if m:
                    if key is None or m.group(4).strip() == key.strip():
                        ml = f"{m.group(1)}{m.group(2)}/{libfile}{m.group(3)}{cval}{m.group(5)}\n"
            if re.match(r'^\s*\.temp\s', ml, re.IGNORECASE):
                ml = f".temp {corner['temperature']}\n"
                temp_inserted = True
            modified.append(ml)
            if not temp_inserted and re.match(r'^\s*\.(tran|ac|dc|op)\s', ml, re.IGNORECASE):
                modified[-1] = f".temp {corner['temperature']}\n"
                modified.append(ml)
                temp_inserted = True
        if not temp_inserted:
            for i in range(len(modified) - 1, -1, -1):
                if re.match(r'^\s*\.end', modified[i], re.IGNORECASE):
                    modified.insert(i, f".temp {corner['temperature']}\n")
                    break
            else:
                modified.append(f".temp {corner['temperature']}\n")
        return "".join(modified)

    def write_corner_netlist(self, corner: Dict, path: str):
        with open(path, "w") as f:
            f.write(self.build_corner_text(corner))

    def write_tian_netlist(self, corner: Dict, path: str, raw_file: str, base_dir: str):
        stb = self.config.stb
        text = instrument_netlist_tian(
            self.build_corner_text(corner),
            stb["probe"], stb["fstart"], stb["fstop"], stb["pts"],
            raw_file, base_dir)
        with open(path, "w") as f:
            f.write(text)


# ---------------------------------------------------------------------------
# Simulation execution and result extraction
# ---------------------------------------------------------------------------

STB_MEASURES = ["a0_db", "ugf_freq", "pm", "gm_freq", "gm_db"]


def _run_ngspice(path: str, timeout: int = 300) -> Tuple[str, str, int]:
    r = subprocess.run(["ngspice", "-b", path],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout, r.stderr, r.returncode


def _extract_measures(output: str, names: List[str]) -> Dict[str, str]:
    out = {}
    for name in names:
        pat = re.compile(r'^\s*' + re.escape(name) + r'\s*=\s*(\S+)',
                         re.MULTILINE | re.IGNORECASE)
        m = pat.search(output)
        out[name] = m.group(1) if m else "N/A"
    return out


def _extract_tian_measures(stdout: str, stderr: str) -> Dict[str, str]:
    """
    Extract Tian results.
    a0_db / ugf_freq / gm_freq come from .meas; pm / gm_db from print.
    All share the same regex pattern.
    """
    combined = stdout + "\n" + stderr
    return _extract_measures(combined, STB_MEASURES)


def print_tian_summary(measures: Dict[str, str], corner_id: str = ""):
    label = f" [{corner_id}]" if corner_id else ""
    print()
    print("=" * 52)
    print(f"  Tian Stability Results{label}")
    print("=" * 52)

    def fmt_freq(v):
        try:
            f = float(v)
            return (f"{f/1e6:.4f} MHz" if f >= 1e6 else
                    f"{f/1e3:.4f} kHz" if f >= 1e3 else f"{f:.4f} Hz")
        except (ValueError, TypeError):
            return str(v)

    a0     = measures.get("a0_db",   "N/A")
    ugf    = measures.get("ugf_freq","N/A")
    pm     = measures.get("pm",      "N/A")
    gm_f   = measures.get("gm_freq", "N/A")
    gm_db  = measures.get("gm_db",   "N/A")

    print(f"  DC Loop Gain (a0):   {a0 if a0 == 'N/A' else a0+' dB'}")
    print(f"  Unity Gain Freq:     {fmt_freq(ugf) if ugf != 'N/A' else 'NOT FOUND'}")

    if pm != "N/A":
        try:
            pf = float(pm)
            warn = ("  *** UNSTABLE ***"        if pf < 0  else
                    "  *** WARNING: may ring ***" if pf < 30 else
                    "  *** CAUTION: < 45 deg ***" if pf < 45 else "")
            print(f"  Phase Margin:        {pf:.2f} deg{warn}")
        except ValueError:
            print(f"  Phase Margin:        {pm}")
    else:
        print("  Phase Margin:        NOT FOUND")

    if gm_f != "N/A" and gm_db != "N/A":
        print(f"  Gain Margin Freq:    {fmt_freq(gm_f)}")
        try:
            gf = float(gm_db)
            warn = ("  *** UNSTABLE ***"        if gf < 0 else
                    "  *** CAUTION: < 6 dB ***" if gf < 6 else "")
            print(f"  Gain Margin:         {gf:.2f} dB{warn}")
        except ValueError:
            print(f"  Gain Margin:         {gm_db}")
    else:
        print("  Gain Margin:         NOT FOUND (no -180 deg crossing)")

    print("=" * 52)


# ---------------------------------------------------------------------------
# Per-corner worker (normal + optional Tian)
# ---------------------------------------------------------------------------

def _run_corner_worker(args_tuple) -> Dict:
    """
    Run one corner.  Designed to be called in-process or via ProcessPoolExecutor.
    args_tuple: (corner, normal_path, tian_path, out_measures, has_out, has_stb)
    normal_path / tian_path may be None.
    """
    corner, normal_path, tian_path, out_measures, has_out, has_stb = args_tuple

    row = {
        "corner_id":   corner["id"],
        "temperature": corner["temperature"],
        **{f"param_{k}": v for k, v in corner["params"].items()},
        **{f"lib_{k[0]}{'_'+k[1] if k[1] else ''}": v
           for k, v in corner["libs"].items()},
    }

    if has_out and normal_path:
        try:
            stdout, stderr, rc = _run_ngspice(normal_path)
            if rc != 0:
                row.update({m: "SIM_ERROR" for m in out_measures})
            else:
                row.update(_extract_measures(stdout + "\n" + stderr, out_measures))
        except subprocess.TimeoutExpired:
            row.update({m: "TIMEOUT" for m in out_measures})
        except Exception:
            row.update({m: "ERROR" for m in out_measures})

    if has_stb and tian_path:
        try:
            stdout, stderr, rc = _run_ngspice(tian_path)
            if rc != 0:
                row.update({m: "SIM_ERROR" for m in STB_MEASURES})
            else:
                row.update(_extract_tian_measures(stdout, stderr))
        except subprocess.TimeoutExpired:
            row.update({m: "TIMEOUT" for m in STB_MEASURES})
        except Exception:
            row.update({m: "ERROR" for m in STB_MEASURES})

    return row


# ---------------------------------------------------------------------------
# Typical (--typ) mode
# ---------------------------------------------------------------------------

def run_typ(netlist_path: str, config: NgConfig, output_file: str):
    base_dir  = os.path.dirname(os.path.abspath(netlist_path))
    base_name = os.path.splitext(os.path.basename(netlist_path))[0]

    with open(netlist_path) as f:
        netlist_text = f.read()

    row = {"corner_id": "typ", "temperature": "typ"}

    if not config.has_out and not config.has_stb:
        print("[ngrun] Warning: no ngr_out or ngr_stb - nothing to extract.")

    # Normal simulation
    if config.has_out:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sp",
                                         prefix="ngrun_typ_", delete=False) as tf:
            tf.write(netlist_text)
            npath = tf.name
        try:
            print("[ngrun] Running normal simulation (--typ)...")
            stdout, stderr, rc = _run_ngspice(npath)
            if rc != 0:
                print(f"[ngrun] ngspice error (rc={rc})")
                print((stdout + stderr)[-3000:])
                row.update({m: "SIM_ERROR" for m in config.outputs})
            else:
                row.update(_extract_measures(stdout + "\n" + stderr, config.outputs))
                print(f"[ngrun] Measures: { {k: row[k] for k in config.outputs} }")
        finally:
            os.unlink(npath)

    # Tian simulation
    if config.has_stb:
        stb = config.stb
        raw_file = os.path.join(base_dir, f"{base_name}_typ_tian.raw")
        try:
            tian_text = instrument_netlist_tian(
                netlist_text, stb["probe"],
                stb["fstart"], stb["fstop"], stb["pts"],
                raw_file, base_dir)
        except ValueError as e:
            print(f"[ngrun] Error instrumenting netlist: {e}", file=sys.stderr)
            row.update({m: "PROBE_ERROR" for m in STB_MEASURES})
            tian_text = None

        if tian_text:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sp",
                                             prefix="ngrun_typ_tian_", delete=False) as tf:
                tf.write(tian_text)
                tpath = tf.name
            try:
                print("[ngrun] Running Tian stability simulation (--typ)...")
                stdout, stderr, rc = _run_ngspice(tpath)
                if rc != 0:
                    print(f"[ngrun] ngspice error (rc={rc})")
                    print((stdout + stderr)[-3000:])
                    row.update({m: "SIM_ERROR" for m in STB_MEASURES})
                else:
                    stb_res = _extract_tian_measures(stdout, stderr)
                    row.update(stb_res)
                    print_tian_summary(stb_res)
                    if os.path.isfile(raw_file):
                        print(f"[ngrun] Raw file: {raw_file}")
            finally:
                os.unlink(tpath)

    _write_csv(output_file, [row], config)
    print(f"[ngrun] Results written to: {output_file}")


# ---------------------------------------------------------------------------
# Corner sweep mode
# ---------------------------------------------------------------------------

def run_corners(netlist_path: str, config: NgConfig, output_file: str,
                parallel: int, keep_netlists: bool, no_run: bool):

    base_dir  = os.path.dirname(os.path.abspath(netlist_path))
    base_name = os.path.splitext(os.path.basename(netlist_path))[0]

    with open(netlist_path) as f:
        lines = f.readlines()

    generator = CornerGenerator(lines, config, netlist_path)
    corners   = generator.generate_corners()
    n         = len(corners)

    print(f"[2/5] {n} corner combination(s)")
    if not config.has_corners:
        print("  (no ngr_param/ngr_lib/ngr_temp - single nominal corner)")

    temp_dir = tempfile.mkdtemp(prefix="ngrun_")
    print(f"[3/5] Creating netlists in: {temp_dir}")

    sim_args = []
    for corner in corners:
        cid = corner["id"]
        normal_path = tian_path = None

        if config.has_out or not config.has_stb:
            normal_path = os.path.join(temp_dir, f"{cid}_norm.sp")
            generator.write_corner_netlist(corner, normal_path)

        if config.has_stb:
            raw_file  = os.path.join(base_dir, f"{base_name}_{cid}_tian.raw")
            tian_path = os.path.join(temp_dir, f"{cid}_tian.sp")
            try:
                generator.write_tian_netlist(corner, tian_path, raw_file, base_dir)
            except ValueError as e:
                print(f"  Warning: Tian probe error for {cid}: {e}")
                tian_path = None

        sim_args.append((corner, normal_path, tian_path,
                         config.outputs, config.has_out, config.has_stb))

    print(f"  {n} netlist(s) created")

    if no_run:
        print("[4/5] Skipping simulations (--no-run)")
        print("[5/5] No results to write")
        if not keep_netlists:
            import shutil; shutil.rmtree(temp_dir)
        else:
            print(f"  Netlists in: {temp_dir}")
        return

    print(f"[4/5] Running simulations (parallel={parallel})...")
    results = []

    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as ex:
            futures = {ex.submit(_run_corner_worker, a): a for a in sim_args}
            done = 0
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"  Corner failed: {e}")
                done += 1
                if done % max(1, n // 10) == 0 or done == n:
                    print(f"  {done}/{n} ({100*done//n}%)")
    else:
        for i, arg in enumerate(sim_args):
            results.append(_run_corner_worker(arg))
            done = i + 1
            if done % max(1, n // 10) == 0 or done == n:
                print(f"  {done}/{n} ({100*done//n}%)")

    results.sort(key=lambda r: r["corner_id"])

    print(f"[5/5] Writing results to: {output_file}")
    _write_csv(output_file, results, config)
    print(f"  {len(results)} row(s) written")

    if not keep_netlists:
        import shutil; shutil.rmtree(temp_dir)
    else:
        print(f"  Netlists preserved in: {temp_dir}")

    # Print stability summaries to terminal
    if config.has_stb:
        print()
        for r in results:
            print_tian_summary({m: r.get(m, "N/A") for m in STB_MEASURES},
                                corner_id=r["corner_id"])


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_csv(path: str, results: List[Dict], config: NgConfig):
    if not results:
        print("  No results to write.")
        return
    # Build deterministic column order
    cols = ["corner_id", "temperature"]
    for k in sorted({k for r in results for k in r if k.startswith("param_")}):
        cols.append(k)
    for k in sorted({k for r in results for k in r if k.startswith("lib_")}):
        cols.append(k)
    for m in config.outputs:
        if m not in cols:
            cols.append(m)
    if config.has_stb:
        for m in STB_MEASURES:
            if m not in cols:
                cols.append(m)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ngrun - NGSpice Corner + Stability Simulation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("netlist", help="Input netlist file")
    parser.add_argument("--typ", action="store_true",
                        help="Run single typical simulation (no corner substitution)")
    parser.add_argument("-k", "--keep-netlists", action="store_true",
                        help="Keep generated netlists in temp directory")
    parser.add_argument("-j", "--parallel", type=int, default=1, metavar="N",
                        help="Run N corners in parallel (default: 1)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV file (default: <netlist>_results.csv)")
    parser.add_argument("-n", "--no-run", action="store_true",
                        help="Generate netlists only, do not simulate (corner mode)")
    args = parser.parse_args()

    if not os.path.isfile(args.netlist):
        print(f"Error: '{args.netlist}' not found.", file=sys.stderr)
        sys.exit(1)

    base = os.path.splitext(os.path.basename(args.netlist))[0]
    output_file = args.output or f"{base}_results.csv"

    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                    NGRUN - NGSpice Simulation Tool                        ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()

    print(f"[1/5] Parsing: {args.netlist}")
    with open(args.netlist) as f:
        lines = f.readlines()
    config = parse_ng_directives(lines)

    print(f"  ngr_param:  {len(config.params)} parameter(s): "
          f"{list(config.params.keys()) or 'none'}")
    print(f"  ngr_lib:    {len(config.libs)} library/libraries")
    print(f"  ngr_temp:   {config.temps or ['25 (default)']}")
    print(f"  ngr_out:    {config.outputs or ['(none)']}")
    print(f"  ngr_stb:    {config.stb['probe'] if config.has_stb else '(none)'}")
    print()

    if not config.has_out and not config.has_stb:
        print("  Warning: no ngr_out or ngr_stb - nothing will be extracted from results")

    if config.has_out and config.has_stb:
        print("  Note: ngr_out + ngr_stb present - 2 simulations per corner")
        print()

    print(f"  Mode: {'--typ (single run)' if args.typ else 'corner sweep'}")
    print()

    if args.typ:
        run_typ(args.netlist, config, output_file)
    else:
        run_corners(args.netlist, config, output_file,
                    args.parallel, args.keep_netlists, args.no_run)

    print()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                             Complete!                                    ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
