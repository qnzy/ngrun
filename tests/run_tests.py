#!/usr/bin/env python3
"""
Self-checking test suite for ngrun.

Two layers:

  unit  - direct calls into ngrun's helpers, no simulator needed
  gen   - run ngrun with -n -k and inspect the generated corner netlists
  sim   - full ngspice runs, results checked against closed-form answers

The sim layer is skipped automatically if ngspice is not on PATH.

Usage:
    python3 run_tests.py [--layer unit|gen|sim|all] [-v] [-k PATTERN]

Exit code is 0 only if every selected test passed.
"""

import argparse
import contextlib
import csv
import io
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NETLISTS = os.path.join(HERE, "netlists")
NGRUN = os.path.join(ROOT, "ngrun.py")

sys.path.insert(0, ROOT)
import ngrun  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny test framework
# ---------------------------------------------------------------------------

_TESTS = []
_VERBOSE = False


class Fail(AssertionError):
    pass


def test(layer):
    def deco(fn):
        _TESTS.append((layer, fn.__name__, fn))
        return fn
    return deco


def check(cond, msg):
    if not cond:
        raise Fail(msg)


def check_close(got, want, tol_rel, msg):
    try:
        g = float(got)
    except (TypeError, ValueError):
        raise Fail(f"{msg}: value {got!r} is not numeric")
    if want == 0:
        ok = abs(g) <= tol_rel
    else:
        ok = abs(g - want) / abs(want) <= tol_rel
    if not ok:
        raise Fail(f"{msg}: got {g:.6g}, want {want:.6g} "
                   f"(rel err {abs(g-want)/abs(want):.3e} > {tol_rel:.0e})")


def note(msg):
    if _VERBOSE:
        print(f"        {msg}")


# ---------------------------------------------------------------------------
# Helpers for driving ngrun
# ---------------------------------------------------------------------------

def stage_netlist(name, workdir):
    """Copy a netlist into workdir, expanding @TESTDIR@ to an absolute path.

    ngrun writes corner netlists into a temp directory, so any relative
    .lib/.include path in the source netlist would no longer resolve.  The
    test netlists use @TESTDIR@ to sidestep that.
    """
    src = os.path.join(NETLISTS, name)
    dst = os.path.join(workdir, name)
    with open(src) as f:
        text = f.read()
    text = text.replace("@TESTDIR@", NETLISTS)
    with open(dst, "w") as f:
        f.write(text)
    return dst


def run_ngrun(netlist, *args, workdir=None, expect_rc=0):
    cmd = [sys.executable, NGRUN, netlist] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=workdir or os.path.dirname(netlist), timeout=900)
    if expect_rc is not None and r.returncode != expect_rc:
        raise Fail(f"ngrun exit {r.returncode}, expected {expect_rc}\n"
                   f"--- stdout ---\n{r.stdout[-2000:]}\n"
                   f"--- stderr ---\n{r.stderr[-2000:]}")
    return r


def temp_dir_from_output(stdout):
    m = re.search(r'(?:Netlists in|Temp directory preserved):\s*(\S+)', stdout)
    if not m:
        raise Fail("could not find the preserved temp directory in ngrun output")
    return m.group(1)


def generated_netlists(stdout):
    d = temp_dir_from_output(stdout)
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".sp"):
            with open(os.path.join(d, fn)) as f:
                out[fn] = f.read()
    return d, out


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def have_ngspice():
    return shutil.which("ngspice") is not None


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

TWOPI = 2 * math.pi


def rc_f3db(rr, cc, kfac, temp_c, tc1=1e-3, tnom=27.0):
    """-3 dB frequency of the rc_corners.sp lowpass."""
    r = rr * kfac * (1 + tc1 * (temp_c - tnom))
    return 1.0 / (TWOPI * r * cc)


def suffix_to_float(s):
    """Minimal SPICE suffix parser for the values used in the test netlists."""
    s = s.strip().lower()
    mults = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3,
             "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
    m = re.match(r'^([-+0-9.eE]+)\s*([a-z]*)$', s)
    if not m:
        raise ValueError(s)
    val = float(m.group(1))
    suf = m.group(2)
    if not suf:
        return val
    for k in ("meg", "t", "g", "k", "m", "u", "n", "p", "f"):
        if suf.startswith(k):
            return val * mults[k]
    raise ValueError(s)


KFAC = {"tt": 1.0, "ff": 0.8, "ss": 1.25}


class Loop3:
    """Closed-form loop gain of loop3.sp / loop3_hier.sp."""
    A0 = 1000.0
    POLES = (100.0, 1e5, 1e6)

    @classmethod
    def mag(cls, f):
        d = 1.0
        for fi in cls.POLES:
            d *= math.hypot(1.0, f / fi)
        return cls.A0 / d

    @classmethod
    def phase_deg(cls, f):
        # Continuous (unwrapped) phase: -sum(atan(f/fi))
        return -sum(math.degrees(math.atan(f / fi)) for fi in cls.POLES)

    @staticmethod
    def _bisect(fn, lo, hi):
        for _ in range(300):
            m = math.sqrt(lo * hi)
            if fn(m) > 0:
                lo = m
            else:
                hi = m
        return math.sqrt(lo * hi)

    @classmethod
    def reference(cls):
        ugf = cls._bisect(lambda f: cls.mag(f) - 1.0, 1e2, 1e8)
        gmf = cls._bisect(lambda f: cls.phase_deg(f) + 180.0, 1e2, 1e9)
        return {
            "a0_db":    20 * math.log10(cls.A0),
            "ugf_freq": ugf,
            "pm":       180.0 + cls.phase_deg(ugf),
            "gm_freq":  gmf,
            "gm_db":    -20 * math.log10(cls.mag(gmf)),
        }


# ---------------------------------------------------------------------------
# Unit tests - ngrun helpers
# ---------------------------------------------------------------------------

@test("unit")
def unit_subst_param_multi_assignment():
    """A swept param is substituted anywhere on a multi-assignment line."""
    line = ".param vdd=3.3 cload=10p rload=2k\n"
    out, n = ngrun._subst_param_assignment(line, "cload", "100p")
    check(n == 1, f"expected 1 substitution, got {n}")
    check(out == ".param vdd=3.3 cload=100p rload=2k\n", f"got {out!r}")


@test("unit")
def unit_subst_param_braced_value():
    """Braced and quoted expressions are replaced as one token."""
    out, n = ngrun._subst_param_assignment(".param a={1k*2} b=3\n", "a", "5k")
    check(out == ".param a=5k b=3\n", f"braced: got {out!r}")
    out, n = ngrun._subst_param_assignment(".param a='1+2' b=3\n", "a", "9")
    check(out == ".param a=9 b=3\n", f"quoted: got {out!r}")


@test("unit")
def unit_subst_param_no_partial_name_match():
    """'cload' must not match 'xcload' or 'cload2'."""
    line = ".param xcload=1p cload2=2p cload=3p\n"
    out, n = ngrun._subst_param_assignment(line, "cload", "99p")
    check(n == 1, f"expected exactly 1 substitution, got {n}: {out!r}")
    check(out == ".param xcload=1p cload2=2p cload=99p\n", f"got {out!r}")


@test("unit")
def unit_param_state_comment_terminates():
    """A comment ends a .param statement; it must not stay 'inside' it."""
    lines = [".param a=1\n", "* b=2 is only a comment\n", "R1 1 0 1k\n"]
    st = False
    states = []
    for ln in lines:
        st = ngrun._param_state(ln, st)
        states.append(st)
    check(states == [True, False, False], f"states={states}")


@test("unit")
def unit_param_state_continuation():
    """'+' continues a .param only if the previous line was part of it."""
    check(ngrun._param_state("+ d=4\n", True) is True, "continuation after .param")
    check(ngrun._param_state("+ d=4\n", False) is False, "orphan continuation")


@test("unit")
def unit_collect_defined_params():
    """Continuations count; comment-only assignments do not."""
    lines = [".param a=1\n", "* b=2 is only a comment\n", "R1 1 0 1k\n",
             ".param c=3\n", "+ d=4\n", "* e=5\n", "+ f=6\n"]
    got = ngrun.collect_defined_params(lines)
    check(got == {"a", "c", "d"}, f"got {sorted(got)}, want ['a','c','d']")


@test("unit")
def unit_extract_measures_last_match():
    """The last 'name = value' wins, not the first."""
    out = "\nf3db = 0.000000e+00\nsome noise\nf3db = 1.591549e+05\n"
    got = ngrun._extract_measures(out, ["f3db"])
    check(got["f3db"] == "1.591549e+05", f"got {got}")


@test("unit")
def unit_extract_measures_missing():
    got = ngrun._extract_measures("nothing here\n", ["vout"])
    check(got["vout"] == "N/A", f"got {got}")


@test("unit")
def unit_error_row_covers_every_column():
    """A dead worker still yields one fully-populated row."""
    corner = {"id": "c0007", "temperature": None,
              "params": {"vdd": "3.3"}, "libs": {("models.lib", "tt"): "ff"}}
    args = (corner, "/x/n.sp", [("/x/t1.sp", ""), ("/x/t2.sp", "_2")],
            ["vout"], True, True)
    row = ngrun._error_row(args)
    check(row["corner_id"] == "c0007", "corner id")
    check(row["temperature"] == "default", f"temperature={row['temperature']!r}")
    check(row["param_vdd"] == "3.3", "param column")
    check(row["lib_models.lib_tt"] == "ff", f"lib column missing: {sorted(row)}")
    for col in ["vout", "a0_db", "pm", "gm_db", "a0_db_2", "pm_2", "gm_db_2"]:
        check(row[col] == "WORKER_ERROR", f"column {col} not tagged")


@test("unit")
def unit_probe_spec_forms():
    check(ngrun._parse_probe_spec("amp.out") == ["amp", "out"], "dot form")
    check(ngrun._parse_probe_spec("amp:2") == ["amp", ":2"], "colon form")
    check(ngrun._parse_probe_spec("core.amp.out") == ["core", "amp", "out"],
          "hierarchical dot form")
    check(ngrun._parse_probe_spec("core.amp:1") == ["core", "amp", ":1"],
          "hierarchical colon form")


# ---------------------------------------------------------------------------
# Generation tests - inspect the netlists ngrun writes
# ---------------------------------------------------------------------------

@test("gen")
def gen_corners_are_all_distinct(work):
    """The core regression: swept params must actually differ per netlist."""
    nl = stage_netlist("rc_corners.sp", work)
    r = run_ngrun(nl, "-n", "-k", workdir=work)
    d, files = generated_netlists(r.stdout)
    check(len(files) == 2 * 2 * 3 * 3, f"expected 36 netlists, got {len(files)}")
    bodies = {}
    for name, text in files.items():
        bodies.setdefault(text, []).append(name)
    dupes = {k: v for k, v in bodies.items() if len(v) > 1}
    check(not dupes,
          "identical netlists generated for different corners: "
          + "; ".join(", ".join(v) for v in dupes.values()))
    note(f"{len(files)} netlists, all distinct")


@test("gen")
def gen_param_values_match_corner(work):
    """Each generated netlist carries the params its CSV row will claim."""
    nl = stage_netlist("rc_corners.sp", work)
    r = run_ngrun(nl, "-n", "-k", workdir=work)
    d, files = generated_netlists(r.stdout)
    seen = set()
    for name, text in files.items():
        cc = re.search(r'^\.param\s+.*\bcc=(\S+)', text, re.M | re.I)
        rr = re.search(r'^\+\s*rr=(\S+)', text, re.M | re.I)
        check(cc is not None, f"{name}: cc assignment not found")
        check(rr is not None, f"{name}: rr continuation assignment not found")
        seen.add((cc.group(1), rr.group(1)))
    want = {("1n", "1k"), ("1n", "4k"), ("10n", "1k"), ("10n", "4k")}
    check(seen == want, f"param combinations {sorted(seen)} != {sorted(want)}")


@test("gen")
def gen_temp_untouched_without_directive(work):
    """No ngr_temp -> the netlist's own .temp survives and none is injected."""
    nl = stage_netlist("temp_fixed.sp", work)
    r = run_ngrun(nl, "-n", "-k", workdir=work)
    d, files = generated_netlists(r.stdout)
    for name, text in files.items():
        temps = re.findall(r'^\s*\.temp\s+(\S+)', text, re.M | re.I)
        check(temps == ["125"],
              f"{name}: expected exactly one '.temp 125', found {temps}")


@test("gen")
def gen_temp_applied_with_directive(work):
    """ngr_temp present -> exactly one .temp per netlist, with swept values."""
    nl = stage_netlist("rc_corners.sp", work)
    r = run_ngrun(nl, "-n", "-k", workdir=work)
    d, files = generated_netlists(r.stdout)
    found = set()
    for name, text in files.items():
        temps = re.findall(r'^\s*\.temp\s+(\S+)', text, re.M | re.I)
        check(len(temps) == 1, f"{name}: expected 1 .temp, found {temps}")
        found.add(temps[0])
    check(found == {"-40", "27", "125"}, f"temperatures seen: {sorted(found)}")


@test("gen")
def gen_lib_key_substitution(work):
    """.lib key is swept while the absolute path is preserved."""
    nl = stage_netlist("rc_corners.sp", work)
    r = run_ngrun(nl, "-n", "-k", workdir=work)
    d, files = generated_netlists(r.stdout)
    keys = set()
    for name, text in files.items():
        m = re.search(r'^\s*\.lib\s+(\S+)\s+(\S+)', text, re.M | re.I)
        check(m is not None, f"{name}: no .lib line")
        check(m.group(1) == os.path.join(NETLISTS, "models.lib"),
              f"{name}: library path was rewritten to {m.group(1)!r}")
        keys.add(m.group(2))
    check(keys == {"tt", "ff", "ss"}, f"library corners seen: {sorted(keys)}")


@test("gen")
def gen_comments_are_not_rewritten(work):
    """A comment that looks like an assignment must be left alone."""
    src = os.path.join(work, "commented.sp")
    with open(src, "w") as f:
        f.write("* comment regression\n"
                "** ngr_param cc 1n 10n\n"
                "** ngr_out f3db\n"
                ".param cc=1n\n"
                "* cc=99p appears only in this comment\n"
                "R1 in out 1k\n"
                "C1 out 0 {cc}\n"
                ".ac dec 10 1 1e6\n"
                ".end\n")
    r = run_ngrun(src, "-n", "-k", workdir=work)
    d, files = generated_netlists(r.stdout)
    for name, text in files.items():
        check("* cc=99p appears only in this comment\n" in text,
              f"{name}: the comment line was modified")


@test("gen")
def gen_bad_param_is_fatal(work):
    """An ngr_param with no matching .param must stop the run, not warn."""
    src = os.path.join(work, "typo.sp")
    with open(src, "w") as f:
        f.write("* typo regression\n"
                "** ngr_param typo 1 2\n"
                "** ngr_out f3db\n"
                ".param cc=1n\n"
                "R1 in out 1k\n"
                "C1 out 0 {cc}\n"
                ".ac dec 10 1 1e6\n"
                ".end\n")
    r = run_ngrun(src, "-n", expect_rc=2, workdir=work)
    check("ERROR" in r.stderr, "expected an ERROR message on stderr")
    # --force overrides
    run_ngrun(src, "-n", "--force", expect_rc=0, workdir=work)
    # --typ does no substitution, so validation must not apply
    r = run_ngrun(src, "--typ", expect_rc=None, workdir=work)
    check(r.returncode != 2,
          f"--typ must bypass ngr_param validation, exit was {r.returncode}")


def _dead_worker_run(work, parallel):
    """Run a sweep in which one corner's worker raises, and return the rows.

    broken.sp covers the case where ngspice itself fails (the worker catches
    that internally).  This covers the other case: the worker dying outright,
    which is what used to drop the corner from the CSV entirely.
    """
    src = os.path.join(work, "flaky.sp")
    with open(src, "w") as f:
        f.write("* dead worker regression\n"
                "** ngr_param rr 1k 2k 4k\n"
                "** ngr_out f3db\n"
                ".param rr=1k\n"
                "V1 in 0 DC 0 AC 1\n"
                "R1 in out {rr}\n"
                "C1 out 0 1n\n"
                ".ac dec 10 1 1e6\n"
                ".meas ac f3db when v(out) = 0.5 cross=1\n"
                ".end\n")
    with open(src) as f:
        config = ngrun.parse_ng_directives(f.readlines())
    out_csv = os.path.join(work, f"flaky_p{parallel}.csv")

    original = ngrun._run_corner_worker

    def flaky(args_tuple):
        corner = args_tuple[0]
        if corner["id"] == "c0002":
            raise RuntimeError("simulated worker crash")
        row = ngrun._base_row(corner)
        row["f3db"] = "1.234e+05"
        return row

    ngrun._run_corner_worker = flaky
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ngrun.run_corners(src, config, out_csv, parallel, False, False)
    finally:
        ngrun._run_corner_worker = original
    return read_csv(out_csv)


@test("gen")
def gen_dead_worker_still_yields_a_row_serial(work):
    rows = _dead_worker_run(work, parallel=1)
    ids = [r["corner_id"] for r in rows]
    check(ids == ["c0001", "c0002", "c0003"],
          f"a corner was dropped from the CSV: {ids}")
    bad = [r for r in rows if r["corner_id"] == "c0002"][0]
    check(bad["f3db"] == "WORKER_ERROR", f"c0002 f3db={bad['f3db']!r}")
    check(bad["param_rr"] == "2k", "the error row lost its corner identity")


@test("gen")
def gen_dead_worker_still_yields_a_row_parallel(work):
    # ProcessPoolExecutor forks on Linux, so the patched worker is inherited.
    rows = _dead_worker_run(work, parallel=3)
    ids = sorted(r["corner_id"] for r in rows)
    check(ids == ["c0001", "c0002", "c0003"],
          f"a corner was dropped from the CSV: {ids}")
    bad = [r for r in rows if r["corner_id"] == "c0002"][0]
    check(bad["f3db"] == "WORKER_ERROR", f"c0002 f3db={bad['f3db']!r}")


@test("unit")
def unit_prose_comment_is_not_a_directive():
    """The line that once turned 36 corners into 252."""
    lines = ["* header\n",
             "** ngr_lib models.lib(tt) tt ff ss\n",
             "*            ngr_lib with an explicit key, ngr_temp, ngr_out from .meas.\n"]
    cfg = ngrun.parse_ng_directives(lines)
    check(len(cfg.libs) == 1,
          f"prose comment was absorbed as configuration: {dict(cfg.libs)}")
    check(("models.lib", "tt") in cfg.libs, f"real directive lost: {dict(cfg.libs)}")


@test("unit")
def unit_unknown_directive_is_ignored():
    cfg = ngrun.parse_ng_directives(["** ngr_frobnicate 1 2 3\n"])
    check(not cfg.has_corners and not cfg.has_out and not cfg.has_stb,
          "an unknown ngr_ command was acted on")


@test("unit")
def unit_malformed_directives_are_rejected():
    cases = [
        ("** ngr_param 9bad 1 2\n",        lambda c: not c.params),
        ("** ngr_param cc one two\n",      lambda c: not c.params),
        ("** ngr_temp cold hot\n",         lambda c: not c.temps),
        ("** ngr_out f3db, gain\n",        lambda c: not c.outputs),
        ("** ngr_stb amp.out fstop=soon\n", lambda c: not c.stb_list),
        ("** ngr_stb amp.out bogus=1\n",   lambda c: not c.stb_list),
    ]
    for text, ok in cases:
        cfg = ngrun.parse_ng_directives([text])
        check(ok(cfg), f"accepted malformed directive: {text.strip()}")


@test("unit")
def unit_valid_directives_still_parse():
    """The validation must not reject legitimate syntax."""
    lines = ["** ngr_param cc 1n 10n 1e-9 {2*cc0}\n",
             "** ngr_param vdd 2.7 3.3 3.6\n",
             "** ngr_lib models.lib tt ff ss\n",
             "** ngr_lib /pdk/res.lib(res_nom) res_nom res_hi\n",
             "** ngr_temp -40 27 125.5\n",
             "** ngr_out f3db gain_ratio iq_ua\n",
             "** ngr_stb ldo.erramp.out fstart=0.1 fstop=1e9 pts=50\n",
             "** ngr_stb xr1:2\n"]
    cfg = ngrun.parse_ng_directives(lines)
    check(set(cfg.params) == {"cc", "vdd"}, f"params: {sorted(cfg.params)}")
    check(cfg.params["cc"] == ["1n", "10n", "1e-9", "{2*cc0}"],
          f"cc values: {cfg.params['cc']}")
    check(len(cfg.libs) == 2, f"libs: {dict(cfg.libs)}")
    check(("/pdk/res.lib", "res_nom") in cfg.libs, f"libs: {dict(cfg.libs)}")
    check(cfg.temps == ["-40", "27", "125.5"], f"temps: {cfg.temps}")
    check(cfg.outputs == ["f3db", "gain_ratio", "iq_ua"], f"out: {cfg.outputs}")
    check(len(cfg.stb_list) == 2, f"stb: {cfg.stb_list}")
    check(cfg.stb_list[0]["pts"] == 50 and cfg.stb_list[0]["fstop"] == 1e9,
          f"stb options: {cfg.stb_list[0]}")


@test("unit")
def unit_collect_lib_statements():
    lines = [".lib /pdk/models.lib tt\n", ".lib res.lib res_nom\n",
             "* .lib commented.lib tt\n", ".include /pdk/other.spi\n"]
    got = ngrun.collect_lib_statements(lines)
    check(("models.lib", "tt") in got, f"got {got}")
    check(("res.lib", "res_nom") in got, f"got {got}")
    check(len(got) == 2, f"picked up something extra: {got}")


@test("gen")
def gen_prose_comment_does_not_inflate_corners(work):
    """End-to-end guard for the 36-vs-252 misparse."""
    nl = stage_netlist("rc_corners.sp", work)
    with open(nl) as f:
        text = f.read()
    text = text.replace(
        "** ngr_param cc 1n 10n\n",
        "* Exercises: multi-assignment .param,\n"
        "*            ngr_lib with an explicit key, ngr_temp, ngr_out from .meas.\n"
        "** ngr_param cc 1n 10n\n")
    with open(nl, "w") as f:
        f.write(text)
    r = run_ngrun(nl, "-n", "-k", workdir=work)
    d, files = generated_netlists(r.stdout)
    check(len(files) == 36,
          f"prose comment changed the corner count: got {len(files)}, want 36")
    check("malformed" in r.stdout.lower(),
          "the bogus directive was absorbed without any warning")


@test("gen")
def gen_bad_lib_is_fatal(work):
    """An ngr_lib matching no .lib statement must stop the run."""
    src = os.path.join(work, "badlib.sp")
    with open(src, "w") as f:
        f.write("* bad lib regression\n"
                "** ngr_lib nosuch.lib tt ff\n"
                "** ngr_out f3db\n"
                ".lib @TESTDIR@/models.lib tt\n".replace("@TESTDIR@", NETLISTS) +
                "V1 in 0 DC 0 AC 1\n"
                "R1 in out 1k\n"
                "C1 out 0 1n\n"
                ".ac dec 10 1 1e6\n"
                ".meas ac f3db when v(out) = 0.5 cross=1\n"
                ".end\n")
    r = run_ngrun(src, "-n", expect_rc=2, workdir=work)
    check("nosuch.lib" in r.stderr, f"error did not name the library: {r.stderr}")
    run_ngrun(src, "-n", "--force", expect_rc=0, workdir=work)
    r = run_ngrun(src, "--typ", expect_rc=None, workdir=work)
    check(r.returncode != 2, f"--typ must bypass ngr_lib validation, exit {r.returncode}")


@test("gen")
def gen_bad_lib_key_is_fatal(work):
    """The key half of 'models.lib(key)' is validated too."""
    src = os.path.join(work, "badkey.sp")
    with open(src, "w") as f:
        f.write("* bad lib key regression\n"
                "** ngr_lib models.lib(wrongkey) tt ff\n"
                "** ngr_out f3db\n"
                f".lib {NETLISTS}/models.lib tt\n"
                "V1 in 0 DC 0 AC 1\n"
                "R1 in out 1k\n"
                "C1 out 0 1n\n"
                ".ac dec 10 1 1e6\n"
                ".meas ac f3db when v(out) = 0.5 cross=1\n"
                ".end\n")
    run_ngrun(src, "-n", expect_rc=2, workdir=work)


@test("gen")
def gen_tian_probe_instrumentation(work):
    """Both probe forms instrument the netlist and zero the existing AC source."""
    nl = stage_netlist("loop3.sp", work)
    r = run_ngrun(nl, "-n", "-k", workdir=work)
    d, files = generated_netlists(r.stdout)
    tian = {k: v for k, v in files.items() if "_tian_" in k}
    check(len(tian) == 2, f"expected 2 Tian netlists, got {sorted(tian)}")
    for name, text in tian.items():
        check("Vi_stb" in text and "Ii_stb" in text, f"{name}: probe sources missing")
        check("Vnodebuffer_stb" in text, f"{name}: node buffer missing")
        check(text.count(".control") == 1, f"{name}: expected one .control block")
        check("unwrap(" in text, f"{name}: phase unwrap missing")


@test("gen")
def gen_tian_hierarchical_clone(work):
    """A hierarchical probe clones the intermediate subckt, not the original."""
    nl = stage_netlist("loop3_hier.sp", work)
    r = run_ngrun(nl, "-n", "-k", workdir=work)
    d, files = generated_netlists(r.stdout)
    text = [v for k, v in files.items() if "_tian_" in k][0]
    check(re.search(r'^\.subckt\s+core\b', text, re.M | re.I),
          "original 'core' subckt was not preserved")
    check(re.search(r'^\.subckt\s+core_stb\b', text, re.M | re.I),
          "cloned 'core_stb' subckt was not created")
    check(re.search(r'^Xcore\s+\S+\s+\S+\s+core_stb\b', text, re.M | re.I),
          "instance was not repointed at the clone")
    check("xcore." in text.lower(), "control block lacks the hierarchical path")


# ---------------------------------------------------------------------------
# Simulation tests - full ngspice runs against closed-form answers
# ---------------------------------------------------------------------------

@test("sim")
def sim_corner_sweep_matches_analytic(work):
    """Every one of the 36 corners must match 1/(2*pi*R(T)*C)."""
    nl = stage_netlist("rc_corners.sp", work)
    run_ngrun(nl, "-j", "4", workdir=work)
    rows = read_csv(os.path.join(work, "rc_corners_results.csv"))
    check(len(rows) == 36, f"expected 36 rows, got {len(rows)}")
    for row in rows:
        rr = suffix_to_float(row["param_rr"])
        cc = suffix_to_float(row["param_cc"])
        kf = KFAC[row["lib_models.lib_tt"]]
        want = rc_f3db(rr, cc, kf, float(row["temperature"]))
        check_close(row["f3db"], want, 2e-3,
                    f"{row['corner_id']} (rr={row['param_rr']} cc={row['param_cc']} "
                    f"lib={row['lib_models.lib_tt']} T={row['temperature']})")
    note(f"36 corners, max deviation within 0.2%")


@test("sim")
def sim_temperature_is_really_applied(work):
    """Distinguishes a real .temp sweep from a cosmetic one."""
    nl = stage_netlist("rc_corners.sp", work)
    run_ngrun(nl, "-j", "4", workdir=work)
    rows = read_csv(os.path.join(work, "rc_corners_results.csv"))
    sel = {r["temperature"]: float(r["f3db"]) for r in rows
           if r["param_rr"] == "1k" and r["param_cc"] == "1n"
           and r["lib_models.lib_tt"] == "tt"}
    check(set(sel) == {"-40", "27", "125"}, f"temperatures: {sorted(sel)}")
    check(sel["-40"] > sel["27"] > sel["125"],
          f"f3db must fall as temperature rises (tc1>0), got {sel}")
    ratio = sel["-40"] / sel["125"]
    want = (1 + 1e-3 * (125 - 27)) / (1 + 1e-3 * (-40 - 27))
    check_close(ratio, want, 2e-3, "cold/hot f3db ratio")


@test("sim")
def sim_temp_untouched_is_125C(work):
    """Bug-1 guard at simulation level, not just netlist text."""
    nl = stage_netlist("temp_fixed.sp", work)
    run_ngrun(nl, workdir=work)
    rows = read_csv(os.path.join(work, "temp_fixed_results.csv"))
    check(len(rows) == 1, f"expected 1 row, got {len(rows)}")
    want_125 = rc_f3db(1e3, 1e-9, 1.0, 125.0)
    want_25 = rc_f3db(1e3, 1e-9, 1.0, 25.0)
    check_close(rows[0]["f3db"], want_125, 2e-3,
                "netlist .temp 125 was not honoured")
    got = float(rows[0]["f3db"])
    check(abs(got - want_25) / want_25 > 0.05,
          "result is indistinguishable from the old hard-coded 25 C default")


@test("sim")
def sim_last_match_beats_decoy(work):
    """ngr_out must report the measured value, not the earlier decoy print."""
    nl = stage_netlist("ctrl.sp", work)
    run_ngrun(nl, workdir=work)
    rows = read_csv(os.path.join(work, "ctrl_results.csv"))
    check(len(rows) == 2, f"expected 2 rows, got {len(rows)}")
    for row in rows:
        rr = suffix_to_float(row["param_rr"])
        want = 1.0 / (TWOPI * rr * 1e-9)
        check(float(row["f3db"]) != 0.0,
              f"{row['corner_id']}: picked up the decoy 'f3db = 0'")
        check_close(row["f3db"], want, 2e-3, f"{row['corner_id']} f3db")
        check_close(row["gain_ratio"], want / 1000.0, 2e-3,
                    f"{row['corner_id']} derived gain_ratio")


@test("sim")
def sim_control_block_raw_file(work):
    """The injected 'set rawfile' redirects a bare write into the temp dir."""
    nl = stage_netlist("ctrl.sp", work)
    r = run_ngrun(nl, "-k", workdir=work)
    d = temp_dir_from_output(r.stdout)
    raws = [f for f in os.listdir(d) if f.endswith("_norm.raw")]
    check(len(raws) == 2, f"expected 2 raw files, got {raws}")
    for f in raws:
        check(os.path.getsize(os.path.join(d, f)) > 0, f"{f} is empty")


@test("sim")
def sim_tian_matches_closed_form(work):
    """Loop gain, UGF, PM and GM against the analytic 3-pole loop."""
    nl = stage_netlist("loop3.sp", work)
    run_ngrun(nl, workdir=work)
    rows = read_csv(os.path.join(work, "loop3_results.csv"))
    check(len(rows) == 1, f"expected 1 row, got {len(rows)}")
    ref = Loop3.reference()
    tol = {"a0_db": 1e-4, "ugf_freq": 2e-3, "pm": 2e-3,
           "gm_freq": 2e-3, "gm_db": 5e-3}
    for col, want in ref.items():
        check_close(rows[0][col], want, tol[col], f"probe 1 {col}")
        note(f"{col}: got {rows[0][col]}, analytic {want:.6g}")


@test("sim")
def sim_tian_break_point_invariance(work):
    """A single loop has the same loop gain wherever it is broken."""
    nl = stage_netlist("loop3.sp", work)
    run_ngrun(nl, workdir=work)
    row = read_csv(os.path.join(work, "loop3_results.csv"))[0]
    for col in ["a0_db", "ugf_freq", "pm", "gm_freq", "gm_db"]:
        a, b = float(row[col]), float(row[col + "_2"])
        check_close(b, a, 1e-6,
                    f"{col} differs between break points (amp.out vs amp:2)")
    note("dot-form and colon-form probes agree to 1e-6")


@test("sim")
def sim_tian_hierarchical_matches_flat(work):
    """Probing through a cloned subckt gives the same answer as the flat loop."""
    flat = stage_netlist("loop3.sp", work)
    hier = stage_netlist("loop3_hier.sp", work)
    run_ngrun(flat, workdir=work)
    run_ngrun(hier, workdir=work)
    a = read_csv(os.path.join(work, "loop3_results.csv"))[0]
    b = read_csv(os.path.join(work, "loop3_hier_results.csv"))[0]
    for col in ["a0_db", "ugf_freq", "pm", "gm_freq", "gm_db"]:
        check_close(b[col], float(a[col]), 1e-6,
                    f"{col} differs between flat and hierarchical probe")


@test("sim")
def sim_failed_corners_still_produce_rows(work):
    """A corner that cannot simulate is tagged, never dropped."""
    nl = stage_netlist("broken.sp", work)
    run_ngrun(nl, workdir=work)
    rows = read_csv(os.path.join(work, "broken_results.csv"))
    check(len(rows) == 3, f"expected 3 rows, got {len(rows)}")
    ids = [r["corner_id"] for r in rows]
    check(ids == ["c0001", "c0002", "c0003"], f"corner ids: {ids}")
    for r in rows:
        check(r["f3db"] not in ("", None),
              f"{r['corner_id']}: measurement column is empty")
        check(not re.match(r'^[-+0-9.]', r["f3db"]),
              f"{r['corner_id']}: expected an error tag, got {r['f3db']!r}")


@test("sim")
def sim_parallel_matches_serial(work):
    """-j must not change results or row count."""
    nl = stage_netlist("rc_corners.sp", work)
    run_ngrun(nl, "-o", "serial.csv", workdir=work)
    run_ngrun(nl, "-j", "4", "-o", "par.csv", workdir=work)
    a = {r["corner_id"]: r for r in read_csv(os.path.join(work, "serial.csv"))}
    b = {r["corner_id"]: r for r in read_csv(os.path.join(work, "par.csv"))}
    check(set(a) == set(b), "corner sets differ between serial and parallel")
    for cid in a:
        check(a[cid] == b[cid], f"{cid} differs between serial and parallel")


@test("sim")
def sim_typ_mode_runs_netlist_unchanged(work):
    """--typ produces a single row and does not substitute anything."""
    nl = stage_netlist("rc_corners.sp", work)
    run_ngrun(nl, "--typ", "-o", "typ.csv", workdir=work)
    rows = read_csv(os.path.join(work, "typ.csv"))
    check(len(rows) == 1, f"expected 1 row, got {len(rows)}")
    check(rows[0]["corner_id"] == "typ", f"corner_id={rows[0]['corner_id']!r}")
    # netlist as written: rr=1k, cc=1n, lib tt, .temp absent -> 27 C default
    want = rc_f3db(1e3, 1e-9, 1.0, 27.0)
    check_close(rows[0]["f3db"], want, 2e-3, "--typ result")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    global _VERBOSE
    ap = argparse.ArgumentParser(description="ngrun test suite")
    ap.add_argument("--layer", default="all",
                    choices=["unit", "gen", "sim", "all"])
    ap.add_argument("-k", "--pattern", default=None,
                    help="only run tests whose name contains this substring")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    _VERBOSE = args.verbose

    if not os.path.isfile(NGRUN):
        print(f"ngrun.py not found at {NGRUN}", file=sys.stderr)
        return 1

    selected = [t for t in _TESTS
                if (args.layer in ("all", t[0]))
                and (args.pattern is None or args.pattern in t[1])]

    ngspice = have_ngspice()
    passed = failed = skipped = 0
    failures = []

    print(f"ngrun test suite  ({len(selected)} selected)")
    print(f"  ngrun:   {NGRUN}")
    print(f"  ngspice: {'found' if ngspice else 'NOT FOUND - sim layer skipped'}")
    print()

    for layer, name, fn in selected:
        if layer == "sim" and not ngspice:
            print(f"  SKIP  {name}")
            skipped += 1
            continue
        work = tempfile.mkdtemp(prefix=f"ngruntest_{name}_")
        try:
            if fn.__code__.co_argcount:
                fn(work)
            else:
                fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            for line in str(e).splitlines():
                print(f"        {line}")
            failed += 1
            failures.append(name)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    print()
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
