# ngrun test suite
#
#   make test     run everything (sim layer is skipped if ngspice is missing)
#   make unit     helper-level tests only, no ngspice needed
#   make gen      netlist-generation tests only, no ngspice needed
#   make sim      full ngspice runs checked against closed-form answers
#   make verbose  run everything with per-test detail
#   make clean    remove strays

PYTHON ?= python3
RUNNER := run_tests.py

.PHONY: test unit gen sim verbose check-ngspice clean help

test:
	@$(PYTHON) $(RUNNER) --layer all

unit:
	@$(PYTHON) $(RUNNER) --layer unit

gen:
	@$(PYTHON) $(RUNNER) --layer gen

sim: check-ngspice
	@$(PYTHON) $(RUNNER) --layer sim

verbose:
	@$(PYTHON) $(RUNNER) --layer all -v

check-ngspice:
	@command -v ngspice >/dev/null 2>&1 || { \
	  echo "ngspice not found on PATH - the sim layer needs it."; exit 1; }

clean:
	@rm -rf __pycache__ ../__pycache__
	@rm -f *_results.csv netlists/*_results.csv *.raw netlists/*.raw
	@rm -rf /tmp/ngruntest_* 2>/dev/null || true
	@echo "cleaned"

help:
	@echo "targets: test unit gen sim verbose clean"
