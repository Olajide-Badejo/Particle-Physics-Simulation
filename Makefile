# Reproducibility is a deliverable, so this file is the contract: a clean clone
# plus `make setup && make all` produces every number, figure and PDF in the
# repository with no manual steps.
#
# The build is configured once with fast math on. That switch reaches only the
# benchmark library; the validation library is compiled with fast math off no
# matter what, because a physics gate that can be relaxed by a compiler flag is
# not a gate. See CMakeLists.txt.

SHELL := /bin/bash
.DEFAULT_GOAL := help

BUILD_DIR ?= build
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc 2>/dev/null || echo 4)
CMAKE ?= cmake
CTEST ?= ctest
PYTHON ?= python3
RESULTS := experiments/results

# Everything the reports are built from.
FIGURES := report/figures/fig_nbody_time.pdf
TABLES := report/tables/nbody_ladder.tex
NUMBERS := report/numbers.tex

.PHONY: help setup setup-host-only build rebuild test test-host test-gpu validate \
        record-results bench bench-quick profile plots assets report report-debug \
        reports docs check-style check-warnings run-demo probe all clean distclean

help:
	@echo "cps: make targets"
	@echo
	@echo "  setup          configure the build (downloads GoogleTest and nlohmann_json)"
	@echo "  setup-host-only  configure without CUDA, for a machine with no toolkit"
	@echo "  build          compile everything"
	@echo "  test           run the whole test suite through ctest"
	@echo "  test-host      run only the tests that need no GPU"
	@echo "  validate       run the physics gates (results go to the build tree)"
	@echo "  record-results run the whole suite and record the gates in $(RESULTS)"
	@echo "  bench          run the full benchmark sweep, resuming where it left off"
	@echo "  profile        collect hardware counters with Nsight Compute"
	@echo "  plots          draw the figures from the recorded results"
	@echo "  assets         generate the LaTeX tables and macros"
	@echo "  report         build report/main.pdf"
	@echo "  report-debug   build report_debug/debug_report.pdf"
	@echo "  check-style    the dash gate over the whole repository"
	@echo "  all            everything above, in order"
	@echo "  clean          remove build outputs but keep recorded results"
	@echo "  distclean      also remove recorded results and snapshots"

# Both setups are the presets of the same name in CMakePresets.json, which hold
# the options. BUILD_DIR and BUILD_TYPE still override them from here.
setup:
	$(CMAKE) --preset dev -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

# No CUDA toolkit required. Builds the configuration layer, the snapshot
# format, the initial condition generators and the physics formulas, with the
# unit tests that cover them.
setup-host-only:
	$(CMAKE) --preset host-only -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

$(BUILD_DIR)/CMakeCache.txt:
	@$(MAKE) setup

build: $(BUILD_DIR)/CMakeCache.txt
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

rebuild: clean build

probe: build
	./$(BUILD_DIR)/bin/gpu_probe --output $(RESULTS)/device.json

# The full suite. On a machine with no device the GPU binaries exit with code
# 77 and ctest reports them as not run, rather than as passed. Recorded results
# go to the build tree, so this never touches experiments/results.
test: build
	cd $(BUILD_DIR) && $(CTEST) --output-on-failure

test-host: build
	cd $(BUILD_DIR) && $(CTEST) --output-on-failure -L host

test-gpu: build
	cd $(BUILD_DIR) && $(CTEST) --output-on-failure -L gpu

validate: build
	cd $(BUILD_DIR) && $(CTEST) --output-on-failure -L validation

# The only target that rewrites the committed gate values and validation
# series in experiments/results. The gate file is merged, so gates that did not
# run keep their previous rows.
record-results: build
	cd $(BUILD_DIR) && CPS_RECORD_GATES=1 $(CTEST) --output-on-failure

bench: build
	./$(BUILD_DIR)/bin/bench_all --tile-sweep

# A smoke test of the driver, on whatever tree is checked out. Its rows go to the
# temporary directory and are labelled as quick, so they can never pass for
# results.
bench-quick: build
	./$(BUILD_DIR)/bin/bench_all --quick --allow-dirty

profile: build
	bash scripts/profile_kernels.sh

# The colours are checked first, against the white page the figures are drawn on.
plots:
	$(PYTHON) scripts/validate_palette.py "$$($(PYTHON) scripts/plot_results.py --print-palette)" \
		--surface "#ffffff" --pairs all
	$(PYTHON) scripts/plot_results.py

assets:
	$(PYTHON) scripts/gen_report_assets.py

# Each report is built by its own makefile so it can be rebuilt on its own.
report: plots assets
	$(MAKE) -C report

report-debug: plots assets
	$(MAKE) -C report_debug

reports: report report-debug

docs: assets
	@echo "documentation is markdown and needs no build step"

check-style:
	$(PYTHON) scripts/check_no_dashes.py .

# Warnings are already errors in the default configuration, so a successful
# build is the gate. This target states that explicitly for anyone reading.
check-warnings:
	@$(MAKE) rebuild
	@echo "build completed with -Wall -Wextra -Werror and the nvcc equivalents"

run-demo: build
	./$(BUILD_DIR)/bin/cps run dam_break_50k --steps 2000

# The order matters: gates before benchmarks so a physics failure stops the
# line before half an hour of timing runs, and the reports last because they
# consume everything the earlier steps produced.
all: build check-style record-results probe bench profile plots assets reports
	@echo
	@echo "make all completed"
	@echo "  results  : $(RESULTS)"
	@echo "  figures  : report/figures"
	@echo "  reports  : report/main.pdf report_debug/debug_report.pdf"

clean:
	rm -rf $(BUILD_DIR)
	$(MAKE) -C report clean 2>/dev/null || true
	$(MAKE) -C report_debug clean 2>/dev/null || true

distclean: clean
	# Removes only untracked or ignored outputs. Tracked results and report
	# assets, and any local changes to them, are left alone.
	git clean -fdx -- $(RESULTS) report/figures report/tables snapshots
