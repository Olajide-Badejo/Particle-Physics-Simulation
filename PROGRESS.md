# Progress

One entry per phase. A phase is not complete until its checks have been run
and their output recorded here.

## Phase 0: toolchain, GPU probe, dash lint

Status: complete.

Environment verified on the target machine (Windows 11 Pro host, build and run
inside WSL2 Ubuntu 26.04):

| Component | Specified floor | Found | Note |
|---|---|---|---|
| CUDA Toolkit | 13.3 | 13.3.73 | as specified |
| Host compiler | GCC 16.1 | GCC 15.2.0 | substitution, see below |
| C++ standard | C++23 host | C++23 host and device | C++26 is not accepted by nvcc 13.3 |
| CMake | 4.4 | 4.4.0 | as specified |
| GoogleTest | FetchContent | v1.17.0 | pinned tag |
| nlohmann_json | FetchContent | v3.12.0 | pinned tag |
| CUB | from the toolkit | CUDA 13 CCCL layout | headers moved to `include/cccl` |
| Python | 3 with matplotlib | 3.14.4, matplotlib 3.10.7, tqdm 4.67.3 | numpy and scipy also present |
| TeX | latexmk in WSL | latexmk 4.87, TeX Live 2025 | `pdftotext` 26.01 available for the dash gate |

Substitutions, as the specification requires them to be recorded:

- **GCC 15.2.0 instead of GCC 16.1.** nvcc 13.3 refuses any GCC newer than 15
  outright: `host_config.h` raises `unsupported GNU version! gcc versions
  later than 15 are not supported`. GCC 16 is installed on this machine, so
  this was verified by compiling against it rather than assumed. The
  `-allow-unsupported-compiler` escape hatch exists but disabling the toolkit's
  own compatibility check on a project whose entire point is trustworthy
  numbers is the wrong trade. GCC 15.2.0 provides the C++23 host standard the
  specification actually asks for.

Measured device properties, from `gpu_probe` rather than from a specification
sheet:

```
device 0: NVIDIA GeForce RTX 5070 (compute capability 12.0)
  multiprocessors      : 48
  core clock           : 2625 MHz
  memory clock         : 14001 MHz effective
  memory bus           : 192 bits
  peak bandwidth       : 672.0 GB/s
  global memory        : 11.94 GiB
  shared memory/block  : 49152 bytes
  max threads/block    : 1024
  cuda runtime/driver  : 13030 / 13030
  measured copy rate   : 545.8 GB/s (81.2% of peak)
```

The theoretical peak of 672.0 GB/s matches the figure in the specification.
The measured streaming copy rate of 545.8 GB/s is 81.2 percent of it, and that
measured number is the denominator used for every bandwidth utilisation figure
in the report. Quoting utilisation against the theoretical peak instead would
flatter every kernel by the same factor without saying anything about the
kernel.

`ncu` counter access was verified and works under WSL2 on this machine, so the
profiling based analysis is available and the timing only fallback described in
the specification is not needed.

Checks run: `gpu_probe` (output above), `scripts/check_no_dashes.py` (clean),
Release build of the core library and both executables with `-Wall -Wextra
-Werror` and the nvcc equivalents (zero warnings).

Four environment defects were found and fixed during this phase; all four are
written up in `docs/ENGINEERING_LOG.md`. One of them, a build silently
targeting the wrong GPU architecture, would have invalidated every performance
number in the project while leaving no symptom other than results that were
mildly disappointing.

## Phase 1: SoA core, naive kernel, unit tests

Status: complete.

Configuration parsing with validation, device resident structure of arrays
particle state, the binary snapshot format, the three initial condition
generators, and the naive force kernel.

Checks run: 59 host only unit tests, covering configuration parsing and its
error paths, snapshot round tripping and corruption handling, the smoothing
kernels by quadrature, the spatial hash, and the initial condition generators.
All pass. One failed on its first run because the test was wrong rather than
the code, which is recorded in the engineering log and produced a house rule: a
tolerance without a scale is not a tolerance.

## Phase 2: tiled and fused kernels, Verlet, snapshot and resume

Status: complete.

Checks run: 17 device unit tests and 16 integration tests, all passing. The
important ones are the parity tests, which assert all three force variants
agree with a double precision host reference to better than one part in ten
thousand, and the bit identity tests, which assert that stopping and resuming
reproduces a straight through run exactly, for all three N-body variants, for
the fluid solver, and for a run resumed twice.

## Phase 3 and Phase 4: physics gates and the fluid solver

Status: complete.

Checks run: 25 validation tests, all passing, against the library compiled with
fast math disabled. Measured values:

| gate | measured | limit |
|---|---|---|
| Two body energy drift over 10,000 steps | 1.33e-05 | < 1e-03 |
| Two body orbit departure from circular | 9.35e-06 | < 1e-03 |
| Plummer momentum drift over 1,000 steps | 1.23e-08 | < 1.19e-02 |
| Force against double precision CPU reference | 5.77e-07 | < 1e-05 |
| Pairwise force antisymmetry residual | 1.53e-07 | < 1e-06 |
| Plummer radial density chi squared p value | 0.4416 | > 0.05 |
| SPH poly6 kernel normalisation error | 6.48e-08 | < 1e-03 |
| SPH spiky kernel normalisation error | 4.39e-08 | < 1e-03 |
| SPH interior density relative error | 0.0420 | < 0.05 |
| SPH neighbour search against brute force | 3.20e-07 | < 1e-04 |

**Correction to the estimate in Section 2 of the specification.** The full
validation suite was estimated at 10 to 20 minutes on the grounds that the
10,000 step orbit would dominate. It completes in about 5 seconds. The estimate
assumed that orbit would be expensive; with two particles it is trivially
cheap, and the cost is dominated instead by the 10,000 particle Plummer runs.

## Phase 5: benchmark sweep, charts and profiling

Status: complete.

Checks run: the full sweep across three variants and six particle counts, the
fluid sweep across five counts, a tile width sweep, and a Nsight Compute
profiling pass.

Headline results:

- Shared memory tiling is worth a factor of 1.93 at 200,000 particles, taking
  the step from 88.27 to 45.78 ms at 17.47 TFLOP/s.
- Kernel fusion, labelled an expectation of 10 to 20 percent in the
  specification, measures +0.1 percent at 200,000 particles. The correction and
  its explanation are in the debug report: the problem is arithmetic bound, so
  removing memory traffic cannot help. The absolute ceiling on what fusion
  could have delivered is 0.019 percent.
- Tiling's gain comes from relieving the first level cache and load store
  pipeline rather than device memory bandwidth. Global load sectors fall by a
  factor of 63.8, L1 throughput falls from a near saturated 79.6 percent of
  peak to 39.7 percent, and warp issue efficiency rises from 49.7 to 74.2
  percent. Device memory throughput never exceeds one percent of peak in any
  variant.
- The fluid solver runs 50,000 particles at 0.338 ms per step, roughly fifty
  times inside a sixty frame per second budget.

**Correction to the estimate in Section 2.** The benchmark sweep was estimated
at 15 to 30 minutes. It completes in under 2 minutes.

Two defects were found in this phase: a file corruption that recorded a random
seed in a tile width column, and two chart layout faults. All are logged.

## Phase 6: documentation from real numbers

Status: complete.

README, `docs/architecture.md`, `docs/physics_validation.md`,
`docs/DESIGN_DECISIONS.md`, `CONTRIBUTING.md` and `CHANGELOG.md`, all written
from the recorded results rather than from memory.

## Phase 7: reports

Status: complete.

Both PDFs are built by `make all` from the same generated tables and macros, so
no number in either is typed by hand.

| document | pages |
|---|---|
| `report/main.pdf` | 28 |
| `report_debug/debug_report.pdf` | 14 |

Three tooling defects surfaced while compiling them, all logged: LaTeX macro
names cannot contain digits, microtype cannot expand the default bitmap fonts,
and BibTeX reintroduces en dashes into page ranges after the source has been
cleaned. The last of these is the justification for the dash gate reading the
compiled PDF rather than only the sources.

## Phase 8: final quality assurance

Status: complete.

`make clean && make all` reproduces everything from a clean tree with no manual
steps: build with warnings as errors, the style gate, the full test suite, the
device probe, the benchmark sweep, the profiling pass, every figure and table,
and all three reports.

Final state:

- 4 of 4 ctest suites pass, 117 tests in total across the four levels.
- Zero compiler warnings, with `-Wall -Wextra -Werror` and the nvcc
  equivalents.
- The dash gate is clean across the repository including all compiled PDFs.
- All 10 physics gates pass.

**Measured wall clock for `make clean && make all` on the reference machine:
128 seconds**, timed rather than estimated, including re-fetching GoogleTest
and nlohmann_json, compiling both device libraries, all four test suites, the
device probe, the full benchmark sweep, the profiling pass, every figure and
table, and all three reports.

That is against the specification's estimate of 45 minutes to 1.5 hours. The
estimate inherited the two component estimates that Phases 3 and 5 already
corrected, and the same reasoning applies: the 10,000 step orbit that was
expected to dominate involves two particles.
