# Engineering log

Dated entries, written as problems happen. Each one records the symptom I
actually saw, the root cause once I found it, the options I considered, the fix
and why I chose it, and how I verified it. This file is the raw material for
`report_debug/debug_report.pdf`.

---

## 2026-08-02: nvcc 13.3 refuses GCC 16 as a host compiler

**Theme:** environment.

**Symptom.** Compiling a trivial `.cu` file with `-ccbin g++-16`:

```
crt/host_config.h:137:2: error: #error -- unsupported GNU version! gcc versions
later than 15 are not supported!
```

**Root cause.** CUDA 13.3 declares support for GCC up to and including 15.
`host_config.h` enforces that with a preprocessor check, so the failure is a
deliberate refusal rather than a parse error in some header. This machine has
GCC 13, 14, 15 and 16 installed and defaults to 15, so the problem only appears
if the specified floor of GCC 16.1 is taken literally.

**Options.**

1. Pass `-allow-unsupported-compiler` and use GCC 16 anyway.
2. Drop to GCC 15.2.0, the newest host compiler the toolkit supports.
3. Install an older toolkit.

**Fix and why.** Option 2. The specification asks for GCC 16.1 as a floor but
also says to confirm that named versions are still current and to record
substitutions, which is exactly this case. What the floor was really buying is
the C++23 host standard, and GCC 15.2.0 provides that in full. Option 1
suppresses the toolkit's own compatibility check on a project whose entire
output is numbers that are supposed to be trustworthy; a miscompiled force
kernel that still runs is the worst possible failure mode here. Option 3 gives
up CUDA 13.3 and with it first class `sm_120` support on this card.

**Verification.** `nvcc -ccbin g++-15 -arch=sm_120 -std=c++23` compiles and
links a kernel plus a CUB radix sort, and the resulting binary runs correctly
on the device. Recorded as a substitution in `PROGRESS.md`.

---

## 2026-08-02: CUDA 13 removed clockRate and memoryClockRate from cudaDeviceProp

**Theme:** environment.

**Symptom.** The first version of the device probe did not compile:

```
error: class "cudaDeviceProp" has no member "clockRate"
error: class "cudaDeviceProp" has no member "memoryClockRate"
```

**Root cause.** Both fields were deprecated during the CUDA 12 series and
removed outright in CUDA 13. The information is still available, but only
through `cudaDeviceGetAttribute` with `cudaDevAttrClockRate` and
`cudaDevAttrMemoryClockRate`.

**Options.**

1. Hard code the RTX 5070 memory clock and bus width from the specification
   sheet.
2. Query both through the attribute interface.

**Fix and why.** Option 2, in `query_device` in `src/utils/cuda_helpers.cpp`.
This matters more than it looks: the memory clock and bus width are what the
theoretical peak bandwidth is computed from, and that number is the denominator
of every bandwidth utilisation figure in the report. Hard coding it would mean
the report quoted a number that had never been measured on the machine that
produced the results, which is the specific failure the no number without a run
rule exists to prevent.

**Verification.** The probe reports 14001 MHz effective on a 192 bit bus, which
gives 672.0 GB/s and agrees with the figure in the specification.

---

## 2026-08-02: CMake cannot identify the CUDA compiler at all

**Theme:** environment.

**Symptom.** `cmake -S . -B build` failed inside `project()`, before a single
line of project code was reached:

```
/usr/include/c++/15/type_traits(3303): error: __bool_constant is not a template
Error limit reached.
100 errors detected in the compilation of "CMakeCUDACompilerId.cu".
```

The confusing part was that nvcc had just compiled my own test kernel against
the same libstdc++ without complaint.

**Root cause.** The difference was the standard flag. My test used
`-std=c++20` explicitly. nvcc 13.3 defaults to C++17, and the libstdc++ 15
headers it is paired with here use `if consteval` inside `bits/c++config.h`,
which is C++23 syntax. Under the default standard the front end cannot parse
its own standard library, so nvcc fails on an empty translation unit. Reduced
to a two line file, the failure is unambiguous:

```
c++config.h(586): error: expected a "("
      if consteval { return true; } else { return false; }
```

CMake identifies the CUDA compiler by compiling a probe file with
`CMAKE_CUDA_FLAGS` and nothing else. `CMAKE_CUDA_STANDARD` has no effect there,
because it applies to targets and no targets exist yet.

**Options.**

1. Set `CMAKE_CUDA_STANDARD` earlier. Does not work; the probe does not consult
   it.
2. Pass `-DCMAKE_CUDA_FLAGS=-std=c++23` on every configure command line.
3. Seed `CMAKE_CUDA_FLAGS` in `CMakeLists.txt` before `project()`.

**Fix and why.** Option 3. Option 2 works but pushes a mandatory flag onto
every person who ever clones the repository, and reproducibility from a clean
clone with no manual steps is a deliverable. The guard is written so that a
value passed on the command line still wins.

**Verification.** Configuration completes and reports `The CUDA compiler
identification is NVIDIA 13.3.73 with host compiler GNU 15.2.0`.

---

## 2026-08-02: the build silently targeted the wrong GPU architecture

**Theme:** environment.

**Symptom.** After configuration started working, the status line read
`CUDA architectures: 75`. Turing, not Blackwell. Nothing failed: the code
compiled, and it would have run on the RTX 5070 through just in time
compilation of the embedded PTX, quietly and at the wrong performance.

**Root cause.** The default was being set with

```cmake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
```

placed after `project()`. Enabling the CUDA language inside `project()` assigns
a default architecture immediately, so by the time the guard runs the variable
is already defined and the branch never fires.

**Fix and why.** Move the guard above `project()` and make it a cache variable,
so an architecture passed on the command line is already in the cache and still
wins. Placing it before `project()` is the only position where the guard can
see the undefined state.

This one is worth recording precisely because nothing broke. Every timing
number in the benchmark ladder would have been measured against a binary built
for the wrong architecture, and the only symptom would have been results that
were somewhat worse than expected, which is indistinguishable from the code
simply being slow.

**Verification.** The status line now reads `CUDA architectures: 120`, and
`cuobjdump --list-elf` confirms `sm_120` SASS in the built objects rather than
PTX only. The object file is the real check; the status line reports intent.

---

## 2026-08-02: a unit test failed because the test was wrong

**Theme:** physics correctness.

**Symptom.** `SphKernels.KernelsAreContinuousAtTheSupportBoundary` failed on its
first run while every other kernel test passed.

**Root cause.** The test asserted each kernel evaluated just inside its support
was within `1e-3` of zero in absolute terms. These kernels carry a `1/h^6`
normalisation: at `h = 0.1` the spiky gradient peaks near `1.4e5` and the
viscosity Laplacian near `1.4e6`. An absolute tolerance of `1e-3` against
quantities of that magnitude says nothing at all. The implementation was
correct.

**Fix and why.** Compare each kernel at the boundary against its own peak, which
is what continuity means for a quantity with a scale, and separately assert the
kernels are exactly zero outside the support so the transition is genuinely
continuous rather than merely small.

**Lesson, now a house rule.** A tolerance without a scale is not a tolerance.
Every gate in the project is expressed relative to the magnitude of the
quantity, the machine epsilon, or a bound derived from the arithmetic.

---

## 2026-08-02: fusion delivered a fortieth of what I expected

**Theme:** performance.

**Symptom.** Not a failure. Kernel fusion, expected to be worth 10 to 20
percent, measured +0.1 percent at 200,000 particles and slightly negative at
some intermediate sizes.

**Root cause.** Arithmetic intensity. The all pairs sum is O(N^2) arithmetic
over O(N) data, so it is not waiting on memory. The 24N bytes fusion removes
take 0.0088 ms at the measured 545.8 GB/s, against a 45.78 ms step: 0.019
percent as an absolute ceiling. Nsight Compute confirms device memory throughput
at 0.50 percent of peak for the naive kernel and 0.84 percent for the tiled one.

**Fix and why.** No code change. The fused kernel is kept because it wins at
small particle counts and halves the memory footprint of a step. What changed is
the practice: size an optimisation against the resource it claims to relieve,
using measured numbers, before writing it. That calculation takes two minutes
and needs only figures Phase 0 already produced.

**Verification.** Recorded in `experiments/results/benchmarks.csv` and
`profile_summary.csv`, and analysed in the main report and the debug report.

---

## 2026-08-02: tiling is not a bandwidth optimisation on this problem

**Theme:** performance.

**Symptom.** Tiling delivers a real factor of 1.93, but device memory throughput
is negligible both before and after, so the textbook explanation of why it helps
cannot be right here.

**Root cause.** The bottleneck is the first level cache and the load store
pipeline, not the memory bus. Counters at 100,000 particles: global load sectors
fall from 1.250e9 to 1.959e7, a factor of 63.8; L1 throughput falls from 79.58
percent of peak, close to saturation, to 39.72 percent; warp issue efficiency
rises from 49.74 to 74.15 percent. The naive kernel spends half its time with
nothing to issue because its warps wait on the load store pipe. Arithmetic
instruction counts are equal across all three variants, which is the control.

**Why it mattered.** Attacking device memory traffic, by compressing state or
prefetching, would have achieved nothing. Profile first, then optimise the
resource the profile names.

---

## 2026-08-02: the benchmark recorded a random seed as a tile width

**Theme:** performance tooling.

**Symptom.** The tile width figure had a bar at a tile width of `20260802`,
which is the random seed.

**Root cause.** The tile sweep opened its output file twice: a truncating stream
for the header, and the row appending helper opening the same path
independently. The header stream stayed buffered for the duration of the sweep
and flushed at offset zero when destroyed, overwriting the start of the file.
The surviving rows had shifted column boundaries, so fields were read from the
wrong columns.

**Fix and why.** Write the header through the same append path as the rows,
removing the file first, so only one handle exists at a time.

**Verification.** The regenerated CSV contains exactly the four intended tile
widths. Note that nothing failed: the benchmark ran, the CSV parsed and the plot
rendered. It was caught by looking at the figure and asking what a tile width of
twenty million could mean.

---

## 2026-08-02: absolute timings move with the thermal state of the card

**Theme:** performance.

**Symptom.** A tile sweep run immediately after the full benchmark sweep gave
timings about 15 percent higher than the same sweep from cold, with no code
change.

**Root cause.** Sustained clocks fall as the card heats. Ordinary behaviour, but
large enough to swamp several of the effects being measured, including the
entire fusion result.

**Fix and why.** All comparisons are made within a single session, and each
point keeps the fastest of three trials rather than the mean, because everything
that perturbs a measurement makes it slower and never faster.

**Verification.** Across the hot and cold runs the tile width ordering was
identical and the percentage difference between the widest and narrowest tile
reproduced to within a tenth of a percentage point, while absolute milliseconds
differed by 15 percent. The conclusions rest on relative differences and survive
the variation.

---

## 2026-08-02: three LaTeX traps in one afternoon

**Theme:** environment and tooling.

**Symptom one.** The report failed with `Missing \begin{document}` pointing at a
generated macro that looked fine.

**Root cause.** A LaTeX control sequence is letters only. In
`\Poly6NormalisationError` the name ends at the digit, so the rest became body
text in the preamble. **Fix:** rename the three offending macros to spell their
digits, and make the generator reject any non alphabetic macro name so this
fails at generation with a message that names the problem.

**Symptom two.** `pdfTeX error (font expansion): auto expansion is only possible
with scalable fonts`, fatal.

**Root cause.** Under T1 the default Computer Modern resolves to bitmap fonts
and microtype cannot expand them. **Fix:** load Latin Modern before microtype.

**Symptom three.** The dash gate failed on the compiled PDF with seven en dashes
in the bibliography, after the source bibliography had been cleaned and reported
clean.

**Root cause.** The `plain` bibliography style dashifies page ranges itself when
writing the intermediate file, so the source was never the problem. A later
instance was found inside `\texttt{cuobjdump --list-elf}`: `\texttt` does not
suppress the ligature, only a verbatim region does. **Fix:** write page ranges
with the word "to", and use `\verb` for command lines containing two hyphens.

**Why this justifies the gate's design.** The rule forbids a character in the
finished document, so the gate has to read the finished document. A source only
check would have reported clean while every reference in the printed
bibliography broke the rule. This is why the gate runs last in the report build
and extracts text from the compiled PDF.

---

## 2026-08-02: the declared CMake minimum was a version that cannot build the project

**Theme:** environment.

**Symptom.** The first continuous integration run failed at the configure step,
in the CUDA job only. The host job passed. Configuration itself succeeded and
then the generate step failed, four times over:

```
CMake Error in CMakeLists.txt:
  Target "physics_core" requires the language dialect "CUDA23" .  But the
  current compiler "NVIDIA" does not support this, or CMake does not know the
  flags to enable it.
```

**Root cause.** Knowing that nvcc supports the C++23 dialect is a property of
the running CMake, not of the compiler. The reference machine runs CMake 4.4,
which lists `cuda_std_23` among the CUDA compile features. The runner used the
CMake 3.28 that Ubuntu 24.04 ships, which does not, so it refused a standard the
compiler in front of it was perfectly capable of. Since
`cmake_minimum_required` declared 3.28, the project was claiming to build with a
version that cannot build it. The error message blames the compiler, and the
compiler is innocent.

That the host only job passed is the tell: it never sets a CUDA standard, so it
never asked the question.

**Options.**

1. Raise `cmake_minimum_required` and install a newer CMake on the runner.
2. Pin the device standard to C++20 for everyone.
3. Ask CMake what it can actually offer and use that.

**Fix and why.** Option 3. The project now checks whether `cuda_std_23` appears
in `CMAKE_CUDA_COMPILE_FEATURES` and falls back to C++20 when it does not,
reporting the choice in the configuration summary. Option 1 forces a manual
toolchain upgrade on anyone who only wants to compile the code. Option 2 gives
up C++23 on the reference machine to accommodate an older tool. Nothing in the
device code needs a C++23 feature, so the fallback costs nothing, and the host
standard stays at C++23 in every configuration.

The seed flag that gets the compiler identification probe through was lowered
from C++23 to C++20 for the same reason. It only has to be new enough for
libstdc++ to parse itself, and demanding C++23 there made the same unwarranted
assumption a second time.

**Verification.** Both paths were built and tested on the reference machine
before the fix was pushed. With the device standard forced to C++20 the whole
project compiles with zero warnings and all four test suites pass, physics gates
included, so the fallback is not a degraded build in any way that matters.

**Lesson.** A declared minimum version is a claim, and an untested claim is
usually wrong. This one had never been exercised because every local build used
the newest toolchain on the machine. Continuous integration on a deliberately
older environment is what turned the assumption into a measurement.
