# Changelog

All notable changes to this project are recorded here. The format follows Keep
a Changelog, and the project uses semantic versioning.

## [Unreleased]

### Added

- The `cps` command: `cps run <preset-or-path>`, `cps presets`, `cps info` and
  `cps convert <input> <output>`, which turns a v1 configuration into schema 2 and, with
  `--config <file>`, a CPS1 snapshot into CPS2. Help is on `cps --help` and `cps <subcommand> --help`.
- The files in `presets/` are compiled in, so an installed `cps` runs them by name with no files
  beside it.

### Changed

- **Breaking:** the program and its CMake target are now `cps`, and the project is `cps`. The name
  `cuda_physics_sim` stays for one major version as a symbolic link to it in the build and install
  trees. The form with flags and no subcommand still works under both names, and prints one line
  to stderr saying it goes away in the next major version. The exported `cps::cuda_physics_sim`
  target is now `cps::cps`.
- **Breaking:** `sph_gather_sorted` and `sph_forces_and_integrate` take the id arrays, and a
  `HostParticleState` given to `ParticleSystem::upload` or `write_snapshot` must have one id per
  particle. `HostParticleState::resize` numbers them for you.
- The presets print the two new `output` keys, regenerated through `--print-config`.
- **Breaking:** the public headers moved under `cps/`, from `include/core`, `include/physics`,
  `include/solvers` and `include/utils` to `include/cps/core` and so on. Include them as
  `#include "cps/core/config.hpp"`. The old paths have no forwarding headers.
- **Breaking:** the generated version header is now `cps/version.hpp`, not `cps_version.hpp`.
- Each CUDA compile now has one `-std`. The seed that gets the compiler probe through no longer
  stays in `CMAKE_CUDA_FLAGS` next to the standard CMake adds, which ends the `nvcc` warning about
  redefining the option on every file. The effective standard does not change. A second configure
  now also produces the same flags as the first, so it no longer rebuilds everything.
- `make setup` and `make setup-host-only` delegate to the `dev` and `host-only` presets.
- Continuous integration uses the presets, and the reports job installs from `requirements.txt`
  on Python 3.12 rather than the distribution's packages.
- **Breaking:** all of the engine's device work now runs on one non-blocking stream it owns,
  instead of the legacy default stream. `step()` no longer waits for anything. Code that reads
  the device arrays after stepping must call the new `SimulationEngine::synchronize()` first, and
  `SimulationEngine::stream()` is what to order other work with. `simulation_engine.hpp` says what
  is safe when.
- **Breaking:** the launchers in `nbody_solver.hpp` and `sph_passes.hpp`, the `DiagnosticsReducer`
  methods, `SphSolver` and `CudaTimer` take their stream with no default, so a caller has to name
  one. `tile_size` and `half_dt` lose their defaults where they sat before it.
- **Breaking:** `ParticleSystem::upload` and `download` take a stream and go through pinned
  staging, in place of their synchronous copies. Both return with that stream drained.
- `DeviceBuffer::zero` now waits for the bytes to be written, because a non-blocking stream is not
  ordered against the `cudaMemset` it issues on the default stream.
- `SimulationEngine` no longer branches on the solver. It holds a `Solver` from the new
  `cps/solvers/solver.hpp` and runs the loop, the diagnostics, snapshots and resume through it.
  The launches, their order and the trajectories do not change.
- **Breaking:** `SphSolver::step` takes no `dt`; the solver uses the one in its configuration.
- `DEFAULT_TILE_SIZE` moved to `cps/core/config.hpp`, which `nbody_solver.hpp` includes, because the
  configuration layer now names it. `offset_velocity_half_step` moved to `solver.hpp`, also included.
- The tile sweep in `bench_all` runs through the engine with `nbody.tile_size` instead of calling the
  kernels itself. Its protocol and its columns do not change.
- A fluid configuration the solver refuses now fails before the derived resolution and the time step
  limits are printed, because the solver prints them. The refusal names the limits itself.
- **Breaking:** configuration files are now schema 2, with `"schema": 2` and the sections `run`,
  `initial_condition`, `nbody`, `sph` and `output`. Migration is automatic: a file with no `schema`
  key, or with `"schema": 1`, still runs, with one warning line, and gives the same simulation.
  `--print-config` prints its schema 2 form. Any other `schema` value is refused.
- **Breaking:** `Config::to_json_string` and `--print-config` print schema 2, sections in a fixed
  order, and floats as the shortest decimal that reads back to the same bits, so `dt` prints as
  `0.001`. The `Config` struct itself does not change.
- `external_gravity` lives in the `sph` section, since only the fluid solver reads it.
- A misplaced key's error names the section it belongs in, from any section: a flat `dt` in a
  schema 2 file suggests `run.dt`. Type errors from the reader name the full path.
- **Breaking:** `configs/` is now `presets/`, in schema 2 form. The old files are kept as migration
  fixtures in `tests/data/configs_v1/`.

### Removed

- `DiagnosticsReducer`'s unused `result_` buffer.

### Added

- Particle ids. `ParticleSystem` and `HostParticleState` carry a `uint32` `id`, 0 to N-1 in the
  initial order, which the fluid solver permutes with the rest of the particle. No force, sort key
  or integrator reads it, and no trajectory changes.
- CPS2 snapshots store an `id` array through the array table, with no format bump. A file with no
  `id`, or a CPS1 file, loads with ids in file order; a bad `id` array is refused by name. Older
  builds skip the array.
- VTK output: XML PolyData frames (`.vtp`) and a `.pvd` series, from `include/cps/io/vtk.hpp`, with
  `output.vtk_interval`, `output.vtk_dir`, `--vtk-every` and `--vtk-dir`. Off by default. Frames and
  the series are replaced atomically, and a resume continues the series.
- `python/cps_io`: NumPy readers for snapshots, frames and series, and `check_vtp.py` to check them.
- The `Solver` interface, with `NBodySolver` and `SphSolver` implementing it and `make_solver(config)`
  choosing between them. `SimulationEngine` has a constructor that takes a `Solver`.
- The configuration key `nbody.tile_size`: 32, 64, 128 or 256, default 256. It is not part of the
  physics fingerprint, so a resume may change it and existing fingerprints do not move.
- `CMakePresets.json`, with configure, build and test presets `dev`, `release-fat`, `host-only`
  and `ci`.
- `cmake --install` and `find_package(cps)`, with the targets `cps::physics_host`,
  `cps::physics_core` and `cps::cuda_physics_sim`, also available under those names in the build
  tree. `tests/install_consumer` is a small project that uses them.
- A `Dockerfile` on the image continuous integration uses, with a `.dockerignore`, and
  `requirements.txt` with pinned versions for the plotting and report scripts.
- An installed nlohmann/json or GoogleTest at the pinned version or newer is used instead of
  fetching one. Fetching at the same pinned tags stays the fallback.
- `cps::Stream` and `cps::HostBuffer`, RAII and move-only, with `copy_to_device_async` and
  `copy_to_host_async`, in `cuda_helpers.hpp`.
- `-DCPS_DEBUG_SYNC=ON` drains the stream after every launch and copy, so an ordering bug can be
  bisected. Off by default, when it compiles to nothing.

## [1.1.0] 2026-11-03

A hardening release. It adds no algorithm. It fixes correctness defects in input handling,
snapshots and diagnostics, makes published numbers trace to a commit that exists, and corrects
claims `v1.0.0` got wrong. Fluid results and benchmark numbers are not comparable with `v1.0.0`:
see Changed.

### Corrections to v1.0.0

Values below that are labelled `v1.0.0` describe the past and are not regenerated.

- **Interior density.** `v1.0.0` reported 4.20 percent against a 5 percent limit. That gate read
  densities in cell sorted order against positions in lattice order, so it measured mismatched
  particles. With the index fixed, the `v1.0.0` mass gives 7.61 percent, which fails that limit.
  The mass did not match the lattice laid out, and started the dam break interior near 76 kPa
  against a hydrostatic 3.9 kPa, so the `v1.0.0` fluid started with a pressure pulse: maximum speed
  after 100 steps was 3.695 m/s, now 0.625 m/s. The explanation `v1.0.0` gave for the residual, a
  finite smoothing radius on a jittered lattice, was wrong. The mean error is now
  0.03 percent by calibration, with an RMS deviation of
  1.49 percent.
- **Fusion.** `v1.0.0` claimed a gain of 0.1 percent, from a difference of 0.12 percent measured in
  separate blocks, with a session spread of about 3 percent. It could not be told from noise.
  `v1.0.0` also described the saving as a round trip of the acceleration array, 24N bytes, when the
  fused kernel still wrote that array. A first correction said 12N, which is also incomplete: the
  split path reads the array in both of its integrator launches, 24N bytes of reads, and the fused
  kernel reads it in neither while still writing 12N that the engine no longer reads. Counting each
  array once per launch, and leaving out the force sum's repeated source reads, which both paths
  share, the split path touches 124N bytes per step and the fused path
  64N. `v1.0.0` called the resulting 0.019 percent an absolute ceiling on
  fusion, which was wrong, and said fusion halves the memory footprint of a step, which was false: it
  adds a second position buffer.
  Now, as paired ratios of fused over tiled at the default tile, median (IQR): 10,000 particles
  0.9445 (0.0075), 20,000
  0.9764 (0.0021), 50,000
  0.9875 (0.0062), 100,000
  0.9892 (0.0075), 150,000
  0.9909 (0.0013), against
  session drift of 0.17 percent. Fused is measurably faster at those sizes. It is
  not resolved at 200,000, 0.9914 (0.0276),
  where tiled trials are bimodal, near 43.1 and 44.4 ms, for a
  reason not yet understood. The source of the gain has not been measured either.
- **Tile width.** `v1.0.0` defaulted to 32. The default is now 256, 5.2
  percent faster at 100,000 particles (paired ratio 0.9483, IQR
  0.0028). `v1.0.0` tile sweep rows timed the force kernel alone for
  tiled, with particles that never moved, against whole steps for fused, so any comparison across
  variants in that table was void.
- **Provenance.** Both commit hashes recorded in the `v1.0.0` results do not exist in the published
  history, because the history was rewritten before release. Every row now carries a hash that
  resolves.
- **Energy gate.** The two body energy figure is dominated by fp32 rounding (`9.9e-13` truncation
  against 1.33e-5 measured), so it says nothing about the integrator's order. `v1.0.0`
  read its shape as evidence of symplecticity. Order is now tested in fp64.
- **Throughput.** `v1.0.0` quoted 17.47 TFLOP/s without stating its convention. Interactions per second
  is now the headline: 927 billion at 200,000, which is
  18.54 TFLOP/s at the conventional 20 flops per
  interaction and 16.72 at the 18.0 the counters
  report.
- **Markdown drift.** The Markdown documents quoted values that contradicted the CSVs: for example a
  bandwidth of 545.8 GB/s against a recorded 549.85 in `v1.0.0` `device.json`, and profiler rows that did
  not match `profile_summary.csv`. Measured values in Markdown now carry marks that
  `scripts/check_doc_numbers.py` checks.
- **Report contradictions.** The report said the tiling advantage grows with particle count, while its
  table showed 2.14x at 10,000 and 1.76x at 20,000 in `v1.0.0`. The debug report and README said eleven
  entries when there were fourteen. The debug report called tile 32 the slowest of the four when its table
  had 64 slower. It also said most gates had orders of magnitude of margin; several do not, and
  `docs/physics_validation.md` now classifies each one. "Run on every build" was also wrong: continuous
  integration has no GPU, so device gates run only on a machine with one.
- **Removed figures.** The 128 second full reproduction and the page counts were `v1.0.0` figures not
  re-measured, and are gone from the README.

### Fixed

- Non finite state stops a run, and a snapshot of it is refused.
- A resumed run appends to its diagnostics file instead of truncating it. A refused resume leaves the
  file alone.
- The fused path's kinetic energy no longer uses a stale acceleration, and time and completed step
  counts are right after a resume.
- `expected_neighbour_count` uses the pitch of the lattice laid out, so the startup line is true.
- The SPH density gate reads positions and densities in the same order.
- The SPH particle mass comes from a kernel sum over the lattice laid out.
- The full test suite no longer rewrites committed results. Skipped device tests show as Skipped.
- `make clean` and `make distclean` no longer delete tracked files.
- The commit stamp is taken at build time with a dirty flag, so no row can claim a clean tree it did
  not have.

### Changed

- The snapshot format is `CPS2`: a self describing array table, 64 bit step, double time, a physics
  fingerprint and a payload checksum. `CPS1` loads with a warning and is never written.
- A resume whose physics differs is refused. Older derived mass SPH snapshots need `--force-resume`.
- Misspelled or out of range config keys now abort, and zero softening is refused for N-body.
- CLI parse errors exit 2. Zero or negative `--dt`, `--particles` and `--steps` are refused.
- The fluid solver refuses a time step above its acoustic, viscous or body force limits. At the shipped
  `dt`, `--particles` above about 190,000 is refused, because the acoustic limit shrinks with the
  smoothing radius.
- The default `diagnostics_interval` is 100. It was 1, and each row costs an all pairs potential pass.
- Diagnostics append on resume. Solver time and wall time are reported separately.
- SPH behaviour changed because the mass changed, so SPH results are not comparable with `v1.0.0`.
- The benchmark protocol changed: ten interleaved trials per configuration, medians with dispersion,
  paired ratios and a session drift figure. Numbers are not directly comparable with `v1.0.0`.
- The default tile width is 256.
- The SPH density gates are mean error below 1 percent, RMS below 5 percent and maximum below 10 percent,
  committed before the code that measures them. The registered gates number 14.
- Documents no longer refer to an unpublished design document. `PROGRESS.md` is removed: this changelog and
  the reports cover it.

### Added

- `--force-resume`, `--allow-unstable-dt` and `--version`.
- `CPS_FAT_BINARY`, which builds `physics_core` for `75;80;86;89;90;120`. The default build stays
  `sm_120`.
- Double precision gates on the order of velocity Verlet.
- `make record-results`, the only target that updates committed gate values, and `make bench-quick`.
- `scripts/check_doc_numbers.py`, `scripts/check_provenance.py`, and `docs/generated_numbers.md`.
- `sm_throughput_pct` in the published counter table.

## [1.0.0] 2026-08-02

First complete release. The engine, the validation suite, the benchmark
pipeline and all three reports.

### Added

**Solvers**

- Direct all pairs gravitational N-body with Plummer softening, in three force
  kernel variants that isolate one technique each: naive global memory, shared
  memory tiled with a 32 particle tile, and a fused force plus integrator
  launch.
- Velocity Verlet integration, split across two launches for the unfused
  variants and in leapfrog form for the fused one.
- Weakly compressible smoothed particle hydrodynamics with spatial hash
  neighbour search, CUB radix sort, cell bounds, and a fused force and
  integration pass.
- Poly6, spiky and viscosity smoothing kernels with derived smoothing radius and
  particle mass.

**Correctness**

- Ten physics gates: two body energy drift, orbit circularity, momentum
  conservation, force against a double precision CPU reference, pairwise
  antisymmetry, Plummer radial density chi squared, two kernel normalisation
  gates, interior density, and neighbour search against a brute force sum.
- 59 host only unit tests, 17 device unit tests, 16 integration tests and 25
  validation tests.
- Snapshot and resume proved bit identical for all three N-body variants and for
  the fluid solver, including a run resumed twice.
- Two device libraries built from identical sources so that fast math can never
  reach a physics gate.

**Measurement**

- Benchmark driver with provenance on every row, resumable sweeps, a calibrated
  timing protocol and a projected total up front.
- Device probe reporting measured streaming bandwidth, which is the denominator
  of every bandwidth figure in the project.
- Nsight Compute profiling pass and a summary table.
- Figure and table generation from the recorded results, so no number in any
  document is typed by hand.

**Documentation**

- Main report and debug report, both compiled from the same generated numbers.
- Architecture, physics validation, design decisions and engineering log.
- A style gate that checks sources and compiled PDFs for en dashes and em
  dashes.

### Measured results

- Shared memory tiling is worth a factor of 1.93 at 200,000 particles, taking
  the step from 88.27 to 45.78 milliseconds at 17.47 TFLOP/s.
- Kernel fusion, expected beforehand to be worth 10 to 20 percent, is worth
  +0.1 percent at 200,000 particles. Device memory throughput is under one
  percent of peak for all three variants, so there was no memory pressure to
  relieve.
- Tiling's gain comes from relieving the first level cache and load store
  pipeline, not device memory bandwidth: global load sectors fall 63.8 times and
  warp issue efficiency rises from 49.7 to 74.2 percent.
- The fluid solver runs 50,000 particles at 0.338 milliseconds per step.
- All ten physics gates pass, most with several orders of magnitude of margin.

Several of these results and claims are corrected under 1.1.0 above.

### Known deviations from the original design

- GCC 15.2.0 is used instead of the planned GCC 16.1, because nvcc 13.3
  refuses any GCC newer than 15.
- Diagnostic reductions accumulate in double precision rather than as a float4
  per step, because a single precision reduction over 10,000 terms would have
  measured itself rather than the integrator.
- An `include/physics` directory was added beyond the planned tree to hold the
  formulas shared between device kernels and host tests.
- The tile width remains 32 as planned, though 256 measures 5.5 percent
  faster on this card. The measurement is reported rather than silently
  applied.
- The planned runtime estimates are far too long: the validation
  suite completes in about five seconds rather than 10 to 20 minutes, and the
  benchmark sweep in under two minutes rather than 15 to 30.
