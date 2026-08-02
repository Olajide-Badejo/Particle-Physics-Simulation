# Changelog

All notable changes to this project are recorded here. The format follows Keep
a Changelog, and the project uses semantic versioning.

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

### Known deviations from the original specification

- GCC 15.2.0 is used instead of the specified GCC 16.1, because nvcc 13.3
  refuses any GCC newer than 15.
- Diagnostic reductions accumulate in double precision rather than as a float4
  per step, because a single precision reduction over 10,000 terms would have
  measured itself rather than the integrator.
- An `include/physics` directory was added beyond the specified tree to hold the
  formulas shared between device kernels and host tests.
- The tile width remains 32 as specified, though 256 measures 5.5 percent
  faster on this card. The measurement is reported rather than silently
  applied.
- The runtime estimates in the specification are far too long: the validation
  suite completes in about five seconds rather than 10 to 20 minutes, and the
  benchmark sweep in under two minutes rather than 15 to 30.
