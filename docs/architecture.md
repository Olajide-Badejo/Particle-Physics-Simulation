# Architecture

How the engine is put together, and why each boundary is where it is.

## Layers

```mermaid
flowchart TD
    CLI[main.cpp: command line and overrides] --> ENGINE[SimulationEngine]
    CONFIG[Config: JSON parsing and validation] --> ENGINE
    ENGINE --> PS[ParticleSystem: device resident state]
    ENGINE --> NB[N-body launchers]
    ENGINE --> SPH[SphSolver]
    ENGINE --> DIAG[DiagnosticsReducer]
    ENGINE --> SNAP[Snapshot read and write]
    NB --> KNAIVE[compute_forces_naive]
    NB --> KTILED[compute_forces_tiled]
    NB --> KFUSED[nbody_fused_step]
    NB --> KINT[Verlet integrator kernels]
    SPH --> KHASH[spatial_hash.cu]
    SPH --> KSPH[sph_forces.cu]
    PHYS[physics headers: host and device] --> KNAIVE
    PHYS --> KTILED
    PHYS --> KFUSED
    PHYS --> KSPH
    PHYS --> HOSTTESTS[host only unit tests]
```

The one boundary worth explaining is the `physics` headers. The force law, the
smoothing kernels, the equation of state and the spatial hash are written once,
marked to compile both as device code and as ordinary host C++, and included by
both the kernels and the tests. The kernels and the tests therefore cannot
drift apart, and the formula level tests run on a machine with no GPU.

The exception is deliberate: the double precision reference used by the force
gate shares no code with the kernels at all. Sharing there would mean the test
and the implementation agreeing because they are the same code, which proves
nothing.

## Library split

There are three library targets and the split is not incidental.

| target | contents | why |
|---|---|---|
| `physics_host` | configuration, snapshots, initial conditions, progress | no CUDA runtime dependency, so continuous integration can build and run a meaningful slice of the tests without a GPU |
| `physics_core` | all device code, respects the fast math switch | what the benchmarks and the demo link |
| `physics_core_strict` | the same sources, fast math forced off | what every validation test links |

Building the device library twice from identical sources is what makes the
physics gates trustworthy. The benchmark builds want `--use_fast_math`, which
lowers the reciprocal square root to a hardware approximation. A gate that a
compiler flag can relax is not a gate. FMA contraction is not disabled in the
strict library.

The default build targets `sm_120`. `-DCPS_FAT_BINARY=ON` builds `physics_core`
for `75;80;86;89;90;120`, which continuous integration and releases use;
`physics_core_strict` and the tests stay on the default.

## The N-body step

```mermaid
flowchart LR
    subgraph split[split path: naive and tiled]
        A1[half kick and drift] --> A2[force evaluation] --> A3[closing half kick]
    end
    subgraph fusedpath[fused path]
        B1[force, kick and drift in one launch] --> B2[swap position buffers]
    end
```

The split path runs true velocity Verlet across three launches. The fused path
runs the leapfrog form in one launch, which produces the same position sequence
to rounding provided the velocity is offset backwards by half a step once
before the loop begins.

Both integrator launches on the split path read the acceleration array, 24N bytes
per step. The fused launch reads it in neither. It still writes it, 12N bytes,
though the engine no longer reads that write, because the diagnostics and the
velocity conversions recompute the force first.

Positions are double buffered on the fused path. There is no grid wide barrier
inside a launch, so a block that has finished its force sum would otherwise
overwrite a position another block is still reading. The second buffer costs
12N bytes of memory and no bandwidth, because the write happens either way.

## The fluid step

```mermaid
flowchart LR
    H[1. hash: cell coordinate, bucket, exact cell key] --> S[2. sort: CUB radix sort of bucket and index]
    S --> B[3. bounds: first and last entry per bucket]
    B --> G[4. gather: permute particle data into cell order]
    G --> D[5. density: poly6 over 27 cells, then equation of state]
    D --> F[6. force and integrate: pressure, viscosity, gravity, walls]
```

Three points matter.

The update in step 6 is the same kick drift leapfrog as the fused N-body kernel,
with no initial offset applied, and its viscous term is first order.

The permutation is kept. After a step the particle arrays are in cell sorted
order, which is the order the next step wants, so the sort has an effect that
outlives the step that paid for it. Because that permutation must be a
bijection, one test checks the multiset of masses is unchanged across a step
and another checks total mass over two thousand steps.

The bucket scan compares an exact 64 bit cell key before accepting a candidate.
This is a correctness requirement, not an optimisation. Hashing is many to one,
so two of the twenty seven visited cells can land in the same bucket, and a
scan without the comparison would count those particles twice. Distance
filtering does not help, because those particles genuinely are within the
smoothing radius.

## Data layout

Structure of arrays: thirteen float arrays rather than one array of structures.

With an array of structures a warp reading the `x` component of thirty two
consecutive particles touches thirty two separate cache lines and uses a
fraction of each. With separate arrays the same read is thirty two consecutive
floats. This is also why the naive kernel is a fair baseline rather than a straw
man: it already starts from a coalesced layout, so the ladder measures tiling
rather than measuring a layout mistake.

| array group | count | purpose |
|---|---|---|
| x, y, z | 3 | positions |
| x_next, y_next, z_next | 3 | staged positions for the fused kernel |
| vx, vy, vz | 3 | velocities, synchronised or half step by convention |
| ax, ay, az | 3 | accelerations |
| mass | 1 | particle mass |

## Determinism

Three mechanisms, all load bearing for the bit identity guarantee.

Diagnostic reductions accumulate in double and sum block partials on the host in
a fixed order rather than with atomics, so two runs agree bit for bit.

The half step velocity correction the fused variant needs is applied inside the
reduction kernel rather than by shifting the velocity array and shifting it
back, because in single precision that round trip does not return the original
bits and would make a diagnostic perturb the run it measures.

Accelerations are not stored in snapshots. They are a deterministic function of
the positions, so on the N-body path recomputing them on load reproduces exactly
the array the original run held. The fluid solver keeps no force pass outside its
step, so the first diagnostic row after a fluid resume reads a zero acceleration;
the trajectory is unaffected.

## Snapshots

The format is `CPS2`, laid out in `include/cps/core/snapshot.hpp`: a 64 byte header, a
table with one 40 byte entry per array, and the payload. It records a 64 bit step, the
simulated time, a physics fingerprint and a payload checksum. `CPS1` from `v1.0.0` is still
read, with a warning, and never written.
