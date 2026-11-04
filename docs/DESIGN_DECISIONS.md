# Design decisions

What was chosen, what was rejected, and why. Decisions that were later
contradicted by measurement are marked, with the correction.

## Solver and algorithm

### Direct all pairs N-body rather than Barnes Hut

At 10,000 to 200,000 particles the O(N^2) sum fits this card comfortably, is
exactly verifiable against a CPU reference, and keeps the memory hierarchy
story clean. A tree code's irregular traversal wins far above this range and is
a genuinely different GPU problem. Kept on the roadmap rather than in scope.

### Structure of arrays rather than array of structures

So that warp threads read consecutive addresses. This also makes the naive
kernel a fair baseline: it starts from a coalesced layout, so the ladder
measures tiling rather than measuring a layout mistake.

### Velocity Verlet

Rejected explicit Euler, which is not symplectic and whose energy error grows
without bound. Rejected fourth order Runge Kutta, which needs four force
evaluations per step when the force is the entire cost. Velocity Verlet is
second order, symplectic, and needs one force evaluation per step.

### Leapfrog form for the fused kernel

The fused kernel cannot run the two launch velocity Verlet, so it runs the
leapfrog form, which produces the same position sequence to rounding provided
the velocity is offset backwards by half a step once before the loop. The cost
is that the stored velocity is no longer synchronised with the positions, which
propagates into the snapshot format and the diagnostics.

### The fluid update is a kick-drift leapfrog with no initial offset

The fluid solver updates `v <- v + a dt`, then `x <- x + v dt`, which is the same
kick-drift leapfrog as the fused kernel. The initial offset is not applied: the
acceleration is still zero when the engine would apply it, so the initial
velocity is taken as `v_{-1/2}` as it stands. The viscous term reads the half
step velocity, which makes it first order. The half step flag stays set, because
it tells the diagnostics to read `v + a dt/2`, a better estimate of the
synchronised velocity than `v` alone. The fluid acceleration is not recomputed on
load, so the first diagnostic row after a resume uses a zero acceleration. The
trajectory is unaffected and stays bit identical.

### Softening at 0.01 is not a collisionless claim

In a 10,000 particle Plummer sphere with scale radius 1, `eps = 0.01` is about 7.5 times below the mean
separation, `(4 pi / 3 / N)^(1/3)`. Close encounters still matter, so the run is mildly collisional.
Softening is standard for collisionless systems, and here its job is to keep the integrator stable. The
default stays, and this is a statement about the shipped runs, not about the method.

### Fluid defaults are documented, not tuned

`sph.viscosity` is the dynamic viscosity in Pa s, and the default of 0.2 is a kinematic viscosity of
2e-4 m^2/s, about 200 times water. `sph.boundary_damping` is the fraction of normal velocity kept at a
wall, so 0.4 keeps 16 percent of the normal kinetic energy per impact. Changing either changes every
fluid number, so they wait for a later release.

### Double buffered positions on the fused path

Required for correctness, not performance. There is no grid wide barrier inside
a launch, so updating in place would let a block that has finished its force
sum overwrite a position another block is still reading. Costs 12N bytes and no
bandwidth.

### A dense grid rather than a spatial hash or a k-d tree

Every fluid domain here is bounded and has walls, so a cell can be named by where it sits in
the grid: `ix + nx * (iy + ny * iz)`, with x varying fastest. That index does three things a hash
could not. It is unique, so the neighbour search has no collisions to survive and no two visited
cells can count a particle twice. It needs no modulus and no per candidate check. And sorting by
it puts neighbouring cells close together in memory, with the cells of an x row contiguous, so the
search reads nine ranges instead of visiting twenty seven cells. The radix sort still stops at the
bits the index can use. Whether all that made the step faster was a prediction, tested against
the rule that the grid stays only if it is not slower at 10,000, 50,000 or 100,000 particles.

A grid costs one thing: it is as large as the domain, whatever the particle count. So the solver
derives its size from the domain and refuses, naming the dimensions and the memory, more than
`2^31` cells. Only bounded domains can use it. A periodic axis would be a wrap of the index, which
`axis_cell` leaves a place for and nothing implements. A k-d tree is a poor fit for a full GPU
rebuild every step.

**Corrected.** Until this change the search was a spatial hash, chosen because a dense grid
needs storage proportional to the bounding volume and a dam break leaves most of that empty. That
argument was wrong: the hash table was larger than the grid it avoided. The table had twice as many
slots as particles, while the shipped dam break's domain comes to about 0.42 cells per particle. At
50,000 particles, by derivation from the domain and the smoothing radius, that is
41 x 26 x 21 = 22,386 cells against a table of 100,003 slots, and the table was two arrays, with
an eight byte key per particle on top. A hash table keeps its collisions rare by being mostly empty,
so counting the grid's empty cells and not the table's was never a fair comparison.

### No cell key in the neighbour scan

The scan used to compare an exact 64 bit cell key for every candidate. That was correct and
necessary, because hashing is many to one and two of the twenty seven cells could land in one
bucket, and it existed only because of the hash. With a unique index there is no key to compare.
The property it protected is now structural, and the brute force gate that proved it still runs
unchanged.

### A prefix sum cell table, filled by binary search

The cell table holds, for each cell, how many particles are in the cells before it. A row of cells
is then two reads, and an empty cell is an entry equal to the next one, so no pass has to test for
one and the table needs no clearing. The two arrays of the hash, with an empty marker, would have
taken six reads per row, or a test per cell.

Each entry is filled by its own binary search of the sorted keys. Each entry then has one writer,
with no atomics, and every thread does about the same work. Writing the gaps from the particles'
side is less work in total, but one thread writes each run of empty cells, and in a sparsely
filled domain one run can be most of the table. A histogram and a scan would add launches.

### A cell slightly wider than the support

With a cell of exactly the smoothing radius, twenty seven cells cover the support only in exact
arithmetic. Two particles just inside the support can come out two cells apart, because the
coordinate is rounded, and the unit tests find such pairs. So a cell is wider than the support by
`(widest + 2) * 2^-21` of it, where `widest` is the grid's longest axis in supports, which is more
than all the rounding involved can add up to. That is about 2e-5 of a cell at 41 cells wide. The hash
had this exposure too, without the margin. The kernels vanish at the edge of the support, so such
a loss would never have shown in a result, which is why it has to be ruled out and not measured.

### A reserved slot for non finite positions

A NaN or Inf position has no cell. It gets the sort key `num_cells`, one past the last cell, and sets
the device flag. All finite particles therefore sort ahead of it, the table's last entry closes the
last real cell's range, and no search ever scans the slot, so it cannot contaminate a neighbour. A
non finite particle searches nothing itself: that test is on the bits, so it does not depend on what
a fast math build may assume about NaN. The sort's bit range counts the slot, because a grid
of exactly `2^k` cells needs one more bit for it than for any real cell.

### Three smoothing kernels rather than one

poly6 for density, because it is a polynomial in r squared and needs no square
root. spiky for the pressure gradient, because its gradient does not vanish at
contact and poly6's does, which means no restoring force exactly where the fluid
needs one and particles clump. A viscosity kernel with a positive Laplacian
throughout its support, so the viscous term always damps and never injects
energy.

### Derived smoothing radius and particle mass

The smoothing radius defaults to two ideal spacings, computed from the fluid volume and
particle count. The mass defaults to a kernel sum over the lattice actually laid out, so that
the block's interior sits at rest density. On that lattice, whose pitch is tighter than the ideal
spacing, the support holds about 36 neighbours at 50,000 particles, which the engine prints at
start up. Fixing either by hand while the particle count moves is the fastest way to a resolution
the kernel cannot support. Explicit values in the configuration still override.

**Corrected.** `v1.0.0` used a mass of `rest_density * volume / count`, which assumes a spacing the
lattice does not have. That started the dam break interior near 76 kPa against a hydrostatic
3.9 kPa, and the gate that should have caught it read the wrong particles. The record is in
[physics_validation.md](physics_validation.md).

### Jittered initial lattice

A perfectly regular lattice is a stationary point of the pressure force, so the
block sits still until rounding noise breaks the symmetry, which looks exactly
like a bug.

### Plummer velocities from the exact distribution function

Rejected a convenient Maxwellian, which would leave the sphere out of virial
equilibrium so that it visibly breathes on the first crossing time, again
looking exactly like an integrator bug. A test holds the virial ratio at t = 0
between 0.85 and 1.15.

## Correctness apparatus

### Two device libraries

`physics_core` respects the fast math switch; `physics_core_strict` is compiled
from identical sources with fast math forced off, and every validation test
links it. A gate that a compiler flag can relax is not a gate.

### Double precision diagnostic accumulators

The state stays single precision, which is the point of the exercise, but a
float32 tree reduction over 10,000 terms loses three to four decimal digits and
the energy gate resolves one part in a thousand. Reducing in single precision
would have measured the reduction rather than the integrator.

The first design described a float4 per step instead. The traffic difference is one
small array and nothing per particle.

### Deterministic reductions

Block partials are summed on the host in a fixed order rather than accumulated
with atomics, whose ordering varies between launches. Atomics would have been
shorter and would have made two runs of the same configuration disagree in the
last bits, which the bit identity test forbids.

### The half step correction lives in the reduction kernel

Shifting the velocity array forward and back would not return the original bits
in single precision, so a run that took a diagnostic would diverge from one that
did not.

On the fused path the engine recomputes the force at the current positions before
each reduction, because the kernel leaves `a_n` in the array and the synchronised
velocity needs `a_{n+1}`. The fused kernel never reads that array, so the extra
pass costs one all pairs sum per diagnostic and cannot affect the trajectory.

### A resumed run continues its diagnostics file

The file is opened after the resume checks pass, so a refused resume leaves it
alone. Rows at or past the resume step are dropped before appending, because the
resumed run writes them again, which also covers a run that died with its
diagnostics ahead of its last snapshot. A header that does not match is an error
rather than a file holding two formats.

### Solver time and wall time are reported separately

Solver time comes from device events around the stretches of `step()` calls
between diagnostics and snapshots. Those two already synchronise, so stopping the
clock there adds no synchronisation to the solver path. Wall time is the host
clock over all of `run()`. Neither counts the steps before a resume.

### Accelerations are not stored in snapshots

They are a deterministic function of the positions, so recomputing them on load
is cheaper than storing them and guarantees the resumed run enters the loop in
exactly the state the original held.

### Physics formulas in host and device headers

So the kernels and the tests cannot drift apart, and so the formula level tests
run without a GPU. The one place this is deliberately avoided is the double
precision force reference, where sharing code would make the test and the
implementation agree by construction.

### Tolerances are always relative to a scale

Forced by a defect: a continuity test compared a quantity of order 1e6 against
an absolute tolerance of 1e-3 and failed a correct implementation. Every gate is
now relative to the magnitude of the quantity, the machine epsilon, or a bound
derived from the arithmetic.

## Build and tooling

### An `include/cps/physics` directory beyond the planned tree

The first layout listed `core`, `solvers` and `utils`. A fourth
directory holds the host and device shared formulas. Added because the sharing
it enables is what makes the CI test slice meaningful, and noted here as an
addition rather than a silent change.

### CUDA flags and architecture set before `project()`

Both are forced. nvcc's default C++17 cannot parse the libstdc++ 15 headers, and
CMake's compiler identification probe runs before any target exists, so the
standard has to be seeded first. `project()` assigns a default architecture the
moment it enables CUDA, so a guard placed after it never fires and the build
silently targets the wrong card.

### The device standard is detected, not asserted

The host standard is always C++23. The device standard is C++23 when the
running CMake lists `cuda_std_23` among its CUDA compile features, and C++20
otherwise, with the choice reported in the configuration summary.

This is not defensive coding for its own sake. Whether nvcc can be asked for
C++23 is a property of CMake rather than of the compiler: CMake 4.4 knows, the
CMake 3.28 that Ubuntu 24.04 ships does not, and asking the older one fails at
generate time with an error that blames the compiler. Nothing in the device
code needs a C++23 feature, and the whole project builds and passes every test
under C++20, so the fallback costs nothing. Override with
`-DCPS_CUDA_STANDARD=20` to exercise it deliberately.

### A host only build mode

`-DCPS_HOST_ONLY=ON` builds the configuration layer, the snapshot format, the
initial condition generators and the physics formulas with their unit tests,
and never enables the CUDA language at all. It exists so a machine with no
toolkit can still run a meaningful part of the suite, which is what the
continuous integration host job does. It is also what isolated the CMake
standard defect above: the host job passed while the CUDA job failed, which
located the fault immediately.

### GCC 15.2 rather than the planned GCC 16.1

Forced: nvcc 13.3 refuses any GCC newer than 15. Rejected
`-allow-unsupported-compiler`, because suppressing the toolkit's compatibility
check on a project whose output is trustworthy numbers is the wrong trade.

### The default tile width is <!-- num:BestTileWidth -->256<!-- /num -->

**Changed by measurement.** `v1.0.0` defaulted to 32 and reported its sweep. A width of
<!-- num:BestTileWidth -->256<!-- /num --> runs at <!-- num:BestTileMs -->11.487<!-- /num --> ms against <!-- num:TileThirtyTwoMs -->12.116<!-- /num --> at <!-- num:TileSweepParticles -->100,000<!-- /num --> particles, a
paired ratio of <!-- num:BestTileOverTileThirtyTwoRatio -->0.9483<!-- /num --> with an IQR of <!-- num:BestTileOverTileThirtyTwoRatioIqr -->0.0028<!-- /num -->,
far outside the session drift of <!-- num:RepeatabilitySpread -->0.17<!-- /num --> percent, so the default moved and the ladder
was measured there. Two barriers are paid per tile, so a narrow tile pays them more often per unit of
arithmetic, which fits the direction of the result; the barrier cost was not measured on its own. Each
width stays a compile time instantiation so the inner loop can be unrolled, and `nbody.tile_size` picks one
when the run starts.

### Bibliography page ranges written with the word "to"

The house style forbids en dashes anywhere, including compiled PDFs. BibTeX's
`plain` style converts hyphens in page ranges into en dashes when writing the
intermediate file, so cleaning the source had no effect. Writing "677 to 695"
passes through untouched and matches how the prose is required to write ranges.

## Decisions contradicted by measurement

### Fusion was expected to be worth 10 to 20 percent

**Measured at about 1 percent, and not resolved at the largest size.** Paired ratios of fused over
tiled are <!-- num:FusedOverTiledRatioAtTenThousand -->0.9445<!-- /num --> at 10,000 particles and
<!-- num:FusedOverTiledRatioAtHundredFiftyThousand -->0.9909<!-- /num --> at 150,000, each outside its IQR and the session drift
of <!-- num:RepeatabilitySpread -->0.17<!-- /num --> percent. At <!-- num:LadderMaxN -->200,000<!-- /num --> it is <!-- num:FusedOverTiledRatioAtMaxN -->0.9914<!-- /num --> with an IQR of
<!-- num:FusedOverTiledRatioIqrAtMaxN -->0.0276<!-- /num -->, because tiled trials are bimodal, near <!-- num:TiledLowModeMsAtMaxN -->43.1<!-- /num --> and
<!-- num:TiledHighModeMsAtMaxN -->44.4<!-- /num --> ms, for a reason not yet understood.

The all pairs sum is O(N^2) arithmetic over O(N) data, and device memory throughput is under one
percent of peak, which is why 10 to 20 percent was too high. But the traffic does not explain the gain
either. The split path reads the acceleration array in both integrator launches, 24N bytes; the fused
kernel reads it in neither, and writes 12N that the engine no longer reads. Counting each array once per
launch, fusion removes <!-- num:ModelBytesPerParticleSaved -->60<!-- /num -->N bytes, <!-- num:ModelSavingPercentAtMaxN -->0.048<!-- /num --> percent of a step at
the measured bandwidth. The source of the gain is unmeasured.

**Corrected.** `v1.0.0` said fusion was worth +0.1 percent, a difference below its own session spread
of about 3 percent. It called 0.019 percent, from a 24N round trip, an absolute ceiling, which the
measurements now exceed. It said the kernel halves the memory footprint of a step, which is false: it
adds a second position buffer. The kernel is kept because it is faster where the difference is
resolved.

### Tiling was framed as a bandwidth optimisation

**It is not one here.** Device memory throughput is <!-- num:NaiveDramPct -->0.34<!-- /num --> percent of peak before
and <!-- num:TiledDramPct -->0.32<!-- /num --> percent after. Tiling delivers its factor of <!-- num:TilingSpeedupAtMaxN -->1.96<!-- /num --> by relieving the
first level cache and load store pipeline: global load sectors fall <!-- num:SectorReduction -->63.9<!-- /num --> times,
L1 throughput falls from a near saturated <!-- num:NaiveLoneTexPct -->88.0<!-- /num --> percent to <!-- num:TiledLoneTexPct -->45.5<!-- /num --> percent, and
warp issue efficiency rises from <!-- num:NaiveIssuePct -->55.0<!-- /num --> to <!-- num:TiledIssuePct -->82.8<!-- /num --> percent.

### Register blocking was built, measured and removed

**It does not help this kernel.** Register blocking holds several target particles in each thread's
registers against one staged source tile, which needs fewer loads per interaction. It was built with
forces bit identical across blocking factors on both libraries, checked with racecheck, and measured
at commit <!-- num:BlockingCommit -->bd9dcfc40fc9<!-- /num --> over
<!-- num:BlockingRounds -->10<!-- /num --> interleaved rounds. Run time against the kernels before it,
with the thread count held constant (lower is faster):

| factor x block | tiled 10,000 | tiled 100,000 | tiled 200,000 | fused 10,000 | fused 100,000 | fused 200,000 |
|---|---:|---:|---:|---:|---:|---:|
| 1 x 256 | <!-- num:BlockingTiledFactorOneTenK -->1.0055<!-- /num --> | <!-- num:BlockingTiledFactorOneHundredK -->1.0007<!-- /num --> | <!-- num:BlockingTiledFactorOneTwoHundredK -->1.0014<!-- /num --> | <!-- num:BlockingFusedFactorOneTenK -->1.0844<!-- /num --> | <!-- num:BlockingFusedFactorOneHundredK -->1.0127<!-- /num --> | <!-- num:BlockingFusedFactorOneTwoHundredK -->1.0120<!-- /num --> |
| 2 x 128 | <!-- num:BlockingTiledFactorTwoTenK -->1.3007<!-- /num --> | <!-- num:BlockingTiledFactorTwoHundredK -->1.0037<!-- /num --> | <!-- num:BlockingTiledFactorTwoTwoHundredK -->0.9962<!-- /num --> | <!-- num:BlockingFusedFactorTwoTenK -->1.4137<!-- /num --> | <!-- num:BlockingFusedFactorTwoHundredK -->1.0342<!-- /num --> | <!-- num:BlockingFusedFactorTwoTwoHundredK -->1.0264<!-- /num --> |
| 4 x 64 | <!-- num:BlockingTiledFactorFourTenK -->2.2646<!-- /num --> | <!-- num:BlockingTiledFactorFourHundredK -->1.1090<!-- /num --> | <!-- num:BlockingTiledFactorFourTwoHundredK -->1.0505<!-- /num --> | <!-- num:BlockingFusedFactorFourTenK -->2.4050<!-- /num --> | <!-- num:BlockingFusedFactorFourHundredK -->1.1288<!-- /num --> | <!-- num:BlockingFusedFactorFourTwoHundredK -->1.0674<!-- /num --> |
| 8 x 32 | <!-- num:BlockingTiledFactorEightTenK -->6.0529<!-- /num --> | <!-- num:BlockingTiledFactorEightHundredK -->1.4249<!-- /num --> | <!-- num:BlockingTiledFactorEightTwoHundredK -->1.2501<!-- /num --> | <!-- num:BlockingFusedFactorEightTenK -->6.4478<!-- /num --> | <!-- num:BlockingFusedFactorEightHundredK -->1.4406<!-- /num --> | <!-- num:BlockingFusedFactorEightTwoHundredK -->1.2637<!-- /num --> |

The one result under 1 is tiled at factor 2 and 200,000 particles, and it is still
<!-- num:BlockingBestOverFusedAtMaxN -->1.0058<!-- /num --> of the existing fused kernel, the fastest
option at every size.

**Why.** The kernel is limited by arithmetic and not by loads: at 200,000 particles it runs
<!-- num:TiledFmaPerLoadAtMaxN -->12.5<!-- /num --> FMA instructions per load and issues at
<!-- num:TiledIssueActivePctTwoHundredK -->86.39<!-- /num --> percent of peak while active. Blocking
removes a small share of the loads and pays for it in registers. In a scratch `ncu` run, not a
measurement of record, registers per thread rose from 40 at factor 1 to 80 at factor 8, and the
occupancy limit fell from 6 to 3 warps. Rewriting the kernels also changed how the compiler scheduled
them, which is why fused is slower at factor 1. In the same scratch work the clean build took about
twice as long and the six architecture binary grew from about 5 MB to about 18 MB.

**Removed.** Code this costly with no gain does not stay in the tree. The data is in
`experiments/results/blocking_sweep.csv`. To reproduce it, check out the commit above and run
`scripts/blocking_sweep.py`, against a build of its parent, `018cc46b0114`.

### The runtime estimates were far too long

The first plan estimated 10 to 20 minutes for the validation suite on the
grounds that the 10,000 step orbit would dominate, but with two particles that
orbit is cheap. The benchmark sweep was estimated at 15 to 30 minutes and takes minutes.
