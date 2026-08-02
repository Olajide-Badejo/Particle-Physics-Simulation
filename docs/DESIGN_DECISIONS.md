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

### Double buffered positions on the fused path

Required for correctness, not performance. There is no grid wide barrier inside
a launch, so updating in place would let a block that has finished its force
sum overwrite a position another block is still reading. Costs 12N bytes and no
bandwidth.

### Spatial hashing rather than a dense grid or a k-d tree

A dense grid needs storage proportional to the bounding volume; a dam break
occupies a small and moving part of its domain, so most of it would be empty. A
k-d tree is a poor fit for a full GPU rebuild every step.

### Exact cell keys in the neighbour scan

Hashing is many to one, so two of the twenty seven visited cells can collide in
the table and a naive scan counts those particles twice. Distance filtering does
not help, because they genuinely are within the smoothing radius. An exact 64
bit cell key compared per candidate rejects both foreign particles and double
visits at one comparison each. Rejected deduplicating the twenty seven hashed
buckets, which costs hundreds of comparisons per particle rather than one per
candidate.

### Three smoothing kernels rather than one

poly6 for density, because it is a polynomial in r squared and needs no square
root. spiky for the pressure gradient, because its gradient does not vanish at
contact and poly6's does, which means no restoring force exactly where the fluid
needs one and particles clump. A viscosity kernel with a positive Laplacian
throughout its support, so the viscous term always damps and never injects
energy.

### Derived smoothing radius and particle mass

Both default to being computed from the fluid volume and particle count: mass so
the block sits at rest density when filled, and the smoothing radius at two
lattice spacings, giving about 33 neighbours. Fixing either by hand while the
particle count moves is the fastest way to a resolution the kernel cannot
support. Explicit values in the configuration still override.

### Jittered initial lattice

A perfectly regular lattice is a stationary point of the pressure force, so the
block sits still until rounding noise breaks the symmetry, which looks exactly
like a bug.

### Plummer velocities from the exact distribution function

Rejected a convenient Maxwellian, which would leave the sphere out of virial
equilibrium so that it visibly breathes on the first crossing time, again
looking exactly like an integrator bug. The measured virial ratio at t = 0 is
1.0009.

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

This is a deliberate deviation from the original specification, which described
a float4 per step. The traffic difference is one small array and nothing per
particle.

### Deterministic reductions

Block partials are summed on the host in a fixed order rather than accumulated
with atomics, whose ordering varies between launches. Atomics would have been
shorter and would have made two runs of the same configuration disagree in the
last bits, which the bit identity test forbids.

### The half step correction lives in the reduction kernel

Shifting the velocity array forward and back would not return the original bits
in single precision, so a run that took a diagnostic would diverge from one that
did not.

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

### An `include/physics` directory beyond the specified tree

The specification's layout lists `core`, `solvers` and `utils`. A fourth
directory holds the host and device shared formulas. Added because the sharing
it enables is what makes the CI test slice meaningful, and noted here as an
addition rather than a silent change.

### CUDA flags and architecture set before `project()`

Both are forced. nvcc's default C++17 cannot parse the libstdc++ 15 headers, and
CMake's compiler identification probe runs before any target exists, so the
standard has to be seeded first. `project()` assigns a default architecture the
moment it enables CUDA, so a guard placed after it never fires and the build
silently targets the wrong card.

### GCC 15.2 rather than the specified GCC 16.1

Forced: nvcc 13.3 refuses any GCC newer than 15. Rejected
`-allow-unsupported-compiler`, because suppressing the toolkit's compatibility
check on a project whose output is trustworthy numbers is the wrong trade.

### Tile width stays at 32 despite the measurement

**Contradicted by measurement.** A tile width of 256 is 5.5 percent faster than
the specified 32 on this card, because two barriers are paid per tile and a
narrow tile pays them more often per unit of arithmetic. The default was kept so
the ladder measures the design as specified; changing it partway would have made
the rungs incomparable. The sweep is reported so the choice is informed, and
exposing the width as a configuration field is the right resolution.

### Bibliography page ranges written with the word "to"

The house style forbids en dashes anywhere, including compiled PDFs. BibTeX's
`plain` style converts hyphens in page ranges into en dashes when writing the
intermediate file, so cleaning the source had no effect. Writing "677 to 695"
passes through untouched and matches how the prose is required to write ranges.

## Decisions contradicted by measurement

### Fusion was expected to be worth 10 to 20 percent

**Measured at +0.1 percent** at 200,000 particles. The all pairs sum is O(N^2)
arithmetic over O(N) data, so it is not waiting on memory: device memory
throughput is under one percent of peak. The 24N bytes fusion removes are worth
0.019 percent of the step as an absolute ceiling. The kernel is kept because it
wins at small particle counts and halves the memory footprint of a step.

### Tiling was framed as a bandwidth optimisation

**It is not one here.** Device memory throughput is 0.50 percent of peak before
and 0.84 percent after. Tiling delivers its factor of 1.93 by relieving the
first level cache and load store pipeline: global load sectors fall 63.8 times,
L1 throughput falls from a saturated 79.6 percent to 39.7 percent, and warp
issue efficiency rises from 49.7 to 74.2 percent.

### The runtime estimates were far too long

The specification estimated 10 to 20 minutes for the validation suite on the
grounds that the 10,000 step orbit would dominate. It completes in about five
seconds, because with two particles that orbit is trivially cheap and the cost
is dominated instead by the 10,000 particle Plummer runs. The benchmark sweep,
estimated at 15 to 30 minutes, completes in under two minutes.
