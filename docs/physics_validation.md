# Physics validation

Each gate, the theory behind it, what it is designed to catch, and what it
measured. Values come from `experiments/results/validation_gates.csv`, which the
validation suite writes on every run.

Every gate runs against `physics_core_strict`, the library compiled with fast
math disabled. The energy gate resolves one part in a thousand and the fast math
reciprocal square root does not leave that much headroom, so the two libraries
exist precisely so that this cannot be quietly relaxed.

## Summary

| gate | measured | limit | margin |
|---|---:|---:|---:|
| Two body energy drift over 10,000 steps | 1.33e-05 | < 1e-03 | 75x |
| Two body orbit departure from circular | 9.35e-06 | < 1e-03 | 107x |
| Plummer momentum drift over 1,000 steps | 1.23e-08 | < 1.19e-02 | 970,000x |
| Force against double precision CPU reference | 5.77e-07 | < 1e-05 | 17x |
| Pairwise force antisymmetry residual | 1.53e-07 | < 1e-06 | 6.6x |
| Plummer radial density chi squared p value | 0.4416 | > 0.05 | 8.8x |
| SPH poly6 kernel normalisation error | 6.48e-08 | < 1e-03 | 15,400x |
| SPH spiky kernel normalisation error | 4.39e-08 | < 1e-03 | 22,800x |
| SPH interior density relative error | 0.0420 | < 0.05 | 1.2x |
| SPH neighbour search against brute force | 3.20e-07 | < 1e-04 | 313x |

## Two body energy drift

**Theory.** Velocity Verlet is symplectic: it exactly conserves a modified
Hamiltonian close to the true one, so its energy error is bounded and oscillates
with the orbit rather than accumulating. A non symplectic scheme of the same
order, such as fourth order Runge Kutta, shows steady drift no matter how small
the step.

**What it catches.** A wrong integrator, an inconsistent softened potential, and
a mis-derived initial orbital speed. The orbital speed is computed from the
*softened* force law, so the orbit is genuinely circular; deriving it from the
bare law would leave a residual radial acceleration and turn the circle into a
slowly precessing ellipse, which the gate would see.

**Measured.** 1.33e-05 relative over 10,000 steps, seventy five times inside the
limit. Two supporting assertions: the worst error in the second half of the run
is compared against the first half, so a scheme that had begun to accumulate
would fail even with a small total error; and the separation stays between
1.000000 and 1.000019, giving a departure from circular of 9.35e-06.

## Momentum conservation

**Theory.** The pair interaction is antisymmetric, so the internal forces of an
isolated system cancel exactly and total momentum is conserved. On the device
they cancel only to rounding, because each thread accumulates its own sum in
single precision in its own order.

**What it catches.** Any asymmetry in the force law, and any bug in the
integrator that adds momentum.

**Why the limit has that shape.** An absolute tolerance would be meaningless.
The bound is what single precision can explain, `FLT_EPSILON * sqrt(N) *
steps`, which for 10,000 particles over 1,000 steps is 1.19e-02, measured
relative to the total absolute momentum in the system. The vector total is zero
by construction, so dividing by it would be dividing by noise.

**Measured.** 1.23e-08, roughly six orders of magnitude inside the bound. Also
checked for all three variants at 4,096 particles.

## Force antisymmetry and the CPU reference

**Theory.** Newton's third law: the force on `i` from `j` is exactly the
negative of the force on `j` from `i`.

**What it catches.** A sign error, a missing mass factor, an incorrect softening
term, and any coalescing or indexing bug in the tiled or fused kernels.

**How independence is preserved.** The reference is written separately in double
precision, as literally as the force law can be written, with no tiling and no
reciprocal square root. Its only job is to be obviously correct.

**Measured.** Worst relative deviation over 100 particles is 5.77e-07. The
pairwise antisymmetry residual over 4,950 pairs is 1.53e-07.

**Supporting checks.** Self interaction contributes exactly zero, so the tiled
kernel can include the diagonal term without a branch. Softening bounds the
acceleration at contact to `m*d/eps^3` as it should. The force falls off as the
inverse square at large separation. And the potential differentiated numerically
agrees with the force law, which is what guarantees the two come from the same
expression.

## Plummer radial density

**Theory.** The cumulative mass profile of a Plummer sphere is
`M(r)/M = r^3 / (r^2 + a^2)^(3/2)`, so the expected count in a shell follows
directly. A chi squared test asks the right question: not whether the sampler
produced a plausible blob, but whether the deviations are the size random
sampling explains.

**What it catches.** A sampler with a wrong exponent, which produces a cloud
that looks entirely reasonable.

**Design.** Shells are chosen to hold equal expected counts, so no bin is too
sparse for the statistic to be valid. The sampled quantile is clamped so the
sphere stays finite, and the test uses the same range, otherwise it would be
testing the clamp.

**Measured.** chi squared 39.635 on 39 degrees of freedom, p = 0.4416. Checked
across five seeds, of which at least four must pass a five percent test.

**The test is itself tested.** Run against a model whose scale radius is ten
percent wrong, it returns p below 1e-06. A test that cannot reject proves
nothing.

## SPH kernel normalisation

**Theory.** Each smoothing kernel integrates to one over the sphere of radius h.

**What it catches.** A mistyped normalisation constant or exponent. This matters
more than it looks: a kernel off by a few percent produces a density field wrong
by the same few percent everywhere, and the fluid still behaves like a fluid, it
just sits at the wrong rest density with the pressure term fighting gravity at
the wrong strength.

**Measured.** By composite Simpson quadrature at the smoothing radius the
50,000 particle configuration actually derives: poly6 error 6.48e-08, spiky
error 4.39e-08.

**Supporting checks.** Each kernel reaches zero at the support boundary relative
to its own peak, is exactly zero outside, and poly6 decreases monotonically. The
spiky gradient stays large near contact, which is the property poly6 lacks and
the reason a second kernel exists: a vanishing gradient at contact means no
restoring force exactly where the fluid needs one, and particles clump.

## Neighbour search against brute force

**Theory.** A hashed neighbour search must find exactly the neighbours a brute
force scan finds, no more and no fewer.

**What it catches.** The central risk in the fluid solver. Hashing is many to
one, so two of the twenty seven visited cells can land in the same bucket and a
naive scan counts those particles twice. Distance filtering does not help,
because they genuinely are within the smoothing radius. The defence is an exact
64 bit cell key compared during the scan, and this gate is what proves it works.

**Measured.** Worst relative density difference against an O(N^2) host sum over
identical positions is 3.20e-07, which is single precision rounding. A single
dropped or double counted neighbour would move a density by several percent.

**Supporting checks.** A separate health check confirms bucket collisions
actually occur at a small rate, so this gate is exercising the defence rather
than a table so sparse nothing collides.

## Interior density

**Theory.** In the interior of a block filled at rest density, the density
estimate must recover that density. Near a free surface it legitimately falls
short, because there is no fluid on the outside.

**Measured.** Mean interior density 1042.03 against a rest density of 1000, a
relative error of 4.20 percent, with a standard deviation of 83.83 over 25,047
interior particles. This is the tightest gate in the suite and the residual is
the expected consequence of a finite smoothing radius on a jittered lattice
rather than a defect.

## Mass conservation and the sort permutation

The fluid solver permutes its particle arrays every step, so mass is conserved
only if that permutation is a bijection. Two checks: the sorted multiset of
particle masses is unchanged across a step, and total mass over 2,000 steps
holds at 48.0000023 before and after. A gather that lost or duplicated an index
would fail both.

## Snapshot and resume bit identity

Not a physics gate but the same kind of claim. Running N steps straight through
must produce bit for bit the same state as running some steps, snapshotting,
resuming in a fresh process and running the rest. Asserted for all three N-body
variants and for the fluid solver, and for a run resumed twice.

A related test asserts that taking a full diagnostic after every step leaves the
final state bit for bit unchanged, which is why the half step velocity
correction is applied inside the reduction kernel rather than by shifting the
velocity array and shifting it back.
