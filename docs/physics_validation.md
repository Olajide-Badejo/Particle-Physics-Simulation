# Physics validation

Each gate, the theory behind it, what it is designed to catch, and what it
measured. Values come from `experiments/results/validation_gates.csv`. An
ordinary test run leaves that file alone and writes to `build/results`; only a
run with `CPS_RECORD_GATES=1` updates it.

Every device gate runs against `physics_core_strict`, the library compiled with
fast math disabled, so a compiler flag cannot quietly relax a gate.

## Summary

The kind says what a pass is evidence of. A **gate** can fail for a defect in the
code it names and its limit means something. A **consistency** check shows two
parts of the code agree, not that the physics is right. A **sanity** check has a
limit so far from the measured value, or so loosely founded, that a pass only
rules out gross breakage.

| gate | what it measures | measured | limit | kind | threshold source |
|---|---|---:|---:|---|---|
| `two_body_energy_drift_10000_steps` | fp32 energy error at 2.25 orbits, dominated by rounding | 1.33e-05 | < 1e-03 | gate | Project decision, empirical. Not derived. A regression test for rounding; it cannot see the integrator's order. |
| `two_body_orbit_out_of_round` | Departure of the fp32 two body orbit from circular | 9.35e-06 | < 1e-03 | sanity | Empirical bound, not derived. |
| `verlet_energy_slope_error_eccentric` | Distance from 2 of the fp64 energy error slope, eccentricity 0.5 (slope 2.007) | 0.0072 | < 0.15 | gate | Stated before measuring. The method's order. |
| `verlet_energy_slope_error_circular` | Distance from 4 of the fp64 energy error slope, unsoftened circular orbit (slope 3.996) | 0.0040 | < 0.15 | gate | Stated before measuring. Rotational symmetry cancels the second order part. |
| `plummer_momentum_drift_1000_steps` | Plummer momentum drift over 1,000 steps | 1.23e-08 | < 1.19e-02 | sanity | Derived upper bound on rounding, `FLT_EPSILON * sqrt(N) * steps`. Six orders of magnitude above the measured value, so only gross breakage reaches it. |
| `force_vs_cpu_reference_100_particles` | Force against the double precision reference | 5.77e-07 | < 1e-05 | gate | Empirical, not derived: a small multiple of single precision rounding. |
| `pairwise_force_antisymmetry_residual` | Pairwise force antisymmetry residual | 1.53e-07 | < 1e-06 | gate | Empirical, not derived: a small multiple of single precision rounding. |
| `plummer_radial_density_chi_square_p_value` | Plummer radial density chi squared p value | 0.4416 | > 0.05 | gate | Conventional five percent test level. With a fixed seed this is a fixed-seed regression: the value does not vary from run to run. |
| `sph_poly6_kernel_normalisation_error` | Poly6 kernel normalisation error | 6.48e-08 | < 1e-03 | sanity | Project decision, empirical. Four orders of magnitude above the quadrature's error. |
| `sph_spiky_kernel_normalisation_error` | Spiky kernel normalisation error | 4.39e-08 | < 1e-03 | sanity | Project decision, empirical. Four orders of magnitude above the quadrature's error. |
| `sph_interior_density_mean_error` | Interior mean density error with the calibrated mass | 2.81e-04 | < 0.01 | consistency | Stated before measuring. Project decision. |
| `sph_interior_density_rms_deviation` | Interior RMS density deviation about the mean | 0.0149 | < 0.05 | gate | Stated before measuring. Project decision. |
| `sph_interior_density_max_deviation` | Interior maximum density deviation from the rest density | 0.0634 | < 0.10 | sanity | Stated before measuring. Empirical bound, not derived. |
| `sph_neighbour_search_vs_brute_force` | Neighbour search density against a brute force sum | 2.85e-07 | < 1e-04 | gate | Empirical, chosen between rounding and the several percent one lost neighbour would cause. |

## Thresholds stated before measuring (v1.1)

These limits were committed before the v1.1 code that measures them existed, so
they cannot have been fitted to its results.

| quantity | limit | kind | source |
|---|---|---|---|
| Order of the double precision velocity Verlet energy error, orbit with eccentricity 0.5 | slope within 0.15 of 2 | gate | The method's order. The leading term of the modified Hamiltonian varies along an eccentric orbit. |
| The same quantity on an unsoftened circular orbit | slope within 0.15 of 4 | gate | Rotational symmetry holds that leading term constant on a circular orbit, so the second order part of the energy error cancels. |
| SPH interior mean density error with the calibrated mass | < 1 percent | consistency | Project decision. The calibration sets this value, so it shows the device density agrees with the host calibration and is not evidence of physical accuracy. |
| SPH interior density RMS deviation | < 5 percent | gate | Project decision. Lattice jitter drives it and the calibration does not set it. |
| SPH interior maximum density deviation | < 10 percent | sanity | Empirical bound, not derived. |

A scratch model on the host informed the choice of orbit and predicts slopes of
1.99 and 4.00 and an RMS deviation near 1.5 percent. Those are predictions, not
measurements of this code.

## Results against those thresholds

| quantity | predicted | measured | limit | result |
|---|---:|---:|---|---|
| Energy error slope, eccentricity 0.5 | 1.99 | 2.007 | within 0.15 of 2 | pass |
| Energy error slope, unsoftened circular orbit | 4.00 | 3.996 | within 0.15 of 4 | pass |
| Interior mean density error | | 0.028 percent | < 1 percent | pass |
| Interior RMS density deviation | 1.5 percent | 1.49 percent | < 5 percent | pass |
| Interior maximum density deviation | | 6.34 percent | < 10 percent | pass |

No limit moved. The interior mean only shows that the device agrees with the host
calibration, as the table above says. The sections below give the details.

## fp32 energy error at 2.25 orbits, dominated by rounding

This is the gate `two_body_energy_drift_10000_steps`, which earlier versions
called the two body energy drift.

**Theory.** Velocity Verlet is symplectic: it exactly conserves a modified
Hamiltonian close to the true one, so its energy error is bounded and oscillates
with the orbit rather than accumulating. A non symplectic scheme of the same
order, such as fourth order Runge Kutta, shows steady drift no matter how small
the step.

**What it catches.** A broken single precision force or update path, an
inconsistent softened potential, and a mis-derived initial orbital speed. The
orbital speed is computed from the *softened* force law, so the orbit is
genuinely circular; deriving it from the bare law would leave a residual radial
acceleration and turn the circle into a slowly precessing ellipse, which the
gate would see.

**What it cannot catch.** The integrator's order. The same step size, softening,
orbit and duration in double precision give a truncation error of 9.88e-13, seven
orders of magnitude below this gate's value, so nearly all of what it records is
single precision rounding. It stays a valid regression test for rounding. The
order has its own gates in the next section.

**Measured.** 1.33e-05 relative over 10,000 steps. Two supporting assertions: the
worst error in the second half of the run is compared against the first half, so
a scheme that had begun to accumulate would fail even with a small total error;
and the separation stays between 1.000000 and 1.000019, giving a departure from
circular of 9.35e-06.

## Order of accuracy

**Theory.** Velocity Verlet is second order, so on a general orbit its energy
error falls as `dt^2`. The leading term of its modified Hamiltonian varies along
an eccentric orbit, which is what makes that error visible. On a circular orbit
rotational symmetry holds that term constant, so its contribution cancels and the
energy error is close to fourth order. A second order slope therefore needs an
eccentric orbit, and a fit on the circular orbit is a separate check.

**What it catches.** An update that is not the method it claims to be. In a scratch
model, symplectic Euler fits slopes of 1.16 and 2.00, and a closing half kick with
a stale acceleration fits 0.62 and 0.66, so each fails both gates.

**How independence is preserved.** The test runs on the host in double precision
with `reference_accelerations`, at a fixed T = 10 over `dt` of 64, 32, 16, 8 and
4 e-3. It fits the slope of log max|dE/E| against log dt.

**Measured.** Slope 2.007 at eccentricity 0.5 with softening 0.01, and 3.996 on
the unsoftened circular orbit. The whole test takes well under a second.

## Momentum conservation

A sanity check, not a gate: the limit is a bound on what rounding could explain.

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

**What the p value means here.** The recorded value comes from one fixed seed, so
it is a fixed-seed regression: the same code gives the same number. It does not
vary with each run the way a fresh p value would, so its distance from 0.05 says
nothing about how well the sampler fits.

**The test is itself tested.** Run against a model whose scale radius is ten
percent wrong, it returns p below 1e-06. A test that cannot reject proves
nothing.

## SPH kernel normalisation

Both are sanity checks: the limit sits four orders of magnitude above the
quadrature's error.

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
identical positions is 2.85e-07, which is single precision rounding. A single
dropped or double counted neighbour would move a density by several percent.

**Supporting checks.** A separate health check confirms bucket collisions
actually occur at a small rate, so this gate is exercising the defence rather
than a table so sparse nothing collides.

## Interior density

**Theory.** In the interior of a block filled at rest density, the density
estimate must recover that density. Near a free surface it legitimately falls
short, because there is no fluid on the outside.

**Mass calibration.** The particle mass is `rest_density / sum_j W(|r_j|, h)`,
summed over an ideal unbounded lattice of the pitch actually laid out, with no
jitter and no boundaries, on the host in double precision. For the 50,000
particle block the pitch is 9.667551e-03 and the sum times `pitch^3` is 1.01257.
An explicit `sph.particle_mass` still overrides it. The function takes the kernel
as a parameter, so a different kernel means running it again.

**What each quantity shows.** Over the interior, which is the 25,047 particles at
least 2h inside the block:

- The mean error is what the calibration sets, so it is a consistency check: it
  shows the device density agrees with the host calibration, and is not evidence
  of physical accuracy.
- The RMS deviation is taken about the interior mean, so the calibration does not
  set it and the lattice jitter drives it.
- The maximum deviation is taken about the rest density, since that difference is
  what the equation of state turns into pressure.

**Measured.** Mean 1000.28 against a rest density of 1000, an error of 0.028
percent. Standard deviation 14.89, an RMS deviation of 1.49 percent. Maximum
deviation 6.34 percent.

**Correction to v1.0.** v1.0 reported an interior density error of 4.20 percent,
with a mean of 1042.03 and a standard deviation of 83.83. Those figures came from
a mismatched index: the test chose interior particles by position in lattice
order, then read densities in cell sorted order, so positions and densities
belonged to different particles. The figures were close to the statistics of the
whole block, free surface included.

With the index fixed, the true value at the v1 mass was a mean of 1076.14 and a
standard deviation of 16.02, an error of 7.61 percent, which fails the 5 percent
limit that gate had. The v1 mass was `rest_density * volume / count`, which assumes
a spacing the lattice does not have: the realised pitch gives a number density 6.2
percent higher, and the kernel's lattice sum adds 1.26 percent.

The earlier explanation, that the residual was "the expected consequence of a
finite smoothing radius on a jittered lattice", was wrong. Almost all of it was the
mass.

**Effect on the dam break.** The equation of state turns excess density directly
into pressure, so the v1 mass started the interior near `k * 76 = 76 kPa`, against
a hydrostatic base pressure of `rho g H = 3.9 kPa`. For `presets/dam_break_50k.json`
the maximum particle speed after 100 steps fell from 3.695 m/s to 0.625 m/s with the
calibrated mass. Free fall alone would give 0.098 m/s, so the start is still not
quiet. There is no gate on this here; a hydrostatic test is planned for a later release.

## Time step limits

The fluid solver computes three limits when it is constructed, prints them at
start up, and refuses a `dt` above the smallest, naming the limit, its value and
`dt`.

| limit | formula | `presets/dam_break_50k.json` |
|---|---|---:|
| acoustic | `0.25 h / sqrt(gas_constant)`, since `c = sqrt(k)` for this equation of state | 1.56e-04 |
| viscous | `0.125 h^2 rest_density / viscosity`, skipped when the viscosity is zero | 0.243 |
| body force | `0.25 sqrt(h / abs(g))`, skipped when gravity is zero | 0.0112 |

The shipped `dt` of 1e-4 is inside all three. The 0.25 and 0.125 factors are the
usual safety margins, not derived for this solver.

This check is what stops a too large step from going unnoticed. The walls clamp
every position, so a fluid run at `dt` from 0.002 up to 1e4 stays finite and gives
meaningless results with no other error. `--allow-unstable-dt`, or
`allow_unstable_dt` in the configuration, downgrades the refusal to a warning. That
field is run control, so it stays out of the physics fingerprint.

## Mass conservation and the sort permutation

The fluid solver permutes its particle arrays every step, so mass is conserved
only if that permutation is a bijection. Two checks: the sorted multiset of
particle masses is unchanged across a step, and total mass over 2,000 steps
holds at 41.9188244 before and after. A gather that lost or duplicated an index
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
