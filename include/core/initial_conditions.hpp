#pragma once

#include "core/config.hpp"
#include "core/snapshot.hpp"

namespace cps {

// Initial conditions are generated on the host from a named seed and then
// uploaded once. Generating them on the host costs a fraction of a second even
// at two hundred thousand particles and buys exact reproducibility across
// driver versions, which a device side random number generator would not.

// Two equal masses on a circular orbit about their common centre of mass.
// The orbital speed follows from balancing the centripetal requirement against
// the softened attraction, so the softening length is part of the setup rather
// than a perturbation applied to it.
HostParticleState make_two_body(const Config& config);

// Plummer sphere, sampled by inverting the cumulative mass profile
//
//     M(r) / M = r^3 / (r^2 + a^2)^{3/2}
//
// which gives r = a / sqrt(X^{-2/3} - 1) for X uniform on the unit interval.
// Velocities are drawn from the exact Plummer distribution function using the
// rejection sampler from Aarseth, Henon and Wielen, so the sphere starts in
// virial equilibrium instead of collapsing on the first crossing time.
HostParticleState make_plummer_sphere(const Config& config);

// Axis aligned block of fluid on a jittered lattice. The jitter breaks the
// lattice symmetry: a perfectly regular lattice is a stationary point of the
// pressure force and the block sits there doing nothing until numerical noise
// eventually breaks it, which looks exactly like a bug.
HostParticleState make_dam_break(const Config& config);

HostParticleState make_initial_state(const Config& config);

// Derived smoothing radius and particle mass for the fluid cases. Exposed
// because the report quotes them and the tests check the neighbour count they
// imply.
struct SphDerivedParameters {
    float particle_mass = 0.0f;
    float smoothing_radius = 0.0f;
    float particle_spacing = 0.0f;
    double expected_neighbour_count = 0.0;
};

SphDerivedParameters derive_sph_parameters(const Config& config);

// Throws std::runtime_error unless |domain_min| / cell_size and
// |domain_max| / cell_size are below SPH_DOMAIN_CELL_LIMIT on every axis. A non
// finite ratio, such as one from a zero cell size, fails as well. Past the
// limit two different cells could share a cell key, and the neighbour search
// would mix up their particles without any error.
void check_sph_domain_cell_range(const Vec3& domain_min, const Vec3& domain_max, float cell_size);

}  // namespace cps
