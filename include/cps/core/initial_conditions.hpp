#pragma once

#include <cmath>
#include <string>

#include "cps/core/config.hpp"
#include "cps/core/snapshot.hpp"
#include "cps/physics/uniform_grid.hpp"

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

// The lattice make_dam_break lays out. The pitch starts at the ideal spacing,
// cbrt(volume / count), and shrinks until the block holds at least the requested
// count, because each axis rounds its site count down. Filling stops at the
// requested count, so the top layer of the block can be partial.
struct SphLattice {
    double pitch = 0.0;
    long long nx = 0;
    long long ny = 0;
    long long nz = 0;
};

// Throws std::runtime_error if the block has no volume or cannot hold the count.
SphLattice sph_lattice(const Config& config);

// Derived smoothing radius and particle mass for the fluid cases. Exposed
// because the report quotes them and the tests check the neighbour count they
// imply.
//
// particle_spacing is the ideal spacing, cbrt(volume / count), which the
// smoothing radius derives from. lattice_pitch is the spacing actually laid out,
// which the particle mass derives from.
struct SphDerivedParameters {
    float particle_mass = 0.0f;
    float smoothing_radius = 0.0f;
    float particle_spacing = 0.0f;
    double lattice_pitch = 0.0;
    double expected_neighbour_count = 0.0;
};

SphDerivedParameters derive_sph_parameters(const Config& config);

// Sum of kernel(|r_j|) over an ideal unbounded cubic lattice of the given pitch,
// centred on one site: no jitter and no boundaries. kernel takes a distance and
// returns a double, and must be zero at and beyond support. Written against any
// kernel so that changing the kernel means running this again, not rewriting it.
template <typename Kernel>
double lattice_kernel_sum(double pitch, double support, Kernel kernel) {
    const long long reach = static_cast<long long>(std::ceil(support / pitch));
    double sum = 0.0;
    for (long long ix = -reach; ix <= reach; ++ix) {
        for (long long iy = -reach; iy <= reach; ++iy) {
            for (long long iz = -reach; iz <= reach; ++iz) {
                const double sites_sq = static_cast<double>(ix * ix + iy * iy + iz * iz);
                sum += kernel(pitch * std::sqrt(sites_sq));
            }
        }
    }
    return sum;
}

// The particle mass that makes the density sum over that lattice come out at
// rest_density exactly. Taking rest_density * pitch^3 instead would leave the
// kernel's own lattice sum error in the density.
template <typename Kernel>
double calibrated_particle_mass(double rest_density, double pitch, double support, Kernel kernel) {
    return rest_density / lattice_kernel_sum(pitch, support, kernel);
}

// Largest stable time step of the fluid solver under each of its three limits.
// A limit that does not apply, because the viscosity or gravity is zero, is
// infinite.
struct SphTimeStepLimits {
    double acoustic = 0.0;
    double viscous = 0.0;
    double body_force = 0.0;

    double smallest() const;
    // Names the limit smallest() returns: "acoustic", "viscous" or "body force".
    const char* smallest_name() const;
};

SphTimeStepLimits sph_time_step_limits(const Config& config, float smoothing_radius);

// Throws std::runtime_error when config.dt exceeds the smallest limit, naming
// the limit, its value and dt. With config.allow_unstable_dt it prints the same
// thing to stderr as a warning and returns. This is the only thing that catches
// a too large step in a fluid run: the walls clamp every position, so such a run
// stays finite and gives no other error.
void check_sph_time_step(const Config& config, float smoothing_radius);

// The grid the fluid solver searches, from its domain and the kernel support. Bounded
// domains only: the grid is as large as the domain and clamps at its walls. Throws
// std::runtime_error unless |domain_min|, |domain_max| and their distance apart are all
// below SPH_GRID_AXIS_LIMIT times the support on every axis, which a non finite ratio
// or a zero support fails as well, and unless check_sph_grid_size accepts the result.
UniformGrid make_sph_grid(const Vec3& domain_min, const Vec3& domain_max, float support);

// Throws std::runtime_error, giving the dimensions and the memory the table would
// take, if a grid of this shape is over SPH_GRID_CELL_LIMIT cells.
void check_sph_grid_size(int nx, int ny, int nz);

// "41 x 26 x 21 cells, ...", for the start up report and for the errors a grid raises.
std::string sph_grid_description(const UniformGrid& grid);

}  // namespace cps
