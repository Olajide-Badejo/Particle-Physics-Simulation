#include "cps/core/initial_conditions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "cps/physics/hd_macros.hpp"
#include "cps/physics/sph_kernels.hpp"
#include "cps/physics/uniform_grid.hpp"

namespace cps {
namespace {

// A small deterministic generator wrapper.
//
// The standard distributions are not specified to produce the same sequence
// across library implementations or versions, and reproducibility is a
// deliverable here, so the uniform mapping is written out. The engine itself
// is standardised bit for bit, so the whole chain is portable.
class Rng {
   public:
    explicit Rng(uint32_t seed) : engine_(seed) {}

    // Uniform on the open interval (0, 1). The half offset keeps both
    // endpoints out, which matters because the Plummer radius sampler divides
    // by a quantity that is singular at both.
    double uniform() {
        return (static_cast<double>(engine_()) + 0.5) / 4294967296.0;
    }

    double uniform(double low, double high) { return low + (high - low) * uniform(); }

   private:
    std::mt19937 engine_;
};

void remove_centre_of_mass(HostParticleState& state) {
    const std::size_t n = state.size();
    if (n == 0) {
        return;
    }
    double total_mass = 0.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double m = state.mass[i];
        total_mass += m;
        cx += m * state.x[i];
        cy += m * state.y[i];
        cz += m * state.z[i];
        vx += m * state.vx[i];
        vy += m * state.vy[i];
        vz += m * state.vz[i];
    }
    if (total_mass <= 0.0) {
        return;
    }
    cx /= total_mass;
    cy /= total_mass;
    cz /= total_mass;
    vx /= total_mass;
    vy /= total_mass;
    vz /= total_mass;
    for (std::size_t i = 0; i < n; ++i) {
        state.x[i] -= static_cast<float>(cx);
        state.y[i] -= static_cast<float>(cy);
        state.z[i] -= static_cast<float>(cz);
        state.vx[i] -= static_cast<float>(vx);
        state.vy[i] -= static_cast<float>(vy);
        state.vz[i] -= static_cast<float>(vz);
    }
}

}  // namespace

HostParticleState make_two_body(const Config& config) {
    HostParticleState state;
    state.resize(2);

    const double m = 0.5 * static_cast<double>(config.total_mass);
    const double d = static_cast<double>(config.two_body_separation);
    const double g = static_cast<double>(config.gravitational_constant);
    const double eps = static_cast<double>(config.softening);

    // Balance the centripetal requirement against the softened attraction.
    // Deriving the speed from the softened law rather than the bare one is the
    // difference between a circular orbit and a slowly precessing ellipse, and
    // the drift gate would see the difference immediately.
    const double denominator = std::pow(d * d + eps * eps, 1.5);
    const double speed = d * std::sqrt(g * m / (2.0 * denominator));

    state.x[0] = static_cast<float>(-0.5 * d);
    state.x[1] = static_cast<float>(0.5 * d);
    state.vy[0] = static_cast<float>(-speed);
    state.vy[1] = static_cast<float>(speed);
    state.mass[0] = static_cast<float>(m);
    state.mass[1] = static_cast<float>(m);
    return state;
}

HostParticleState make_plummer_sphere(const Config& config) {
    const int n = config.num_particles;
    HostParticleState state;
    state.resize(static_cast<std::size_t>(n));

    Rng rng(config.seed);
    const double total_mass = config.total_mass;
    const double a = config.plummer_radius;
    const double g = config.gravitational_constant;
    const double per_particle = total_mass / static_cast<double>(n);
    const double velocity_scale = std::sqrt(g * total_mass / a);

    for (int i = 0; i < n; ++i) {
        // Radius by inverting the cumulative mass profile
        //   M(r)/M = r^3 / (r^2 + a^2)^{3/2}
        // The tail is unbounded, so a sample very close to one lands
        // arbitrarily far out; clamping the quantile keeps the sphere finite
        // without measurably disturbing the profile.
        double mass_fraction = rng.uniform();
        if (mass_fraction > 0.999) {
            mass_fraction = 0.999;
        }
        const double radius = a / std::sqrt(std::pow(mass_fraction, -2.0 / 3.0) - 1.0);

        // Uniform direction on the sphere.
        const double cos_theta = rng.uniform(-1.0, 1.0);
        const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        const double phi = rng.uniform(0.0, 2.0 * PI_D);

        state.x[i] = static_cast<float>(radius * sin_theta * std::cos(phi));
        state.y[i] = static_cast<float>(radius * sin_theta * std::sin(phi));
        state.z[i] = static_cast<float>(radius * cos_theta);

        // Speed from the exact Plummer distribution function by the rejection
        // sampler of Aarseth, Henon and Wielen. The accepted variable q is the
        // speed as a fraction of the local escape speed, and its density is
        // proportional to q^2 (1 - q^2)^{7/2}, whose maximum is below 0.1.
        // Drawing an isotropic Maxwellian instead would leave the sphere out
        // of virial equilibrium and it would visibly breathe on the first
        // crossing time, which looks exactly like an integrator bug.
        double q = 0.0;
        while (true) {
            const double candidate = rng.uniform();
            const double trial = rng.uniform(0.0, 0.1);
            const double density =
                candidate * candidate * std::pow(1.0 - candidate * candidate, 3.5);
            if (trial < density) {
                q = candidate;
                break;
            }
        }

        const double escape_speed =
            std::sqrt(2.0) * velocity_scale * std::pow(1.0 + radius * radius / (a * a), -0.25);
        const double speed = q * escape_speed;

        const double v_cos_theta = rng.uniform(-1.0, 1.0);
        const double v_sin_theta = std::sqrt(1.0 - v_cos_theta * v_cos_theta);
        const double v_phi = rng.uniform(0.0, 2.0 * PI_D);

        state.vx[i] = static_cast<float>(speed * v_sin_theta * std::cos(v_phi));
        state.vy[i] = static_cast<float>(speed * v_sin_theta * std::sin(v_phi));
        state.vz[i] = static_cast<float>(speed * v_cos_theta);
        state.mass[i] = static_cast<float>(per_particle);
    }

    // Sampling leaves a small random net momentum and centre of mass offset.
    // Removing both means the momentum gate starts from zero rather than from
    // an arbitrary constant, so it measures conservation and not the seed.
    remove_centre_of_mass(state);
    return state;
}

namespace {

// Past this many lattice pitches inside the support, each particle would have
// tens of millions of neighbours, and the cubic lattice sum would stall start up.
constexpr double MAX_SUPPORT_IN_PITCHES = 256.0;

struct FluidBlock {
    double lx = 0.0;
    double ly = 0.0;
    double lz = 0.0;
    double volume = 0.0;
};

FluidBlock fluid_block(const SphConfig& sph) {
    FluidBlock block;
    block.lx = static_cast<double>(sph.fluid_max.x) - static_cast<double>(sph.fluid_min.x);
    block.ly = static_cast<double>(sph.fluid_max.y) - static_cast<double>(sph.fluid_min.y);
    block.lz = static_cast<double>(sph.fluid_max.z) - static_cast<double>(sph.fluid_min.z);
    block.volume = block.lx * block.ly * block.lz;
    if (!(block.volume > 0.0)) {
        throw std::runtime_error("the initial fluid block has no volume");
    }
    return block;
}

float ideal_spacing(const FluidBlock& block, int count) {
    return static_cast<float>(std::cbrt(block.volume / static_cast<double>(count)));
}

}  // namespace

SphLattice sph_lattice(const Config& config) {
    const FluidBlock block = fluid_block(config.sph);
    const int n = config.num_particles;

    // Shrink the lattice pitch until the block holds at least the requested
    // number of sites. One pass of the ideal spacing usually falls a little
    // short because each axis rounds down.
    SphLattice lattice;
    lattice.pitch = ideal_spacing(block, n);
    for (int attempt = 0; attempt < 64; ++attempt) {
        lattice.nx = static_cast<long long>(std::floor(block.lx / lattice.pitch));
        lattice.ny = static_cast<long long>(std::floor(block.ly / lattice.pitch));
        lattice.nz = static_cast<long long>(std::floor(block.lz / lattice.pitch));
        lattice.nx = lattice.nx < 1 ? 1 : lattice.nx;
        lattice.ny = lattice.ny < 1 ? 1 : lattice.ny;
        lattice.nz = lattice.nz < 1 ? 1 : lattice.nz;
        if (lattice.nx * lattice.ny * lattice.nz >= n) {
            return lattice;
        }
        lattice.pitch *= 0.98;
    }
    throw std::runtime_error("cannot fit the requested particle count into the fluid block");
}

SphDerivedParameters derive_sph_parameters(const Config& config) {
    const SphConfig& sph = config.sph;
    const FluidBlock block = fluid_block(sph);

    SphDerivedParameters derived;
    derived.particle_spacing = ideal_spacing(block, config.num_particles);
    derived.lattice_pitch = sph_lattice(config).pitch;

    // Two ideal spacings puts roughly thirty three particles inside the support
    // at the ideal spacing, and a few more on the tighter lattice actually laid
    // out, which is the usual working range: fewer and the density estimate is
    // noisy, more and the neighbour loop dominates the step.
    derived.smoothing_radius =
        sph.smoothing_radius > 0.0f ? sph.smoothing_radius : 2.0f * derived.particle_spacing;
    const double h = derived.smoothing_radius;

    // Mass follows from asking the lattice actually laid out to sit at rest
    // density. Dividing the block's mass by the count instead assumes a spacing
    // the lattice does not have and ignores the kernel's own lattice sum error,
    // and by the equation of state the excess density is pressure, which
    // launches the fluid on the first step.
    if (sph.particle_mass > 0.0f) {
        derived.particle_mass = sph.particle_mass;
    } else {
        if (h / derived.lattice_pitch > MAX_SUPPORT_IN_PITCHES) {
            throw std::runtime_error(
                "the smoothing radius " + std::to_string(h) + " spans more than " +
                std::to_string(static_cast<int>(MAX_SUPPORT_IN_PITCHES)) + " lattice pitches of " +
                std::to_string(derived.lattice_pitch) + "; lower it, or set sph.particle_mass");
        }
        derived.particle_mass = static_cast<float>(calibrated_particle_mass(
            sph.rest_density, derived.lattice_pitch, h,
            [h](double r) { return poly6_kernel_fp64(r, h); }));
    }

    // Counted on the lattice laid out, whose pitch is what sets the number density.
    const double pitch = derived.lattice_pitch;
    derived.expected_neighbour_count = (4.0 / 3.0) * PI_D * h * h * h / (pitch * pitch * pitch);
    return derived;
}

double SphTimeStepLimits::smallest() const {
    return std::min(acoustic, std::min(viscous, body_force));
}

const char* SphTimeStepLimits::smallest_name() const {
    const double limit = smallest();
    if (limit == acoustic) {
        return "acoustic";
    }
    return limit == viscous ? "viscous" : "body force";
}

SphTimeStepLimits sph_time_step_limits(const Config& config, float smoothing_radius) {
    const SphConfig& sph = config.sph;
    const double h = smoothing_radius;
    const double unbounded = std::numeric_limits<double>::infinity();
    const double gx = config.external_gravity.x;
    const double gy = config.external_gravity.y;
    const double gz = config.external_gravity.z;
    const double gravity = std::sqrt(gx * gx + gy * gy + gz * gz);
    const double viscosity = sph.viscosity;

    SphTimeStepLimits limits;
    // The linear equation of state p = k (rho - rho_0) has a sound speed of
    // sqrt(k), whatever the rest density.
    limits.acoustic = 0.25 * h / std::sqrt(static_cast<double>(sph.gas_constant));
    limits.viscous = viscosity > 0.0
                         ? 0.125 * h * h * static_cast<double>(sph.rest_density) / viscosity
                         : unbounded;
    limits.body_force = gravity > 0.0 ? 0.25 * std::sqrt(h / gravity) : unbounded;
    return limits;
}

void check_sph_time_step(const Config& config, float smoothing_radius) {
    const SphTimeStepLimits limits = sph_time_step_limits(config, smoothing_radius);
    const double dt = config.dt;
    if (!(dt > limits.smallest())) {
        return;
    }
    char buffer[192];
    std::snprintf(buffer, sizeof(buffer),
                  "the time step dt = %g exceeds the %s limit %g (acoustic %g, viscous %g, body "
                  "force %g)",
                  dt, limits.smallest_name(), limits.smallest(), limits.acoustic, limits.viscous,
                  limits.body_force);
    if (!config.allow_unstable_dt) {
        throw std::runtime_error(std::string(buffer) +
                                 ". The walls would keep such a run finite and its results "
                                 "meaningless; lower dt, or pass --allow-unstable-dt to run "
                                 "anyway");
    }
    std::fprintf(stderr, "warning: %s; running anyway because unstable time steps are allowed\n",
                 buffer);
}

namespace {

// The dimensions of a grid and what its table takes, for the start up report and for
// every error about a grid. Takes the three counts rather than a grid, because a
// grid over the cell limit has products that do not fit its fields.
std::string describe_grid(int nx, int ny, int nz) {
    const uint64_t cells = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) *
                           static_cast<uint64_t>(nz);
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "%d x %d x %d cells, %llu in all, cell table %.3f MiB",
                  nx, ny, nz, static_cast<unsigned long long>(cells),
                  static_cast<double>(cell_table_bytes(cells)) / (1024.0 * 1024.0));
    return buffer;
}

}  // namespace

std::string sph_grid_description(const UniformGrid& grid) {
    return describe_grid(grid.nx, grid.ny, grid.nz);
}

void check_sph_grid_size(int nx, int ny, int nz) {
    const uint64_t cells = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) *
                           static_cast<uint64_t>(nz);
    if (cells <= SPH_GRID_CELL_LIMIT) {
        return;
    }
    throw std::runtime_error("the fluid grid is too large: " + describe_grid(nx, ny, nz) +
                             ", against a limit of " + std::to_string(SPH_GRID_CELL_LIMIT) +
                             " cells; raise the smoothing radius or shrink the domain");
}

UniformGrid make_sph_grid(const Vec3& domain_min, const Vec3& domain_max, float support) {
    const struct {
        const char* name;
        float low;
        float high;
    } axes[] = {{"x", domain_min.x, domain_max.x},
                {"y", domain_min.y, domain_max.y},
                {"z", domain_min.z, domain_max.z}};

    // In double precision, which is exact on the bounds and lets the limit be hit on
    // the nose. Every comparison is written so a NaN ratio fails it. The bounds count as
    // well as the width, because in a domain far from the origin the spacing of a float
    // position says nothing useful about the support, however few cells wide it is.
    const double h = static_cast<double>(support);
    const double limit = static_cast<double>(SPH_GRID_AXIS_LIMIT);
    const bool usable_support = support > 0.0f && std::isfinite(support);
    for (const auto& axis : axes) {
        const double low = static_cast<double>(axis.low);
        const double high = static_cast<double>(axis.high);
        if (usable_support && std::abs(low / h) < limit && std::abs(high / h) < limit &&
            (high - low) / h < limit) {
            continue;
        }
        char buffer[160];
        std::snprintf(buffer, sizeof(buffer),
                      "the domain [%g, %g] on axis %s is too large for cell size %g: ", low, high,
                      axis.name, h);
        throw std::runtime_error(std::string(buffer) +
                                 "|domain| and its width over the cell size must stay below " +
                                 std::to_string(SPH_GRID_AXIS_LIMIT) +
                                 " on every axis; raise the smoothing radius or shrink the domain");
    }

    const UniformGrid grid =
        make_uniform_grid(domain_min.x, domain_min.y, domain_min.z, domain_max.x, domain_max.y,
                          domain_max.z, support);
    check_sph_grid_size(grid.nx, grid.ny, grid.nz);
    return grid;
}

HostParticleState make_dam_break(const Config& config) {
    const int n = config.num_particles;
    const SphConfig& sph = config.sph;
    const SphDerivedParameters derived = derive_sph_parameters(config);
    const SphLattice lattice = sph_lattice(config);
    const FluidBlock block = fluid_block(sph);
    const double spacing = lattice.pitch;
    const long long nx = lattice.nx;
    const long long ny = lattice.ny;
    const long long nz = lattice.nz;
    const double lx = block.lx;
    const double ly = block.ly;
    const double lz = block.lz;

    HostParticleState state;
    state.resize(static_cast<std::size_t>(n));

    Rng rng(config.seed);
    const double jitter = static_cast<double>(sph.jitter) * spacing;

    // Centre the lattice inside the block so the fluid does not start pressed
    // against one wall by half a spacing.
    const double offset_x = 0.5 * (lx - static_cast<double>(nx - 1) * spacing);
    const double offset_y = 0.5 * (ly - static_cast<double>(ny - 1) * spacing);
    const double offset_z = 0.5 * (lz - static_cast<double>(nz - 1) * spacing);

    int written = 0;
    for (long long iy = 0; iy < ny && written < n; ++iy) {
        for (long long iz = 0; iz < nz && written < n; ++iz) {
            for (long long ix = 0; ix < nx && written < n; ++ix) {
                const double px = static_cast<double>(sph.fluid_min.x) + offset_x +
                                  static_cast<double>(ix) * spacing + rng.uniform(-jitter, jitter);
                const double py = static_cast<double>(sph.fluid_min.y) + offset_y +
                                  static_cast<double>(iy) * spacing + rng.uniform(-jitter, jitter);
                const double pz = static_cast<double>(sph.fluid_min.z) + offset_z +
                                  static_cast<double>(iz) * spacing + rng.uniform(-jitter, jitter);
                state.x[static_cast<std::size_t>(written)] = static_cast<float>(px);
                state.y[static_cast<std::size_t>(written)] = static_cast<float>(py);
                state.z[static_cast<std::size_t>(written)] = static_cast<float>(pz);
                state.mass[static_cast<std::size_t>(written)] = derived.particle_mass;
                ++written;
            }
        }
    }

    return state;
}

HostParticleState make_initial_state(const Config& config) {
    switch (config.initial_condition) {
        case InitialCondition::TwoBody:
            return make_two_body(config);
        case InitialCondition::Plummer:
            return make_plummer_sphere(config);
        case InitialCondition::DamBreak:
            return make_dam_break(config);
    }
    throw std::runtime_error("unhandled initial condition");
}

}  // namespace cps
