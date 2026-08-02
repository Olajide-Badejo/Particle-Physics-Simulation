#include "core/initial_conditions.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

#include "physics/hd_macros.hpp"

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

SphDerivedParameters derive_sph_parameters(const Config& config) {
    const SphConfig& sph = config.sph;
    const double lx = static_cast<double>(sph.fluid_max.x) - static_cast<double>(sph.fluid_min.x);
    const double ly = static_cast<double>(sph.fluid_max.y) - static_cast<double>(sph.fluid_min.y);
    const double lz = static_cast<double>(sph.fluid_max.z) - static_cast<double>(sph.fluid_min.z);
    const double volume = lx * ly * lz;
    if (!(volume > 0.0)) {
        throw std::runtime_error("the initial fluid block has no volume");
    }

    SphDerivedParameters derived;
    const double n = static_cast<double>(config.num_particles);
    derived.particle_spacing = static_cast<float>(std::cbrt(volume / n));

    // Mass follows from asking the block to sit at rest density when it is
    // filled. Choosing mass and density independently guarantees the fluid
    // either explodes or collapses on the first step.
    derived.particle_mass = sph.particle_mass > 0.0f
                                ? sph.particle_mass
                                : static_cast<float>(sph.rest_density * volume / n);

    // Two lattice spacings puts roughly thirty three particles inside the
    // support, which is the usual working range: fewer and the density
    // estimate is noisy, more and the neighbour loop dominates the step.
    derived.smoothing_radius =
        sph.smoothing_radius > 0.0f ? sph.smoothing_radius : 2.0f * derived.particle_spacing;

    const double h = derived.smoothing_radius;
    derived.expected_neighbour_count = (4.0 / 3.0) * PI_D * h * h * h * n / volume;
    return derived;
}

HostParticleState make_dam_break(const Config& config) {
    const int n = config.num_particles;
    const SphConfig& sph = config.sph;
    const SphDerivedParameters derived = derive_sph_parameters(config);

    HostParticleState state;
    state.resize(static_cast<std::size_t>(n));

    const double lx = static_cast<double>(sph.fluid_max.x) - static_cast<double>(sph.fluid_min.x);
    const double ly = static_cast<double>(sph.fluid_max.y) - static_cast<double>(sph.fluid_min.y);
    const double lz = static_cast<double>(sph.fluid_max.z) - static_cast<double>(sph.fluid_min.z);

    // Shrink the lattice pitch until the block holds at least the requested
    // number of sites. One pass of the ideal spacing usually falls a little
    // short because each axis rounds down.
    double spacing = derived.particle_spacing;
    long long nx = 0, ny = 0, nz = 0;
    for (int attempt = 0; attempt < 64; ++attempt) {
        nx = static_cast<long long>(std::floor(lx / spacing));
        ny = static_cast<long long>(std::floor(ly / spacing));
        nz = static_cast<long long>(std::floor(lz / spacing));
        nx = nx < 1 ? 1 : nx;
        ny = ny < 1 ? 1 : ny;
        nz = nz < 1 ? 1 : nz;
        if (nx * ny * nz >= n) {
            break;
        }
        spacing *= 0.98;
    }
    if (nx * ny * nz < n) {
        throw std::runtime_error("cannot fit the requested particle count into the fluid block");
    }

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
