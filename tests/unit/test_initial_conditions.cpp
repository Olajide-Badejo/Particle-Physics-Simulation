#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/initial_conditions.hpp"

namespace {

cps::Config two_body_config() {
    cps::Config config;
    config.initial_condition = cps::InitialCondition::TwoBody;
    config.num_particles = 2;
    config.total_mass = 2.0f;
    config.two_body_separation = 1.0f;
    config.gravitational_constant = 1.0f;
    config.softening = 0.01f;
    return config;
}

cps::Config plummer_config(int count) {
    cps::Config config;
    config.initial_condition = cps::InitialCondition::Plummer;
    config.num_particles = count;
    config.total_mass = 1.0f;
    config.plummer_radius = 1.0f;
    config.gravitational_constant = 1.0f;
    config.seed = 20260802u;
    return config;
}

cps::Config dam_break_config(int count) {
    cps::Config config;
    config.solver = cps::SolverType::Sph;
    config.initial_condition = cps::InitialCondition::DamBreak;
    config.num_particles = count;
    config.seed = 20260802u;
    config.sph.domain_min = {0.0f, 0.0f, 0.0f};
    config.sph.domain_max = {0.8f, 0.5f, 0.4f};
    config.sph.fluid_min = {0.0f, 0.0f, 0.0f};
    config.sph.fluid_max = {0.3f, 0.4f, 0.4f};
    config.sph.rest_density = 1000.0f;
    return config;
}

TEST(TwoBody, PlacesTheMassesSymmetricallyAboutTheOrigin) {
    const cps::HostParticleState state = cps::make_two_body(two_body_config());
    ASSERT_EQ(state.size(), 2u);
    EXPECT_FLOAT_EQ(state.x[0], -0.5f);
    EXPECT_FLOAT_EQ(state.x[1], 0.5f);
    EXPECT_FLOAT_EQ(state.y[0], 0.0f);
    EXPECT_FLOAT_EQ(state.z[0], 0.0f);
    EXPECT_FLOAT_EQ(state.mass[0], 1.0f);
    EXPECT_FLOAT_EQ(state.mass[1], 1.0f);
}

TEST(TwoBody, StartsWithZeroNetMomentum) {
    const cps::HostParticleState state = cps::make_two_body(two_body_config());
    EXPECT_FLOAT_EQ(state.mass[0] * state.vx[0] + state.mass[1] * state.vx[1], 0.0f);
    EXPECT_FLOAT_EQ(state.mass[0] * state.vy[0] + state.mass[1] * state.vy[1], 0.0f);
    EXPECT_FLOAT_EQ(state.mass[0] * state.vz[0] + state.mass[1] * state.vz[1], 0.0f);
}

// The orbital speed has to come from the softened force law, not the bare one.
// Using the bare law leaves a residual radial acceleration that turns the
// circle into a precessing ellipse, and the energy drift gate would then be
// measuring the initial condition rather than the integrator.
TEST(TwoBody, DerivesTheOrbitalSpeedFromTheSoftenedForceLaw) {
    const cps::Config config = two_body_config();
    const cps::HostParticleState state = cps::make_two_body(config);

    const double m = 0.5 * config.total_mass;
    const double d = config.two_body_separation;
    const double eps = config.softening;
    const double expected = d * std::sqrt(m / (2.0 * std::pow(d * d + eps * eps, 1.5)));

    EXPECT_NEAR(std::abs(state.vy[1]), expected, expected * 1.0e-6);
    EXPECT_FLOAT_EQ(state.vy[0], -state.vy[1]);

    // And it must differ measurably from the unsoftened answer, otherwise the
    // test proves nothing.
    const double unsoftened = std::sqrt(m / (2.0 * d));
    EXPECT_GT(std::abs(expected - unsoftened), 1.0e-6);
}

TEST(Plummer, ConservesTheRequestedTotalMass) {
    const cps::Config config = plummer_config(5000);
    const cps::HostParticleState state = cps::make_plummer_sphere(config);
    ASSERT_EQ(state.size(), 5000u);
    double total = 0.0;
    for (float m : state.mass) {
        total += m;
    }
    EXPECT_NEAR(total, config.total_mass, config.total_mass * 1.0e-5);
}

TEST(Plummer, StartsCentredAndAtRestAsAWhole) {
    const cps::HostParticleState state = cps::make_plummer_sphere(plummer_config(5000));
    double mx = 0.0, my = 0.0, mz = 0.0, px = 0.0, py = 0.0, pz = 0.0, total = 0.0;
    for (std::size_t i = 0; i < state.size(); ++i) {
        const double m = state.mass[i];
        total += m;
        mx += m * state.x[i];
        my += m * state.y[i];
        mz += m * state.z[i];
        px += m * state.vx[i];
        py += m * state.vy[i];
        pz += m * state.vz[i];
    }
    // The generator subtracts the sampled centre of mass and net momentum, so
    // the momentum gate starts from zero rather than from a value that depends
    // on the seed.
    EXPECT_NEAR(mx / total, 0.0, 1.0e-5);
    EXPECT_NEAR(my / total, 0.0, 1.0e-5);
    EXPECT_NEAR(mz / total, 0.0, 1.0e-5);
    EXPECT_NEAR(px, 0.0, 1.0e-5);
    EXPECT_NEAR(py, 0.0, 1.0e-5);
    EXPECT_NEAR(pz, 0.0, 1.0e-5);
}

// The half mass radius of a Plummer sphere is a / sqrt(2^{2/3} - 1), about
// 1.305 a. Getting this right is what says the cumulative mass profile was
// inverted correctly rather than merely producing a plausible looking blob.
TEST(Plummer, ReproducesTheAnalyticHalfMassRadius) {
    const cps::Config config = plummer_config(20000);
    const cps::HostParticleState state = cps::make_plummer_sphere(config);

    std::vector<double> radii;
    radii.reserve(state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        radii.push_back(std::sqrt(static_cast<double>(state.x[i]) * state.x[i] +
                                  static_cast<double>(state.y[i]) * state.y[i] +
                                  static_cast<double>(state.z[i]) * state.z[i]));
    }
    std::sort(radii.begin(), radii.end());
    const double measured = radii[radii.size() / 2];
    const double expected = config.plummer_radius / std::sqrt(std::pow(2.0, 2.0 / 3.0) - 1.0);
    EXPECT_NEAR(measured, expected, expected * 0.05);
}

TEST(Plummer, IsReproducibleFromTheSeedAndSensitiveToIt) {
    const cps::HostParticleState first = cps::make_plummer_sphere(plummer_config(2000));
    const cps::HostParticleState second = cps::make_plummer_sphere(plummer_config(2000));
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(first.x[i], second.x[i]) << "index " << i;
        EXPECT_EQ(first.vy[i], second.vy[i]) << "index " << i;
    }

    cps::Config other = plummer_config(2000);
    other.seed = 99u;
    const cps::HostParticleState third = cps::make_plummer_sphere(other);
    EXPECT_NE(first.x[0], third.x[0]);
}

TEST(Plummer, ProducesNoNonFiniteValues) {
    const cps::HostParticleState state = cps::make_plummer_sphere(plummer_config(20000));
    for (std::size_t i = 0; i < state.size(); ++i) {
        ASSERT_TRUE(std::isfinite(state.x[i])) << "index " << i;
        ASSERT_TRUE(std::isfinite(state.y[i])) << "index " << i;
        ASSERT_TRUE(std::isfinite(state.z[i])) << "index " << i;
        ASSERT_TRUE(std::isfinite(state.vx[i])) << "index " << i;
        ASSERT_TRUE(std::isfinite(state.vy[i])) << "index " << i;
        ASSERT_TRUE(std::isfinite(state.vz[i])) << "index " << i;
    }
}

TEST(SphDerivedParameters, PutARoughlyStandardNumberOfNeighboursInsideTheSupport) {
    const cps::SphDerivedParameters derived = cps::derive_sph_parameters(dam_break_config(50000));
    EXPECT_GT(derived.smoothing_radius, 0.0f);
    EXPECT_GT(derived.particle_mass, 0.0f);
    // Two lattice spacings of support gives about thirty three neighbours.
    EXPECT_GT(derived.expected_neighbour_count, 20.0);
    EXPECT_LT(derived.expected_neighbour_count, 60.0);
}

TEST(SphDerivedParameters, SetTheParticleMassSoTheBlockSitsAtRestDensity) {
    const cps::Config config = dam_break_config(50000);
    const cps::SphDerivedParameters derived = cps::derive_sph_parameters(config);
    const double volume = 0.3 * 0.4 * 0.4;
    const double expected = config.sph.rest_density * volume / config.num_particles;
    EXPECT_NEAR(derived.particle_mass, expected, expected * 1.0e-4);
}

TEST(SphDerivedParameters, RespectExplicitOverrides) {
    cps::Config config = dam_break_config(1000);
    config.sph.smoothing_radius = 0.05f;
    config.sph.particle_mass = 0.001f;
    const cps::SphDerivedParameters derived = cps::derive_sph_parameters(config);
    EXPECT_FLOAT_EQ(derived.smoothing_radius, 0.05f);
    EXPECT_FLOAT_EQ(derived.particle_mass, 0.001f);
}

TEST(DamBreak, PlacesExactlyTheRequestedCountInsideTheFluidBlock) {
    const cps::Config config = dam_break_config(20000);
    const cps::HostParticleState state = cps::make_dam_break(config);
    ASSERT_EQ(state.size(), 20000u);

    const cps::SphDerivedParameters derived = cps::derive_sph_parameters(config);
    const float slack = derived.particle_spacing;
    for (std::size_t i = 0; i < state.size(); ++i) {
        EXPECT_GE(state.x[i], config.sph.fluid_min.x - slack);
        EXPECT_LE(state.x[i], config.sph.fluid_max.x + slack);
        EXPECT_GE(state.y[i], config.sph.fluid_min.y - slack);
        EXPECT_LE(state.y[i], config.sph.fluid_max.y + slack);
        EXPECT_GE(state.z[i], config.sph.fluid_min.z - slack);
        EXPECT_LE(state.z[i], config.sph.fluid_max.z + slack);
    }
}

TEST(DamBreak, StartsAtRestWithAUniformParticleMass) {
    const cps::Config config = dam_break_config(5000);
    const cps::HostParticleState state = cps::make_dam_break(config);
    const cps::SphDerivedParameters derived = cps::derive_sph_parameters(config);
    for (std::size_t i = 0; i < state.size(); ++i) {
        EXPECT_FLOAT_EQ(state.vx[i], 0.0f);
        EXPECT_FLOAT_EQ(state.vy[i], 0.0f);
        EXPECT_FLOAT_EQ(state.vz[i], 0.0f);
        EXPECT_FLOAT_EQ(state.mass[i], derived.particle_mass);
    }
}

// A perfectly regular lattice is a stationary point of the pressure force, so
// the block would sit still until rounding noise broke the symmetry. The
// jitter has to actually be there.
TEST(DamBreak, JittersTheLatticeOffItsExactSites) {
    const cps::Config config = dam_break_config(5000);
    const cps::HostParticleState state = cps::make_dam_break(config);
    const cps::SphDerivedParameters derived = cps::derive_sph_parameters(config);

    int off_lattice = 0;
    for (std::size_t i = 0; i < state.size(); ++i) {
        const float scaled = state.x[i] / derived.particle_spacing;
        if (std::abs(scaled - std::round(scaled)) > 1.0e-4f) {
            ++off_lattice;
        }
    }
    EXPECT_GT(off_lattice, static_cast<int>(state.size()) / 2);
}

TEST(DamBreak, IsReproducibleFromTheSeed) {
    const cps::HostParticleState first = cps::make_dam_break(dam_break_config(3000));
    const cps::HostParticleState second = cps::make_dam_break(dam_break_config(3000));
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(first.x[i], second.x[i]) << "index " << i;
        EXPECT_EQ(first.y[i], second.y[i]) << "index " << i;
        EXPECT_EQ(first.z[i], second.z[i]) << "index " << i;
    }
}

TEST(InitialConditions, DispatchOnTheConfiguredCase) {
    EXPECT_EQ(cps::make_initial_state(two_body_config()).size(), 2u);
    EXPECT_EQ(cps::make_initial_state(plummer_config(128)).size(), 128u);
    EXPECT_EQ(cps::make_initial_state(dam_break_config(512)).size(), 512u);
}

}  // namespace
