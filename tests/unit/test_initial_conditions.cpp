#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/core/initial_conditions.hpp"
#include "cps/physics/spatial_hash.hpp"
#include "cps/physics/sph_kernels.hpp"

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
    // Two ideal spacings of support, on the lattice actually laid out.
    EXPECT_GT(derived.expected_neighbour_count, 20.0);
    EXPECT_LT(derived.expected_neighbour_count, 60.0);
}

TEST(SphDerivedParameters, ExposeThePitchTheLatticeActuallyHas) {
    const cps::Config config = dam_break_config(50000);
    const cps::SphLattice lattice = cps::sph_lattice(config);
    const cps::SphDerivedParameters derived = cps::derive_sph_parameters(config);
    EXPECT_NEAR(lattice.pitch, 9.667551e-03, 1.0e-9);
    EXPECT_EQ(derived.lattice_pitch, lattice.pitch);
    EXPECT_EQ(lattice.nx, 31);
    EXPECT_EQ(lattice.ny, 41);
    EXPECT_EQ(lattice.nz, 41);

    // The smoothing radius still derives from the ideal spacing, which one pass
    // of the search shrank away from.
    EXPECT_FLOAT_EQ(derived.smoothing_radius, 2.0f * derived.particle_spacing);
    EXPECT_LT(derived.lattice_pitch, static_cast<double>(derived.particle_spacing));
}

TEST(SphDerivedParameters, CalibrateTheMassAgainstTheLatticeKernelSum) {
    const cps::Config config = dam_break_config(50000);
    const cps::SphDerivedParameters derived = cps::derive_sph_parameters(config);
    const double h = derived.smoothing_radius;
    const double pitch = derived.lattice_pitch;
    const double sum = cps::lattice_kernel_sum(
        pitch, h, [h](double r) { return cps::poly6_kernel_fp64(r, h); });

    // The value this configuration predicts, by a separate calculation.
    EXPECT_NEAR(sum * pitch * pitch * pitch, 1.01257, 5.0e-6);

    // Times the sum, the mass recovers the rest density on the ideal lattice.
    const double recovered = static_cast<double>(derived.particle_mass) * sum;
    EXPECT_NEAR(recovered, config.sph.rest_density, config.sph.rest_density * 1.0e-6);
}

// A kernel with a known sum, to show the sum really is over the lattice and is
// not tied to poly6. Sites strictly closer than two pitches are the 27 with
// squared distances 0, 1, 2 and 3: 1 + 6 + 12 + 8. The sites at exactly two
// pitches are out.
TEST(SphCalibration, WorksWithAnyKernel) {
    const double pitch = 0.5;
    const double support = 2.0 * pitch;
    const auto top_hat = [support](double r) { return r < support ? 1.0 : 0.0; };
    EXPECT_EQ(cps::lattice_kernel_sum(pitch, support, top_hat), 27.0);
    EXPECT_DOUBLE_EQ(cps::calibrated_particle_mass(1000.0, pitch, support, top_hat), 1000.0 / 27.0);
}

// A wider kernel than its radius parameter, as a kernel with support 2h has: the
// support is what bounds the sum.
TEST(SphCalibration, TheSumReachesTheWholeSupport) {
    const double pitch = 1.0;
    const auto top_hat = [](double r) { return r < 3.5 ? 1.0 : 0.0; };
    int expected = 0;
    for (int ix = -4; ix <= 4; ++ix) {
        for (int iy = -4; iy <= 4; ++iy) {
            for (int iz = -4; iz <= 4; ++iz) {
                expected += ix * ix + iy * iy + iz * iz < 3.5 * 3.5 ? 1 : 0;
            }
        }
    }
    EXPECT_EQ(cps::lattice_kernel_sum(pitch, 3.5, top_hat), static_cast<double>(expected));
}

TEST(SphCalibration, TheDoublePrecisionPolySixMatchesTheOneTheDeviceRuns) {
    const float h = 0.02f;
    for (int i = 0; i <= 25; ++i) {
        const double r = 0.001 * i;
        const double single = cps::poly6_kernel(static_cast<float>(r * r), h);
        const double fp64 = cps::poly6_kernel_fp64(r, static_cast<double>(h));
        EXPECT_NEAR(single, fp64, 1.0e-5 * std::max(1.0, fp64)) << "r = " << r;
    }
}

TEST(SphCalibration, RefusesASupportTooWideToSum) {
    cps::Config config = dam_break_config(50000);
    config.sph.smoothing_radius = 1.0e3f;
    EXPECT_THROW(cps::derive_sph_parameters(config), std::runtime_error);
    // An explicit mass needs no sum, so the same radius is not refused here.
    config.sph.particle_mass = 0.001f;
    EXPECT_NO_THROW(cps::derive_sph_parameters(config));
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

// The lattice laid out and the lattice the mass was calibrated for must be the
// same one. A row of neighbouring sites steps by the exposed pitch.
TEST(DamBreak, LaysOutTheLatticeTheMassWasCalibratedFor) {
    const cps::Config config = dam_break_config(50000);
    const cps::HostParticleState state = cps::make_dam_break(config);
    const cps::SphLattice lattice = cps::sph_lattice(config);
    const double pitch = cps::derive_sph_parameters(config).lattice_pitch;
    const std::size_t last = static_cast<std::size_t>(lattice.nx) - 1;
    const double span = static_cast<double>(state.x[last]) - static_cast<double>(state.x[0]);
    const double jitter = 2.0 * static_cast<double>(config.sph.jitter) * pitch;
    EXPECT_NEAR(span, static_cast<double>(last) * pitch, jitter + 1.0e-6);
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

// The message of the error the time step check has to raise.
std::string time_step_error(const cps::Config& config) {
    const float h = cps::derive_sph_parameters(config).smoothing_radius;
    try {
        cps::check_sph_time_step(config, h);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    ADD_FAILURE() << "expected the time step check to throw";
    return "";
}

cps::Config shipped_dam_break() {
    return cps::Config::from_json_file(std::filesystem::path(CPS_PROJECT_ROOT) / "configs" /
                                       "dam_break_50k.json");
}

TEST(SphTimeStep, TheLimitsFollowTheirFormulas) {
    const cps::Config config = shipped_dam_break();
    const float h = cps::derive_sph_parameters(config).smoothing_radius;
    const double wide_h = h;
    const double viscosity = config.sph.viscosity;
    const double gravity = std::abs(static_cast<double>(config.external_gravity.y));
    const cps::SphTimeStepLimits limits = cps::sph_time_step_limits(config, h);
    EXPECT_DOUBLE_EQ(limits.acoustic, 0.25 * wide_h / std::sqrt(1000.0));
    EXPECT_DOUBLE_EQ(limits.viscous, 0.125 * wide_h * wide_h * 1000.0 / viscosity);
    EXPECT_DOUBLE_EQ(limits.body_force, 0.25 * std::sqrt(wide_h / gravity));
    EXPECT_STREQ(limits.smallest_name(), "acoustic");
    EXPECT_DOUBLE_EQ(limits.smallest(), limits.acoustic);
}

TEST(SphTimeStep, TheShippedConfigurationPasses) {
    const cps::Config config = shipped_dam_break();
    EXPECT_NO_THROW(cps::check_sph_time_step(
        config, cps::derive_sph_parameters(config).smoothing_radius));
}

// With k = 1e4 the sound speed is 100, so the limit is 0.25 h / 100, about
// 4.93e-5, which a step of 1e-4 exceeds.
TEST(SphTimeStep, AStiffGasConstantIsRefusedAndNamed) {
    cps::Config config = shipped_dam_break();
    config.sph.gas_constant = 1.0e4f;
    const std::string message = time_step_error(config);
    EXPECT_NE(message.find("acoustic limit 4.93"), std::string::npos) << message;
    EXPECT_NE(message.find("dt = 0.0001"), std::string::npos) << message;
    EXPECT_NE(message.find("--allow-unstable-dt"), std::string::npos) << message;
}

TEST(SphTimeStep, TheFlagDowngradesTheRefusalToAWarning) {
    cps::Config config = shipped_dam_break();
    config.sph.gas_constant = 1.0e4f;
    config.allow_unstable_dt = true;
    const float h = cps::derive_sph_parameters(config).smoothing_radius;
    ::testing::internal::CaptureStderr();
    EXPECT_NO_THROW(cps::check_sph_time_step(config, h));
    const std::string warning = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(warning.find("warning:"), std::string::npos) << warning;
    EXPECT_NE(warning.find("acoustic limit"), std::string::npos) << warning;
}

TEST(SphTimeStep, WarnsOfNothingWhenTheStepIsStable) {
    cps::Config config = shipped_dam_break();
    config.allow_unstable_dt = true;
    const float h = cps::derive_sph_parameters(config).smoothing_radius;
    ::testing::internal::CaptureStderr();
    cps::check_sph_time_step(config, h);
    EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
}

TEST(SphTimeStep, TheViscousLimitIsNamedWhenItIsTheSmallest) {
    cps::Config config = shipped_dam_break();
    config.sph.viscosity = 1.0e4f;
    EXPECT_NE(time_step_error(config).find("viscous limit"), std::string::npos);
}

TEST(SphTimeStep, TheBodyForceLimitIsNamedWhenItIsTheSmallest) {
    cps::Config config = shipped_dam_break();
    config.external_gravity = {0.0f, -1.0e9f, 0.0f};
    EXPECT_NE(time_step_error(config).find("body force limit"), std::string::npos);
}

TEST(SphTimeStep, ZeroViscosityAndGravitySkipTheirLimits) {
    cps::Config config = shipped_dam_break();
    config.sph.viscosity = 0.0f;
    config.external_gravity = {0.0f, 0.0f, 0.0f};
    const cps::SphTimeStepLimits limits = cps::sph_time_step_limits(config, 0.02f);
    EXPECT_TRUE(std::isinf(limits.viscous));
    EXPECT_TRUE(std::isinf(limits.body_force));
    EXPECT_DOUBLE_EQ(limits.smallest(), limits.acoustic);
    EXPECT_STREQ(limits.smallest_name(), "acoustic");
}

// Walls clamp positions, so all of these used to run to the end, finite and
// meaningless, with no error at all.
TEST(SphTimeStep, EveryStepTheWallsWouldHaveHiddenIsRefused) {
    for (const float dt : {0.002f, 0.01f, 0.1f, 1.0f, 100.0f, 1.0e4f}) {
        SCOPED_TRACE(dt);
        cps::Config config = shipped_dam_break();
        config.dt = dt;
        EXPECT_NE(time_step_error(config).find("acoustic limit"), std::string::npos);
    }
}

// The message of the error the check has to raise.
std::string domain_error(const cps::Vec3& low, const cps::Vec3& high, float cell_size) {
    try {
        cps::check_sph_domain_cell_range(low, high, cell_size);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    ADD_FAILURE() << "expected the domain check to throw";
    return "";
}

// Writes one value into one axis of a vector otherwise full of fill.
cps::Vec3 with_axis(int axis, float value, float fill = 0.0f) {
    cps::Vec3 v{fill, fill, fill};
    (axis == 0 ? v.x : axis == 1 ? v.y : v.z) = value;
    return v;
}

TEST(SphDomainCheck, AcceptsTheShippedDamBreak) {
    const cps::Config config = cps::Config::from_json_file(
        std::filesystem::path(CPS_PROJECT_ROOT) / "configs" / "dam_break_50k.json");
    const float h = cps::derive_sph_parameters(config).smoothing_radius;
    EXPECT_NO_THROW(
        cps::check_sph_domain_cell_range(config.sph.domain_min, config.sph.domain_max, h));
}

TEST(SphDomainCheck, RejectsATooLargeDomainOnEachAxisAndNamesIt) {
    const char* const names[] = {"axis x", "axis y", "axis z"};
    for (int axis = 0; axis < 3; ++axis) {
        SCOPED_TRACE(names[axis]);
        const std::string high = domain_error({0.0f, 0.0f, 0.0f}, with_axis(axis, 2.0e5f), 0.1f);
        EXPECT_NE(high.find(names[axis]), std::string::npos) << high;
        EXPECT_NE(high.find("0.1"), std::string::npos) << high;
        EXPECT_NE(high.find("200000"), std::string::npos) << high;
        EXPECT_NE(high.find("raise the smoothing radius or shrink the domain"), std::string::npos)
            << high;

        const std::string low = domain_error(with_axis(axis, -2.0e5f), {1.0f, 1.0f, 1.0f}, 0.1f);
        EXPECT_NE(low.find(names[axis]), std::string::npos) << low;
    }
}

// 2^20 - 2 is exactly representable, as is the float just below it, and a cell
// size of a half or of one divides exactly, so both sides of the limit can be
// hit on the nose. The limit itself is out.
TEST(SphDomainCheck, TheLimitIsExclusiveOnBothSidesOfZero) {
    const float limit = static_cast<float>(cps::SPH_DOMAIN_CELL_LIMIT);
    ASSERT_EQ(static_cast<double>(limit), 1048574.0);
    const float below = std::nextafter(limit, 0.0f);

    for (const float h : {1.0f, 0.5f}) {
        SCOPED_TRACE(h);
        const float at_limit = limit * h;
        const float inside = below * h;
        for (int axis = 0; axis < 3; ++axis) {
            SCOPED_TRACE(axis);
            EXPECT_NO_THROW(cps::check_sph_domain_cell_range({0.0f, 0.0f, 0.0f},
                                                             with_axis(axis, inside), h));
            EXPECT_THROW(cps::check_sph_domain_cell_range({0.0f, 0.0f, 0.0f},
                                                          with_axis(axis, at_limit), h),
                         std::runtime_error);
            EXPECT_NO_THROW(cps::check_sph_domain_cell_range(with_axis(axis, -inside),
                                                             {0.0f, 0.0f, 0.0f}, h));
            EXPECT_THROW(cps::check_sph_domain_cell_range(with_axis(axis, -at_limit),
                                                          {0.0f, 0.0f, 0.0f}, h),
                         std::runtime_error);
        }
    }
}

// Every ratio the check can be shown fails it, whether it came out infinite or
// NaN. A comparison written the other way round would wave a NaN through.
TEST(SphDomainCheck, RejectsANonFiniteRatio) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const cps::Vec3 origin{0.0f, 0.0f, 0.0f};
    const cps::Vec3 unit{1.0f, 1.0f, 1.0f};

    // x / 0 is infinite and 0 / 0 is NaN.
    EXPECT_THROW(cps::check_sph_domain_cell_range(origin, unit, 0.0f), std::runtime_error);
    EXPECT_THROW(cps::check_sph_domain_cell_range(origin, origin, 0.0f), std::runtime_error);
    EXPECT_THROW(cps::check_sph_domain_cell_range(origin, unit, nan), std::runtime_error);
    for (int axis = 0; axis < 3; ++axis) {
        SCOPED_TRACE(axis);
        EXPECT_THROW(cps::check_sph_domain_cell_range(origin, with_axis(axis, inf, 1.0f), 0.1f),
                     std::runtime_error);
        EXPECT_THROW(cps::check_sph_domain_cell_range(origin, with_axis(axis, nan, 1.0f), 0.1f),
                     std::runtime_error);
        EXPECT_THROW(cps::check_sph_domain_cell_range(with_axis(axis, -inf), unit, 0.1f),
                     std::runtime_error);
    }
}

}  // namespace
