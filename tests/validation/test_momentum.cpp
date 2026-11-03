#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <filesystem>
#include <vector>

#include "cps/core/simulation_engine.hpp"
#include "test_support_gpu.hpp"

namespace {

// Gate: total linear momentum of an isolated Plummer sphere is conserved to
// the tolerance a float32 force sum allows, over a thousand steps at ten
// thousand particles.
//
// Momentum conservation is exact in the mathematics, because the pair
// interaction is antisymmetric and the internal forces cancel. On the device
// it is not exact, for one reason only: each thread accumulates its own force
// sum in single precision and in its own order, so the computed accelerations
// satisfy sum_i m_i a_i = 0 only to rounding. That residual is what this gate
// bounds. Setting an absolute tolerance would be meaningless; the bound has to
// be expressed against the scale of the momentum actually present and against
// what single precision can deliver.

cps::Config plummer_config(cps::NBodyVariant variant, int count, long long steps) {
    cps::Config config;
    config.name = "plummer_momentum";
    config.solver = cps::SolverType::NBody;
    config.variant = variant;
    config.initial_condition = cps::InitialCondition::Plummer;
    config.num_particles = count;
    config.steps = steps;
    config.dt = 1.0e-3f;
    config.softening = 0.01f;
    config.gravitational_constant = 1.0f;
    config.total_mass = 1.0f;
    config.plummer_radius = 1.0f;
    config.seed = 20260802u;
    config.block_size = 256;
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    return config;
}

// Scale of the momentum in the system: the sum of the magnitudes of the
// individual momenta. The total is near zero by construction, so dividing the
// drift by the total would divide by noise.
double momentum_scale(cps::SimulationEngine& engine) {
    cps::HostParticleState state;
    engine.particles().download(state);
    double scale = 0.0;
    for (std::size_t i = 0; i < state.size(); ++i) {
        const double m = state.mass[i];
        scale += m * std::sqrt(static_cast<double>(state.vx[i]) * state.vx[i] +
                               static_cast<double>(state.vy[i]) * state.vy[i] +
                               static_cast<double>(state.vz[i]) * state.vz[i]);
    }
    return scale;
}

TEST(MomentumConservation, PlummerSphereHoldsMomentumOverAThousandSteps) {
    SKIP_WITHOUT_GPU();
    const int count = 10000;
    const long long steps = 1000;
    cps::SimulationEngine engine(plummer_config(cps::NBodyVariant::Tiled, count, steps));
    engine.initialize();

    const double scale = momentum_scale(engine);
    ASSERT_GT(scale, 0.0);

    const cps::Diagnostics initial = engine.diagnostics(false);
    const double p0x = initial.px;
    const double p0y = initial.py;
    const double p0z = initial.pz;

    std::vector<double> times;
    std::vector<double> drifts;
    double worst = 0.0;

    for (long long step = 0; step < steps; ++step) {
        engine.step();
        if ((step + 1) % 20 != 0) {
            continue;
        }
        const cps::Diagnostics row = engine.diagnostics(false);
        const double dx = row.px - p0x;
        const double dy = row.py - p0y;
        const double dz = row.pz - p0z;
        const double drift = std::sqrt(dx * dx + dy * dy + dz * dz) / scale;
        times.push_back(row.time);
        drifts.push_back(drift);
        worst = std::max(worst, drift);
    }

    const std::filesystem::path out = cps_test::results_directory();
    std::filesystem::create_directories(out);
    cps_test::write_series_csv((out / "momentum_drift_plummer.csv").string(),
                               "time,relative_momentum_drift", times, drifts);

    // What single precision allows. Each of the count force sums carries a
    // relative error of order the machine epsilon times the square root of the
    // term count; the residual net force accumulates over the steps taken.
    // This is a bound on that quantity, not a number chosen to make the test
    // pass, and the measured value is printed so the margin is visible.
    const double float_epsilon = FLT_EPSILON;
    const double allowed =
        float_epsilon * std::sqrt(static_cast<double>(count)) * static_cast<double>(steps);

    std::printf("momentum: scale %.6g, worst relative drift %.3e, float32 bound %.3e\n", scale,
                worst, allowed);
    cps_test::record_gate("plummer_momentum_drift_1000_steps", worst, allowed, "less_than",
                          worst < allowed);

    EXPECT_LT(worst, allowed) << "momentum drifted by more than a float32 force sum can explain";
}

TEST(MomentumConservation, HoldsForEveryVariant) {
    SKIP_WITHOUT_GPU();
    const int count = 4096;
    const long long steps = 300;
    const double allowed =
        FLT_EPSILON * std::sqrt(static_cast<double>(count)) * static_cast<double>(steps);

    for (cps::NBodyVariant variant :
         {cps::NBodyVariant::Naive, cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
        cps::SimulationEngine engine(plummer_config(variant, count, steps));
        engine.initialize();
        const double scale = momentum_scale(engine);
        const cps::Diagnostics initial = engine.diagnostics(false);

        for (long long step = 0; step < steps; ++step) {
            engine.step();
        }
        const cps::Diagnostics final_row = engine.diagnostics(false);
        const double dx = final_row.px - initial.px;
        const double dy = final_row.py - initial.py;
        const double dz = final_row.pz - initial.pz;
        const double drift = std::sqrt(dx * dx + dy * dy + dz * dz) / scale;
        std::printf("momentum drift, %s: %.3e\n", cps::to_string(variant).c_str(), drift);
        EXPECT_LT(drift, allowed) << cps::to_string(variant);
    }
}

// A Plummer sphere sampled from the exact distribution function starts in
// virial equilibrium, so 2T + U is close to zero and stays that way. If the
// velocity sampler were wrong the sphere would collapse or expand on the first
// crossing time, and this is the check that would catch it.
TEST(MomentumConservation, ThePlummerSphereStartsNearVirialEquilibrium) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(plummer_config(cps::NBodyVariant::Tiled, 20000, 0));
    engine.initialize();

    const cps::Diagnostics row = engine.diagnostics(true);
    const double virial_ratio = 2.0 * row.kinetic / std::abs(row.potential);
    std::printf("virial ratio 2T/|U| at t = 0: %.4f (kinetic %.6g, potential %.6g)\n", virial_ratio,
                row.kinetic, row.potential);

    // Exactly one for a system in equilibrium; the softening and the finite
    // particle count move it slightly.
    EXPECT_GT(virial_ratio, 0.85);
    EXPECT_LT(virial_ratio, 1.15);
}

// Total energy of the sphere should also be roughly conserved, though the
// tolerance is far looser than the two body case: close encounters between ten
// thousand particles are genuinely stiff and the softened potential is only a
// partial remedy.
TEST(MomentumConservation, PlummerEnergyStaysBounded) {
    SKIP_WITHOUT_GPU();
    const long long steps = 1000;
    cps::SimulationEngine engine(plummer_config(cps::NBodyVariant::Tiled, 10000, steps));
    engine.initialize();

    const cps::Diagnostics initial = engine.diagnostics(true);
    for (long long step = 0; step < steps; ++step) {
        engine.step();
    }
    const cps::Diagnostics final_row = engine.diagnostics(true);

    const double drift = std::abs(final_row.total - initial.total) / std::abs(initial.total);
    std::printf("plummer energy: E0 = %.9g, E1 = %.9g, relative drift %.3e\n", initial.total,
                final_row.total, drift);
    EXPECT_LT(drift, 1.0e-2);
}

}  // namespace
