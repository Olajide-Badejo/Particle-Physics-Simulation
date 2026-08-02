#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "core/simulation_engine.hpp"
#include "test_support.hpp"

namespace {

// Gate: a two body circular orbit must conserve total energy to better than
// 0.1 percent over ten thousand velocity Verlet steps.
//
// This is the sharpest single statement that can be made about an integrator.
// Velocity Verlet is symplectic, so its energy error is bounded and oscillates
// with the orbit rather than accumulating; a scheme that merely looks accurate,
// such as fourth order Runge Kutta, would show a steady downward drift here no
// matter how small the step. A failure of this gate is a stop the line defect.
//
// The gate runs against the strict library, which is compiled with fast math
// off. That is not a formality: the tolerance here is 1e-3 relative, and the
// approximate reciprocal square root does not leave that much headroom.

constexpr double ENERGY_DRIFT_LIMIT = 1.0e-3;  // 0.1 percent

cps::Config two_body_config(cps::NBodyVariant variant, long long steps) {
    cps::Config config;
    config.name = "two_body_" + cps::to_string(variant);
    config.solver = cps::SolverType::NBody;
    config.variant = variant;
    config.initial_condition = cps::InitialCondition::TwoBody;
    config.num_particles = 2;
    config.steps = steps;
    config.dt = 1.0e-3f;
    config.softening = 0.01f;
    config.gravitational_constant = 1.0f;
    config.total_mass = 2.0f;
    config.two_body_separation = 1.0f;
    config.block_size = 32;
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    return config;
}

struct DriftResult {
    double initial_energy = 0.0;
    double worst_relative_drift = 0.0;
    std::vector<double> time;
    std::vector<double> relative_drift;
};

DriftResult measure_drift(cps::NBodyVariant variant, long long steps, int sample_every) {
    cps::SimulationEngine engine(two_body_config(variant, steps));
    engine.initialize();

    DriftResult result;
    const cps::Diagnostics first = engine.diagnostics(true);
    result.initial_energy = first.total;
    result.time.push_back(first.time);
    result.relative_drift.push_back(0.0);

    for (long long step = 0; step < steps; ++step) {
        engine.step();
        if ((step + 1) % sample_every == 0) {
            const cps::Diagnostics row = engine.diagnostics(true);
            const double drift =
                std::abs(row.total - result.initial_energy) / std::abs(result.initial_energy);
            result.time.push_back(row.time);
            result.relative_drift.push_back(drift);
            result.worst_relative_drift = std::max(result.worst_relative_drift, drift);
        }
    }
    return result;
}

TEST(EnergyDrift, TwoBodyOrbitHoldsEnergyOverTenThousandSteps) {
    SKIP_WITHOUT_GPU();
    const long long steps = 10000;
    const DriftResult result = measure_drift(cps::NBodyVariant::Tiled, steps, 10);

    ASSERT_LT(result.initial_energy, 0.0) << "a bound orbit must have negative total energy";

    const std::filesystem::path out =
        std::filesystem::path(CPS_PROJECT_ROOT) / "experiments" / "results";
    std::filesystem::create_directories(out);
    cps_test::write_series_csv((out / "energy_drift_two_body.csv").string(),
                               "time,relative_energy_drift", result.time, result.relative_drift);

    std::printf("two body energy: E0 = %.12g, worst relative drift over %lld steps = %.3e\n",
                result.initial_energy, steps, result.worst_relative_drift);
    cps_test::record_gate("two_body_energy_drift_10000_steps", result.worst_relative_drift,
                          ENERGY_DRIFT_LIMIT, "less_than",
                          result.worst_relative_drift < ENERGY_DRIFT_LIMIT);

    EXPECT_LT(result.worst_relative_drift, ENERGY_DRIFT_LIMIT)
        << "energy drifted by " << result.worst_relative_drift * 100.0 << " percent over " << steps
        << " steps, against a limit of " << ENERGY_DRIFT_LIMIT * 100.0 << " percent";
}

TEST(EnergyDrift, TheFusedVariantHoldsEnergyJustAsWell) {
    SKIP_WITHOUT_GPU();
    const long long steps = 10000;
    const DriftResult result = measure_drift(cps::NBodyVariant::Fused, steps, 10);
    std::printf("two body energy, fused: worst relative drift = %.3e\n",
                result.worst_relative_drift);
    EXPECT_LT(result.worst_relative_drift, ENERGY_DRIFT_LIMIT);
}

TEST(EnergyDrift, TheNaiveVariantHoldsEnergyJustAsWell) {
    SKIP_WITHOUT_GPU();
    const long long steps = 10000;
    const DriftResult result = measure_drift(cps::NBodyVariant::Naive, steps, 10);
    std::printf("two body energy, naive: worst relative drift = %.3e\n",
                result.worst_relative_drift);
    EXPECT_LT(result.worst_relative_drift, ENERGY_DRIFT_LIMIT);
}

// The distinguishing property of a symplectic integrator is that its energy
// error oscillates rather than accumulating. If the error in the second half
// of the run were systematically larger than in the first, the scheme would be
// dissipative and the gate above would only be passing because the run was
// short.
TEST(EnergyDrift, TheEnergyErrorOscillatesRatherThanAccumulating) {
    SKIP_WITHOUT_GPU();
    const long long steps = 10000;
    const DriftResult result = measure_drift(cps::NBodyVariant::Tiled, steps, 10);
    ASSERT_GT(result.relative_drift.size(), 100u);

    const std::size_t half = result.relative_drift.size() / 2;
    double first_half = 0.0;
    double second_half = 0.0;
    for (std::size_t i = 0; i < half; ++i) {
        first_half = std::max(first_half, result.relative_drift[i]);
    }
    for (std::size_t i = half; i < result.relative_drift.size(); ++i) {
        second_half = std::max(second_half, result.relative_drift[i]);
    }
    std::printf("worst drift: first half %.3e, second half %.3e\n", first_half, second_half);

    // A dissipative scheme would grow without bound. Allowing a factor of
    // three leaves room for the bounded oscillation to be sampled at a
    // different phase in each half.
    EXPECT_LT(second_half, std::max(first_half * 3.0, 1.0e-9));
}

// The orbit must actually be an orbit. If the initial speed were wrong the
// pair would spiral or fly apart, and a system that flies apart conserves
// energy beautifully while telling you nothing.
TEST(EnergyDrift, TheOrbitStaysCircular) {
    SKIP_WITHOUT_GPU();
    cps::Config config = two_body_config(cps::NBodyVariant::Tiled, 10000);
    cps::SimulationEngine engine(config);
    engine.initialize();

    double smallest = 1.0e30;
    double largest = 0.0;
    for (long long step = 0; step < config.steps; ++step) {
        engine.step();
        if (step % 50 != 0) {
            continue;
        }
        cps::HostParticleState state;
        engine.particles().download(state);
        const double dx = static_cast<double>(state.x[1]) - state.x[0];
        const double dy = static_cast<double>(state.y[1]) - state.y[0];
        const double dz = static_cast<double>(state.z[1]) - state.z[0];
        const double separation = std::sqrt(dx * dx + dy * dy + dz * dz);
        smallest = std::min(smallest, separation);
        largest = std::max(largest, separation);
    }

    const double eccentricity_proxy = (largest - smallest) / (largest + smallest);
    std::printf("separation stays between %.6f and %.6f, out of round by %.3e\n", smallest, largest,
                eccentricity_proxy);
    cps_test::record_gate("two_body_orbit_out_of_round", eccentricity_proxy, 1.0e-3, "less_than",
                          eccentricity_proxy < 1.0e-3);
    EXPECT_LT(eccentricity_proxy, 1.0e-3) << "the orbit is not circular";
}

}  // namespace
