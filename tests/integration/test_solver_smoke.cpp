#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "cps/core/simulation_engine.hpp"
#include "cps/solvers/sph_solver.hpp"
#include "test_support_gpu.hpp"

namespace {

cps::Config nbody_config(cps::NBodyVariant variant, int count, long long steps) {
    cps::Config config;
    config.name = "smoke_" + cps::to_string(variant);
    config.solver = cps::SolverType::NBody;
    config.variant = variant;
    config.initial_condition = cps::InitialCondition::Plummer;
    config.num_particles = count;
    config.steps = steps;
    config.dt = 1.0e-3f;
    config.softening = 0.05f;
    config.gravitational_constant = 1.0f;
    config.block_size = 128;
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    return config;
}

cps::Config sph_config(int count, long long steps) {
    cps::Config config;
    config.name = "smoke_sph";
    config.solver = cps::SolverType::Sph;
    config.initial_condition = cps::InitialCondition::DamBreak;
    config.num_particles = count;
    config.steps = steps;
    config.dt = 1.0e-4f;
    config.block_size = 128;
    config.external_gravity = {0.0f, -9.81f, 0.0f};
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    config.sph.domain_min = {0.0f, 0.0f, 0.0f};
    config.sph.domain_max = {0.8f, 0.5f, 0.4f};
    config.sph.fluid_min = {0.0f, 0.0f, 0.0f};
    config.sph.fluid_max = {0.3f, 0.4f, 0.4f};
    return config;
}

void expect_all_finite(const cps::HostParticleState& state, const std::string& label) {
    for (std::size_t i = 0; i < state.size(); ++i) {
        ASSERT_TRUE(std::isfinite(state.x[i])) << label << " position x, index " << i;
        ASSERT_TRUE(std::isfinite(state.y[i])) << label << " position y, index " << i;
        ASSERT_TRUE(std::isfinite(state.z[i])) << label << " position z, index " << i;
        ASSERT_TRUE(std::isfinite(state.vx[i])) << label << " velocity x, index " << i;
        ASSERT_TRUE(std::isfinite(state.vy[i])) << label << " velocity y, index " << i;
        ASSERT_TRUE(std::isfinite(state.vz[i])) << label << " velocity z, index " << i;
    }
}

TEST(SolverSmoke, EveryNBodyVariantRunsAHundredStepsCleanly) {
    SKIP_WITHOUT_GPU();
    for (cps::NBodyVariant variant :
         {cps::NBodyVariant::Naive, cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
        const std::string label = cps::to_string(variant);
        cps::SimulationEngine engine(nbody_config(variant, 256, 100));
        engine.initialize();
        for (int step = 0; step < 100; ++step) {
            engine.step();
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << label;
        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << label;

        cps::HostParticleState state;
        engine.particles().download(state);
        expect_all_finite(state, label);
        EXPECT_EQ(engine.current_step(), 100);
    }
}

TEST(SolverSmoke, TheFluidSolverRunsAHundredStepsCleanly) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(sph_config(256, 100));
    engine.initialize();
    for (int step = 0; step < 100; ++step) {
        engine.step();
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    cps::HostParticleState state;
    engine.particles().download(state);
    expect_all_finite(state, "sph");
}

// Walls are not optional: a particle that leaves the domain lands in a cell
// whose coordinate is far outside the occupied range, and the neighbour search
// then returns nothing, so it never comes back.
TEST(SolverSmoke, TheFluidStaysInsideItsDomain) {
    SKIP_WITHOUT_GPU();
    cps::Config config = sph_config(4096, 400);
    cps::SimulationEngine engine(config);
    engine.initialize();
    for (int step = 0; step < config.steps; ++step) {
        engine.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cps::HostParticleState state;
    engine.particles().download(state);
    for (std::size_t i = 0; i < state.size(); ++i) {
        EXPECT_GE(state.x[i], config.sph.domain_min.x) << "index " << i;
        EXPECT_LE(state.x[i], config.sph.domain_max.x) << "index " << i;
        EXPECT_GE(state.y[i], config.sph.domain_min.y) << "index " << i;
        EXPECT_LE(state.y[i], config.sph.domain_max.y) << "index " << i;
        EXPECT_GE(state.z[i], config.sph.domain_min.z) << "index " << i;
        EXPECT_LE(state.z[i], config.sph.domain_max.z) << "index " << i;
    }
}

// The column has to actually collapse. A fluid that simply sits there would
// pass every finiteness check ever written.
TEST(SolverSmoke, TheDamBreakColumnCollapsesAndSpreads) {
    SKIP_WITHOUT_GPU();
    cps::Config config = sph_config(8192, 1500);
    cps::SimulationEngine engine(config);
    engine.initialize();

    cps::HostParticleState before;
    engine.particles().download(before);
    float initial_reach = 0.0f;
    float initial_height = 0.0f;
    for (std::size_t i = 0; i < before.size(); ++i) {
        initial_reach = std::max(initial_reach, before.x[i]);
        initial_height = std::max(initial_height, before.y[i]);
    }

    for (int step = 0; step < config.steps; ++step) {
        engine.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cps::HostParticleState after;
    engine.particles().download(after);
    float final_reach = 0.0f;
    double mean_height = 0.0;
    double initial_mean_height = 0.0;
    for (std::size_t i = 0; i < after.size(); ++i) {
        final_reach = std::max(final_reach, after.x[i]);
        mean_height += after.y[i];
        initial_mean_height += before.y[i];
    }
    mean_height /= static_cast<double>(after.size());
    initial_mean_height /= static_cast<double>(before.size());

    EXPECT_GT(final_reach, initial_reach) << "the column did not spread along the floor";
    EXPECT_LT(mean_height, initial_mean_height) << "the column did not fall";
    EXPECT_GT(initial_height, 0.0f);
}

// The density estimate is what everything else in the fluid solver is built
// on. In the interior of a block filled at rest density it must recover that
// density; near the free surface it legitimately falls short because there is
// no fluid on the outside.
TEST(SolverSmoke, InteriorDensityRecoversTheRestDensity) {
    SKIP_WITHOUT_GPU();
    cps::Config config = sph_config(30000, 0);
    cps::SimulationEngine engine(config);
    engine.initialize();

    cps::SphSolver solver(config, config.num_particles);
    solver.build_neighbours_and_density(engine.particles());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> density(static_cast<std::size_t>(config.num_particles));
    CUDA_CHECK(cudaMemcpy(density.data(), solver.density(), density.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // The density is in cell sorted order and the positions are in particle
    // order, so sorted_index is what pairs the two.
    std::vector<uint32_t> sorted_index(static_cast<std::size_t>(config.num_particles));
    CUDA_CHECK(cudaMemcpy(sorted_index.data(), solver.sorted_index(),
                          sorted_index.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cps::HostParticleState state;
    engine.particles().download(state);

    const float h = solver.derived().smoothing_radius;
    double sum = 0.0;
    int interior = 0;
    for (std::size_t k = 0; k < density.size(); ++k) {
        const std::size_t i = sorted_index[k];
        const bool inside = state.x[i] > config.sph.fluid_min.x + 2.0f * h &&
                            state.x[i] < config.sph.fluid_max.x - 2.0f * h &&
                            state.y[i] > config.sph.fluid_min.y + 2.0f * h &&
                            state.y[i] < config.sph.fluid_max.y - 2.0f * h &&
                            state.z[i] > config.sph.fluid_min.z + 2.0f * h &&
                            state.z[i] < config.sph.fluid_max.z - 2.0f * h;
        if (inside) {
            sum += density[k];
            ++interior;
        }
        ASSERT_TRUE(std::isfinite(density[k])) << "sorted index " << k;
        ASSERT_GT(density[k], 0.0f) << "sorted index " << k;
    }
    ASSERT_GT(interior, 100) << "not enough interior particles to judge the density";

    const double mean = sum / interior;
    EXPECT_NEAR(mean, config.sph.rest_density, config.sph.rest_density * 0.1)
        << "interior density is " << mean << " against a rest density of "
        << config.sph.rest_density;
}

// The hash is meant to spread cells across the table. This records what the
// collision rate actually is on the shipped configuration rather than assuming
// it is small.
TEST(SolverSmoke, TheHashTableCollisionRateIsSmall) {
    SKIP_WITHOUT_GPU();
    cps::Config config = sph_config(30000, 0);
    cps::SimulationEngine engine(config);
    engine.initialize();

    cps::SphSolver solver(config, config.num_particles);
    const int collisions = solver.count_bucket_collisions(engine.particles());
    const double rate = static_cast<double>(collisions) / config.num_particles;
    EXPECT_LT(rate, 0.05) << collisions << " of " << config.num_particles
                          << " particles sit next to a foreign cell in their bucket";
}

TEST(SolverSmoke, TheEngineRefusesToStepBeforeInitialisation) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(nbody_config(cps::NBodyVariant::Tiled, 128, 10));
    EXPECT_THROW(engine.step(), std::runtime_error);
}

}  // namespace
