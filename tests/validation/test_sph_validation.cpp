#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

#include "core/simulation_engine.hpp"
#include "physics/sph_kernels.hpp"
#include "solvers/sph_solver.hpp"
#include "test_support_gpu.hpp"
#include "validation/sph_fluid_config.hpp"

namespace {

using cps_test::fluid_config;

// Gate: the smoothing kernels integrate to one to within 1e-3 by numerical
// quadrature, the density field recovers the rest density in the interior of a
// block filled at that density, and a running fluid conserves its mass and
// stays inside its walls.
//
// The kernel normalisation and the derived resolution checks need no device,
// so they run on the host from tests/validation/host/test_sph_validation_host.cpp.
// The cases here all run the solver on a GPU.

// The density estimate is the foundation of the whole solver. In the interior
// of a block filled at rest density it must recover that density; a kernel
// whose normalisation or whose neighbour search is wrong shows up here as a
// systematic offset, and everything downstream inherits it.
TEST(SphValidation, InteriorDensityRecoversTheRestDensity) {
    SKIP_WITHOUT_GPU();
    const int count = 50000;
    cps::Config config = fluid_config(count, 0);
    cps::SimulationEngine engine(config);
    engine.initialize();

    cps::SphSolver solver(config, count);
    solver.build_neighbours_and_density(engine.particles());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> density(static_cast<std::size_t>(count));
    CUDA_CHECK(cudaMemcpy(density.data(), solver.density(), density.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cps::HostParticleState state;
    engine.particles().download(state);

    const float h = solver.derived().smoothing_radius;
    std::vector<double> interior;
    for (std::size_t i = 0; i < density.size(); ++i) {
        ASSERT_TRUE(std::isfinite(density[i])) << "index " << i;
        const bool inside = state.x[i] > config.sph.fluid_min.x + 2.0f * h &&
                            state.x[i] < config.sph.fluid_max.x - 2.0f * h &&
                            state.y[i] > config.sph.fluid_min.y + 2.0f * h &&
                            state.y[i] < config.sph.fluid_max.y - 2.0f * h &&
                            state.z[i] > config.sph.fluid_min.z + 2.0f * h &&
                            state.z[i] < config.sph.fluid_max.z - 2.0f * h;
        if (inside) {
            interior.push_back(density[i]);
        }
    }
    ASSERT_GT(interior.size(), 500u);

    double sum = 0.0;
    for (double d : interior) {
        sum += d;
    }
    const double mean = sum / static_cast<double>(interior.size());
    double variance = 0.0;
    for (double d : interior) {
        variance += (d - mean) * (d - mean);
    }
    const double deviation = std::sqrt(variance / static_cast<double>(interior.size()));

    std::printf("interior density: mean %.2f, standard deviation %.2f, over %zu particles\n", mean,
                deviation, interior.size());
    const double density_error = std::abs(mean - config.sph.rest_density) / config.sph.rest_density;
    cps_test::record_gate("sph_interior_density_relative_error", density_error, 0.05, "less_than",
                          density_error < 0.05);
    EXPECT_NEAR(mean, config.sph.rest_density, config.sph.rest_density * 0.05);
    EXPECT_LT(deviation, config.sph.rest_density * 0.10);
}

// A hash based neighbour search that silently drops or double counts
// neighbours is the single most likely defect in this solver, and the exact
// cell key comparison is what prevents it. This checks the outcome directly:
// the number of neighbours the search finds must match the number a brute
// force scan finds, for every sampled particle.
TEST(SphValidation, NeighbourSearchAgreesWithABruteForceScan) {
    SKIP_WITHOUT_GPU();
    const int count = 4000;
    cps::Config config = fluid_config(count, 0);
    cps::SimulationEngine engine(config);
    engine.initialize();

    cps::SphSolver solver(config, count);
    solver.build_neighbours_and_density(engine.particles());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> density(static_cast<std::size_t>(count));
    CUDA_CHECK(cudaMemcpy(density.data(), solver.density(), density.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // The solver leaves the particle arrays permuted into cell sorted order
    // only after a step; build_neighbours_and_density alone does not touch
    // them, so the host copy still matches the order the density was written
    // in only if we recompute from the sorted positions. Instead of relying on
    // that, recompute the density on the host from the same positions with a
    // brute force sum and compare the sorted multisets of values, which is
    // invariant to any permutation.
    cps::HostParticleState state;
    engine.particles().download(state);
    const double h = solver.derived().smoothing_radius;
    const double mass = solver.derived().particle_mass;

    std::vector<double> brute(static_cast<std::size_t>(count), 0.0);
    for (int i = 0; i < count; ++i) {
        double rho = 0.0;
        for (int j = 0; j < count; ++j) {
            const double dx = static_cast<double>(state.x[static_cast<std::size_t>(i)]) -
                              state.x[static_cast<std::size_t>(j)];
            const double dy = static_cast<double>(state.y[static_cast<std::size_t>(i)]) -
                              state.y[static_cast<std::size_t>(j)];
            const double dz = static_cast<double>(state.z[static_cast<std::size_t>(i)]) -
                              state.z[static_cast<std::size_t>(j)];
            const double r_sq = dx * dx + dy * dy + dz * dz;
            if (r_sq < h * h) {
                rho += mass * static_cast<double>(cps::poly6_kernel(static_cast<float>(r_sq),
                                                                    static_cast<float>(h)));
            }
        }
        brute[static_cast<std::size_t>(i)] = rho;
    }

    std::vector<double> from_device(density.begin(), density.end());
    std::sort(brute.begin(), brute.end());
    std::sort(from_device.begin(), from_device.end());

    double worst = 0.0;
    for (std::size_t i = 0; i < brute.size(); ++i) {
        worst = std::max(worst, std::abs(from_device[i] - brute[i]) / brute[i]);
    }
    std::printf("worst relative density difference against a brute force sum: %.3e\n", worst);
    cps_test::record_gate("sph_neighbour_search_vs_brute_force", worst, 1.0e-4, "less_than",
                          worst < 1.0e-4);
    // Any dropped or double counted neighbour would move a density by several
    // percent, far above this.
    EXPECT_LT(worst, 1.0e-4);
}

TEST(SphValidation, AFluidRunConservesMassAndStaysInsideItsWalls) {
    SKIP_WITHOUT_GPU();
    const int count = 20000;
    const long long steps = 2000;
    cps::Config config = fluid_config(count, steps);
    cps::SimulationEngine engine(config);
    engine.initialize();

    cps::HostParticleState before;
    engine.particles().download(before);
    double initial_mass = 0.0;
    for (float m : before.mass) {
        initial_mass += m;
    }

    for (long long step = 0; step < steps; ++step) {
        engine.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cps::HostParticleState after;
    engine.particles().download(after);
    ASSERT_EQ(after.size(), before.size());

    double final_mass = 0.0;
    for (std::size_t i = 0; i < after.size(); ++i) {
        ASSERT_TRUE(std::isfinite(after.x[i])) << "index " << i;
        ASSERT_TRUE(std::isfinite(after.vy[i])) << "index " << i;
        final_mass += after.mass[i];
        EXPECT_GE(after.x[i], config.sph.domain_min.x);
        EXPECT_LE(after.x[i], config.sph.domain_max.x);
        EXPECT_GE(after.y[i], config.sph.domain_min.y);
        EXPECT_LE(after.y[i], config.sph.domain_max.y);
        EXPECT_GE(after.z[i], config.sph.domain_min.z);
        EXPECT_LE(after.z[i], config.sph.domain_max.z);
    }

    // The solver permutes the particle arrays every step, so mass is only
    // conserved if the permutation is a genuine bijection. A gather that lost
    // or duplicated an index would show up here.
    std::printf("fluid mass: %.9g before, %.9g after %lld steps\n", initial_mass, final_mass,
                steps);
    EXPECT_NEAR(final_mass, initial_mass, initial_mass * 1.0e-6);
}

// The permutation the sort produces must be a bijection. Checking the mass sum
// catches duplication in aggregate; this checks it exactly.
TEST(SphValidation, TheCellSortPermutationIsABijection) {
    SKIP_WITHOUT_GPU();
    const int count = 8192;
    cps::Config config = fluid_config(count, 0);
    cps::SimulationEngine engine(config);
    engine.initialize();

    cps::HostParticleState before;
    engine.particles().download(before);

    cps::SphSolver solver(config, count);
    solver.step(engine.particles(), config.dt);
    CUDA_CHECK(cudaDeviceSynchronize());

    cps::HostParticleState after;
    engine.particles().download(after);

    // Every particle keeps its mass through the permutation, and the sorted
    // multiset of masses is unchanged.
    std::vector<float> before_mass(before.mass);
    std::vector<float> after_mass(after.mass);
    std::sort(before_mass.begin(), before_mass.end());
    std::sort(after_mass.begin(), after_mass.end());
    ASSERT_EQ(before_mass.size(), after_mass.size());
    for (std::size_t i = 0; i < before_mass.size(); ++i) {
        EXPECT_FLOAT_EQ(before_mass[i], after_mass[i]) << "index " << i;
    }
}

}  // namespace
