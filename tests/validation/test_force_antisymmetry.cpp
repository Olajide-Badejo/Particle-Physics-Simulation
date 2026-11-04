#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "cps/core/particle_system.hpp"
#include "cps/physics/gravity.hpp"
#include "cps/solvers/nbody_solver.hpp"
#include "test_support_gpu.hpp"

namespace {

using cps_test::random_cloud;
using cps_test::reference_accelerations;

constexpr float SOFTENING = 0.02f;
constexpr float GRAVITY = 1.0f;

// Gate: the device force on one hundred randomly placed particles agrees with
// an independent double precision host computation, and the pair interaction
// is antisymmetric.
//
// Antisymmetry is the property that makes momentum conservation possible in
// the first place, so checking it directly is worth more than checking the
// consequence. The host reference is written as literally as the force law can
// be written: no tiling, no reciprocal square root, double precision
// throughout. If the two disagree, the kernel is what is wrong.
//
// Only the device comparison lives here. The antisymmetry, self interaction,
// softening, inverse square and potential checks run on the host and live in
// tests/validation/host/test_force_antisymmetry_host.cpp.

TEST(ForceAntisymmetry, DeviceForcesMatchTheHostReferenceOnAHundredParticles) {
    SKIP_WITHOUT_GPU();
    const int count = 100;
    const cps::HostParticleState state = random_cloud(count, 20260802u);

    cps::ParticleSystem particles(count);
    particles.upload(state, nullptr);
    cps::compute_forces_tiled(particles.x(), particles.y(), particles.z(), particles.mass(),
                              particles.ax(), particles.ay(), particles.az(), count, SOFTENING,
                              GRAVITY, 128, cps::DEFAULT_TILE_SIZE, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> ax(static_cast<std::size_t>(count));
    std::vector<float> ay(static_cast<std::size_t>(count));
    std::vector<float> az(static_cast<std::size_t>(count));
    CUDA_CHECK(cudaMemcpy(ax.data(), particles.ax(), ax.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ay.data(), particles.ay(), ay.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(az.data(), particles.az(), az.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::vector<double> rx, ry, rz;
    reference_accelerations(state, SOFTENING, GRAVITY, rx, ry, rz);

    double worst = 0.0;
    for (int i = 0; i < count; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        const double dx = ax[k] - rx[k];
        const double dy = ay[k] - ry[k];
        const double dz = az[k] - rz[k];
        const double magnitude = std::sqrt(rx[k] * rx[k] + ry[k] * ry[k] + rz[k] * rz[k]);
        ASSERT_GT(magnitude, 0.0) << "index " << i;
        worst = std::max(worst, std::sqrt(dx * dx + dy * dy + dz * dz) / magnitude);
    }
    std::printf("worst relative deviation from the double precision reference: %.3e\n", worst);
    cps_test::record_gate("force_vs_cpu_reference_100_particles", worst, 1.0e-5, "less_than",
                          worst < 1.0e-5);

    // One hundred single precision terms per sum. A few parts in a million is
    // what the arithmetic gives; anything larger is a defect in the kernel.
    EXPECT_LT(worst, 1.0e-5);
}

}  // namespace
