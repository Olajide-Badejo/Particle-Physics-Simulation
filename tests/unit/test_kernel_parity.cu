#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "cps/core/particle_system.hpp"
#include "cps/solvers/nbody_solver.hpp"
#include "test_support_gpu.hpp"

namespace {

using cps_test::random_cloud;
using cps_test::reference_accelerations;

struct DeviceForces {
    std::vector<float> ax;
    std::vector<float> ay;
    std::vector<float> az;
};

DeviceForces download_forces(cps::ParticleSystem& particles) {
    const std::size_t n = static_cast<std::size_t>(particles.size());
    DeviceForces forces;
    forces.ax.resize(n);
    forces.ay.resize(n);
    forces.az.resize(n);
    CUDA_CHECK(cudaMemcpy(forces.ax.data(), particles.ax(), n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forces.ay.data(), particles.ay(), n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(forces.az.data(), particles.az(), n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return forces;
}

// Relative error of a vector against a double precision reference, measured
// against the magnitude of the whole acceleration rather than per component,
// so a component that happens to be near zero does not dominate.
double worst_relative_error(const DeviceForces& measured, const std::vector<double>& ax,
                            const std::vector<double>& ay, const std::vector<double>& az) {
    double worst = 0.0;
    for (std::size_t i = 0; i < measured.ax.size(); ++i) {
        const double dx = measured.ax[i] - ax[i];
        const double dy = measured.ay[i] - ay[i];
        const double dz = measured.az[i] - az[i];
        const double error = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double scale = std::sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]);
        if (scale > 0.0) {
            worst = std::max(worst, error / scale);
        }
    }
    return worst;
}

constexpr float SOFTENING = 0.05f;
constexpr float GRAVITY = 1.0f;

TEST(KernelParity, NaiveMatchesTheDoublePrecisionReference) {
    SKIP_WITHOUT_GPU();
    const int count = 2000;
    const cps::HostParticleState state = random_cloud(count, 31337u);

    cps::ParticleSystem particles(count);
    particles.upload(state);
    cps::compute_forces_naive(particles.x(), particles.y(), particles.z(), particles.mass(),
                              particles.ax(), particles.ay(), particles.az(), count, SOFTENING,
                              GRAVITY, 256);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> ax, ay, az;
    reference_accelerations(state, SOFTENING, GRAVITY, ax, ay, az);
    const DeviceForces measured = download_forces(particles);

    // Single precision accumulation over two thousand terms; a few parts in
    // ten thousand is what the arithmetic allows and anything worse is a bug.
    EXPECT_LT(worst_relative_error(measured, ax, ay, az), 1.0e-4);
}

TEST(KernelParity, TiledMatchesTheDoublePrecisionReference) {
    SKIP_WITHOUT_GPU();
    const int count = 2000;
    const cps::HostParticleState state = random_cloud(count, 4242u);

    cps::ParticleSystem particles(count);
    particles.upload(state);
    cps::compute_forces_tiled(particles.x(), particles.y(), particles.z(), particles.mass(),
                              particles.ax(), particles.ay(), particles.az(), count, SOFTENING,
                              GRAVITY, 256);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> ax, ay, az;
    reference_accelerations(state, SOFTENING, GRAVITY, ax, ay, az);
    EXPECT_LT(worst_relative_error(download_forces(particles), ax, ay, az), 1.0e-4);
}

// The three variants are supposed to differ only in how they move memory. If
// tiling changed the answer, the benchmark ladder would be comparing kernels
// that compute different things.
TEST(KernelParity, NaiveAndTiledAgreeWithEachOther) {
    SKIP_WITHOUT_GPU();
    const int count = 1533;  // deliberately not a multiple of any tile width
    const cps::HostParticleState state = random_cloud(count, 909u);

    cps::ParticleSystem particles(count);
    particles.upload(state);

    cps::compute_forces_naive(particles.x(), particles.y(), particles.z(), particles.mass(),
                              particles.ax(), particles.ay(), particles.az(), count, SOFTENING,
                              GRAVITY, 128);
    CUDA_CHECK(cudaDeviceSynchronize());
    const DeviceForces naive = download_forces(particles);

    cps::compute_forces_tiled(particles.x(), particles.y(), particles.z(), particles.mass(),
                              particles.ax(), particles.ay(), particles.az(), count, SOFTENING,
                              GRAVITY, 128);
    CUDA_CHECK(cudaDeviceSynchronize());
    const DeviceForces tiled = download_forces(particles);

    for (int i = 0; i < count; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        const double scale =
            std::sqrt(static_cast<double>(naive.ax[k]) * naive.ax[k] +
                      static_cast<double>(naive.ay[k]) * naive.ay[k] +
                      static_cast<double>(naive.az[k]) * naive.az[k]) +
            1.0e-12;
        EXPECT_LT(std::abs(tiled.ax[k] - naive.ax[k]) / scale, 1.0e-4) << "index " << i;
        EXPECT_LT(std::abs(tiled.ay[k] - naive.ay[k]) / scale, 1.0e-4) << "index " << i;
        EXPECT_LT(std::abs(tiled.az[k] - naive.az[k]) / scale, 1.0e-4) << "index " << i;
    }
}

TEST(KernelParity, EveryTileWidthGivesTheSameAnswer) {
    SKIP_WITHOUT_GPU();
    const int count = 1000;
    const cps::HostParticleState state = random_cloud(count, 5150u);

    cps::ParticleSystem particles(count);
    particles.upload(state);

    std::vector<double> ax, ay, az;
    reference_accelerations(state, SOFTENING, GRAVITY, ax, ay, az);

    for (int tile : {32, 64, 128, 256}) {
        cps::compute_forces_tiled(particles.x(), particles.y(), particles.z(), particles.mass(),
                                  particles.ax(), particles.ay(), particles.az(), count, SOFTENING,
                                  GRAVITY, 256, tile);
        CUDA_CHECK(cudaDeviceSynchronize());
        EXPECT_LT(worst_relative_error(download_forces(particles), ax, ay, az), 1.0e-4)
            << "tile width " << tile;
    }
}

TEST(KernelParity, TiledRejectsAnUnsupportedTileWidth) {
    SKIP_WITHOUT_GPU();
    cps::ParticleSystem particles(64);
    particles.upload(random_cloud(64, 1u));
    EXPECT_THROW(cps::compute_forces_tiled(particles.x(), particles.y(), particles.z(),
                                           particles.mass(), particles.ax(), particles.ay(),
                                           particles.az(), 64, SOFTENING, GRAVITY, 64, 48),
                 std::runtime_error);
}

// Newton's third law in aggregate: the internal forces of an isolated system
// must sum to zero. This is the cheapest possible check that the pair
// interaction is antisymmetric, and it holds for any configuration.
TEST(KernelParity, InternalForcesSumToZero) {
    SKIP_WITHOUT_GPU();
    const int count = 4096;
    const cps::HostParticleState state = random_cloud(count, 777u);

    cps::ParticleSystem particles(count);
    particles.upload(state);
    cps::compute_forces_tiled(particles.x(), particles.y(), particles.z(), particles.mass(),
                              particles.ax(), particles.ay(), particles.az(), count, SOFTENING,
                              GRAVITY, 256);
    CUDA_CHECK(cudaDeviceSynchronize());

    const DeviceForces forces = download_forces(particles);
    double fx = 0.0, fy = 0.0, fz = 0.0, scale = 0.0;
    for (int i = 0; i < count; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        const double m = state.mass[k];
        fx += m * forces.ax[k];
        fy += m * forces.ay[k];
        fz += m * forces.az[k];
        scale += m * std::sqrt(static_cast<double>(forces.ax[k]) * forces.ax[k] +
                               static_cast<double>(forces.ay[k]) * forces.ay[k] +
                               static_cast<double>(forces.az[k]) * forces.az[k]);
    }
    const double residual = std::sqrt(fx * fx + fy * fy + fz * fz) / scale;
    EXPECT_LT(residual, 1.0e-5) << "net internal force is not zero";
}

// The fused kernel runs a leapfrog and the split path runs velocity Verlet.
// They are the same scheme written differently, so their position sequences
// have to track each other to rounding once the fused run is started with the
// half step offset applied.
TEST(KernelParity, FusedTracksTheSplitIntegrator) {
    SKIP_WITHOUT_GPU();
    const int count = 512;
    const int steps = 50;
    const float dt = 1.0e-3f;
    const cps::HostParticleState state = random_cloud(count, 2024u);

    cps::ParticleSystem split(count);
    split.upload(state);
    cps::compute_forces_tiled(split.x(), split.y(), split.z(), split.mass(), split.ax(), split.ay(),
                              split.az(), count, SOFTENING, GRAVITY, 128);

    cps::ParticleSystem fused(count);
    fused.upload(state);
    cps::compute_forces_tiled(fused.x(), fused.y(), fused.z(), fused.mass(), fused.ax(), fused.ay(),
                              fused.az(), count, SOFTENING, GRAVITY, 128);
    cps::offset_velocity_half_step(fused.vx(), fused.vy(), fused.vz(), fused.ax(), fused.ay(),
                                   fused.az(), count, -dt, 128);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int s = 0; s < steps; ++s) {
        cps::integrate_verlet_position(split.x(), split.y(), split.z(), split.vx(), split.vy(),
                                       split.vz(), split.ax(), split.ay(), split.az(), count, dt,
                                       128);
        cps::compute_forces_tiled(split.x(), split.y(), split.z(), split.mass(), split.ax(),
                                  split.ay(), split.az(), count, SOFTENING, GRAVITY, 128);
        cps::integrate_verlet_velocity(split.vx(), split.vy(), split.vz(), split.ax(), split.ay(),
                                       split.az(), count, dt, 128);

        cps::nbody_fused_step(fused.x(), fused.y(), fused.z(), fused.x_next(), fused.y_next(),
                              fused.z_next(), fused.vx(), fused.vy(), fused.vz(), fused.ax(),
                              fused.ay(), fused.az(), fused.mass(), count, SOFTENING, GRAVITY, dt,
                              128);
        fused.swap_position_buffers();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cps::HostParticleState split_out;
    cps::HostParticleState fused_out;
    split.download(split_out);
    fused.download(fused_out);

    double worst = 0.0;
    for (int i = 0; i < count; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        const double dx = static_cast<double>(split_out.x[k]) - fused_out.x[k];
        const double dy = static_cast<double>(split_out.y[k]) - fused_out.y[k];
        const double dz = static_cast<double>(split_out.z[k]) - fused_out.z[k];
        worst = std::max(worst, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    EXPECT_LT(worst, 1.0e-4) << "the fused and split trajectories have diverged";
}

}  // namespace
