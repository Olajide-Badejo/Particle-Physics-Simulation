#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/diagnostics.hpp"
#include "core/particle_system.hpp"
#include "test_support.hpp"

namespace {

using cps_test::random_cloud;
using cps_test::reference_potential;
using cps_test::relative_error;

TEST(DeviceReductions, KineticEnergyMatchesTheHostSum) {
    SKIP_WITHOUT_GPU();
    const int count = 12345;
    const cps::HostParticleState state = random_cloud(count, 616u);

    cps::ParticleSystem particles(count);
    particles.upload(state);

    cps::DiagnosticsReducer reducer(count);
    double kinetic = 0.0, px = 0.0, py = 0.0, pz = 0.0;
    reducer.reduce_kinetic_and_momentum(particles, kinetic, px, py, pz);

    double expected = 0.0;
    for (std::size_t i = 0; i < state.size(); ++i) {
        const double m = state.mass[i];
        const double vx = state.vx[i];
        const double vy = state.vy[i];
        const double vz = state.vz[i];
        expected += 0.5 * m * (vx * vx + vy * vy + vz * vz);
    }
    // Both sides accumulate in double, so this is essentially exact. That is
    // the point of reducing in double: the tolerance here is set by the
    // summation order alone, not by the precision of the state.
    EXPECT_LT(relative_error(kinetic, expected), 1.0e-12);
}

TEST(DeviceReductions, MomentumMatchesTheHostSum) {
    SKIP_WITHOUT_GPU();
    const int count = 9999;
    const cps::HostParticleState state = random_cloud(count, 8080u);

    cps::ParticleSystem particles(count);
    particles.upload(state);

    cps::DiagnosticsReducer reducer(count);
    double kinetic = 0.0, px = 0.0, py = 0.0, pz = 0.0;
    reducer.reduce_kinetic_and_momentum(particles, kinetic, px, py, pz);

    double ex = 0.0, ey = 0.0, ez = 0.0;
    for (std::size_t i = 0; i < state.size(); ++i) {
        const double m = state.mass[i];
        ex += m * state.vx[i];
        ey += m * state.vy[i];
        ez += m * state.vz[i];
    }
    EXPECT_LT(relative_error(px, ex), 1.0e-12);
    EXPECT_LT(relative_error(py, ey), 1.0e-12);
    EXPECT_LT(relative_error(pz, ez), 1.0e-12);
}

TEST(DeviceReductions, PotentialEnergyMatchesTheDoublePrecisionReference) {
    SKIP_WITHOUT_GPU();
    const int count = 3000;
    const float softening = 0.05f;
    const float gravity = 1.5f;
    const cps::HostParticleState state = random_cloud(count, 24u);

    cps::ParticleSystem particles(count);
    particles.upload(state);

    cps::DiagnosticsReducer reducer(count);
    const double measured = reducer.reduce_potential(particles, softening, gravity);
    const double expected = reference_potential(state, softening, gravity);

    // The pair terms themselves are formed from single precision positions on
    // both sides, so the agreement is limited by the input precision.
    EXPECT_LT(relative_error(measured, expected), 1.0e-6);
}

// Self interaction has to be excluded. With softening it would not blow up, it
// would quietly add G m^2 / eps per particle, which for the plummer case is a
// large constant offset that would make the total energy look wrong but stable.
TEST(DeviceReductions, PotentialExcludesSelfInteraction) {
    SKIP_WITHOUT_GPU();
    cps::HostParticleState state;
    state.resize(2);
    state.x[0] = 0.0f;
    state.x[1] = 1.0f;
    state.mass[0] = 1.0f;
    state.mass[1] = 1.0f;

    cps::ParticleSystem particles(2);
    particles.upload(state);

    cps::DiagnosticsReducer reducer(2);
    const double measured = reducer.reduce_potential(particles, 0.0f, 1.0f);
    EXPECT_NEAR(measured, -1.0, 1.0e-6);
}

TEST(DeviceReductions, PotentialUsesTheSoftenedDistance) {
    SKIP_WITHOUT_GPU();
    cps::HostParticleState state;
    state.resize(2);
    state.x[1] = 3.0f;
    state.mass[0] = 2.0f;
    state.mass[1] = 5.0f;

    cps::ParticleSystem particles(2);
    particles.upload(state);
    cps::DiagnosticsReducer reducer(2);

    const float softening = 4.0f;
    const double expected = -2.0 * 5.0 / std::sqrt(9.0 + 16.0);
    EXPECT_NEAR(reducer.reduce_potential(particles, softening, 1.0f), expected, 1.0e-6);
}

TEST(DeviceReductions, TheHalfStepCorrectionUsesTheStoredAcceleration) {
    SKIP_WITHOUT_GPU();
    const int count = 1000;
    cps::HostParticleState state = random_cloud(count, 4321u);

    cps::ParticleSystem particles(count);
    particles.upload(state);

    // Give the acceleration array a known value.
    std::vector<float> accel(static_cast<std::size_t>(count), 2.0f);
    CUDA_CHECK(cudaMemcpy(particles.ax(), accel.data(), accel.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(particles.ay(), accel.data(), accel.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(particles.az(), accel.data(), accel.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cps::DiagnosticsReducer reducer(count);
    const float half_dt = 0.25f;
    double kinetic = 0.0, px = 0.0, py = 0.0, pz = 0.0;
    reducer.reduce_kinetic_and_momentum(particles, kinetic, px, py, pz, half_dt);

    double expected = 0.0;
    for (std::size_t i = 0; i < state.size(); ++i) {
        const double m = state.mass[i];
        const double vx = static_cast<double>(state.vx[i]) + half_dt * 2.0;
        const double vy = static_cast<double>(state.vy[i]) + half_dt * 2.0;
        const double vz = static_cast<double>(state.vz[i]) + half_dt * 2.0;
        expected += 0.5 * m * (vx * vx + vy * vy + vz * vz);
    }
    EXPECT_LT(relative_error(kinetic, expected), 1.0e-12);
}

// Two identical reductions must return identical bits. Accumulating block
// partials with atomics would fail this, and would make every diagnostic in a
// resumed run differ slightly from the original.
TEST(DeviceReductions, AreBitwiseRepeatable) {
    SKIP_WITHOUT_GPU();
    const int count = 20000;
    cps::ParticleSystem particles(count);
    particles.upload(random_cloud(count, 13u));

    cps::DiagnosticsReducer reducer(count);
    double k1 = 0.0, x1 = 0.0, y1 = 0.0, z1 = 0.0;
    double k2 = 0.0, x2 = 0.0, y2 = 0.0, z2 = 0.0;
    reducer.reduce_kinetic_and_momentum(particles, k1, x1, y1, z1);
    reducer.reduce_kinetic_and_momentum(particles, k2, x2, y2, z2);
    EXPECT_EQ(k1, k2);
    EXPECT_EQ(x1, x2);
    EXPECT_EQ(y1, y2);
    EXPECT_EQ(z1, z2);

    const double p1 = reducer.reduce_potential(particles, 0.01f, 1.0f);
    const double p2 = reducer.reduce_potential(particles, 0.01f, 1.0f);
    EXPECT_EQ(p1, p2);
}

TEST(ParticleSystem, RoundTripsStateThroughTheDevice) {
    SKIP_WITHOUT_GPU();
    const int count = 5000;
    const cps::HostParticleState state = random_cloud(count, 60u);

    cps::ParticleSystem particles(count);
    particles.upload(state);

    cps::HostParticleState restored;
    particles.download(restored);
    ASSERT_EQ(restored.size(), state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        EXPECT_EQ(restored.x[i], state.x[i]) << "index " << i;
        EXPECT_EQ(restored.vy[i], state.vy[i]) << "index " << i;
        EXPECT_EQ(restored.mass[i], state.mass[i]) << "index " << i;
    }
}

TEST(ParticleSystem, SwapsPositionBuffers) {
    SKIP_WITHOUT_GPU();
    cps::ParticleSystem particles(16);
    const float* before = particles.x();
    float* other = particles.x_next();
    particles.swap_position_buffers();
    EXPECT_EQ(particles.x(), other);
    EXPECT_EQ(particles.x_next(), before);
}

TEST(ParticleSystem, ReportsItsDeviceFootprint) {
    SKIP_WITHOUT_GPU();
    const int count = 1000;
    cps::ParticleSystem particles(count);
    // Thirteen float arrays: three positions, three staged positions, three
    // velocities, three accelerations and the mass.
    EXPECT_EQ(particles.device_bytes(), static_cast<std::size_t>(count) * 13u * sizeof(float));
}

}  // namespace
