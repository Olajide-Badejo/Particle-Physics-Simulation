#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "cps/core/initial_conditions.hpp"
#include "cps/core/simulation_engine.hpp"
#include "test_support_gpu.hpp"

namespace fs = std::filesystem;

namespace {

cps::Config fluid_config() {
    cps::Config config;
    config.solver = cps::SolverType::Sph;
    config.initial_condition = cps::InitialCondition::DamBreak;
    config.num_particles = 8000;
    config.dt = 1.0e-4f;
    config.block_size = 128;
    config.seed = 20260802u;
    config.external_gravity = {0.0f, -9.81f, 0.0f};
    config.sph.domain_min = {0.0f, 0.0f, 0.0f};
    config.sph.domain_max = {0.8f, 0.5f, 0.4f};
    config.sph.fluid_min = {0.0f, 0.0f, 0.0f};
    config.sph.fluid_max = {0.3f, 0.4f, 0.4f};
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    config.snapshot_interval = 0;
    return config;
}

// Every value from 0 to N - 1, once each. Slot i of the result holds the slot that
// id i is in, so a comparison across steps can look a particle up by its id.
std::vector<std::size_t> slots_by_id(const cps::HostParticleState& state) {
    const std::size_t none = state.size();
    std::vector<std::size_t> slots(state.size(), none);
    for (std::size_t i = 0; i < state.size(); ++i) {
        EXPECT_LT(state.id[i], state.size()) << "slot " << i;
        if (state.id[i] >= state.size()) {
            return {};
        }
        EXPECT_EQ(slots[state.id[i]], none) << "id " << state.id[i] << " appears twice";
        slots[state.id[i]] = i;
    }
    return slots;
}

// What a step can move a particle by, along one axis. The fluid step is
//
//   v' = v + a dt,   x' = x + v' dt
//
// so the bound is |v'| dt, except that a particle crossing a wall is put back on it
// and has its normal velocity reversed and damped. It then moved at most the
// undamped velocity's distance while the state records the damped one.
double axis_speed(float position, float velocity, float low, float high, float damping) {
    const bool on_a_wall = position <= low || position >= high;
    const double speed = std::abs(static_cast<double>(velocity));
    return on_a_wall ? speed / static_cast<double>(damping) : speed;
}

// This is the test that catches a wrong permutation. A particle that a broken gather
// or write-back mislabels turns up, under its id, at some other particle's position,
// a distance of the order of the particle spacing, far more than a step can move it
// in this early flow.
TEST(ParticleIds, EachIdMovesContinuouslyThroughTheSort) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = fluid_config();
    cps::SimulationEngine engine(config);
    engine.initialize();

    cps::HostParticleState previous;
    engine.particles().download(previous, engine.stream());
    const std::vector<std::size_t> initial_slots = slots_by_id(previous);
    ASSERT_FALSE(initial_slots.empty());
    for (std::size_t i = 0; i < previous.size(); ++i) {
        ASSERT_EQ(previous.id[i], i) << "ids start in the initial state's order";
    }

    // Rounding in x + v dt is an ulp or two of x, far below the spacing.
    constexpr double ROUNDING = 1.0e-6;
    const double dt = static_cast<double>(config.dt);
    const cps::SphConfig& sph = config.sph;
    std::size_t slots_moved = 0;
    for (int step = 1; step <= 60; ++step) {
        engine.step();
        cps::HostParticleState current;
        engine.particles().download(current, engine.stream());
        const std::vector<std::size_t> now = slots_by_id(current);
        const std::vector<std::size_t> before = slots_by_id(previous);
        ASSERT_FALSE(now.empty()) << "step " << step;
        ASSERT_FALSE(before.empty()) << "step " << step;

        for (std::size_t id = 0; id < now.size(); ++id) {
            const std::size_t a = before[id];
            const std::size_t b = now[id];
            slots_moved += a != b ? 1u : 0u;
            const double dx = static_cast<double>(current.x[b]) - static_cast<double>(previous.x[a]);
            const double dy = static_cast<double>(current.y[b]) - static_cast<double>(previous.y[a]);
            const double dz = static_cast<double>(current.z[b]) - static_cast<double>(previous.z[a]);
            const double sx = axis_speed(current.x[b], current.vx[b], sph.domain_min.x,
                                         sph.domain_max.x, sph.boundary_damping);
            const double sy = axis_speed(current.y[b], current.vy[b], sph.domain_min.y,
                                         sph.domain_max.y, sph.boundary_damping);
            const double sz = axis_speed(current.z[b], current.vz[b], sph.domain_min.z,
                                         sph.domain_max.z, sph.boundary_damping);
            const double moved = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double allowed = std::sqrt(sx * sx + sy * sy + sz * sz) * dt + ROUNDING;
            ASSERT_LE(moved, allowed) << "id " << id << " jumped at step " << step << " from slot "
                                      << a << " to slot " << b;
        }
        previous = std::move(current);
    }
    // Without this the test would pass on a sort that never moved anything.
    EXPECT_GT(slots_moved, previous.size()) << "the sort hardly moved a particle";
}

// The ids and the rest of a particle stay together through a resume, which the
// bit identity tests in test_snapshot_resume.cpp also compare.
TEST(ParticleIds, FluidIdsSurviveASnapshotAndAreNotTheFileOrder) {
    SKIP_WITHOUT_GPU();
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path directory = fs::temp_directory_path() / ("cps_ids_" + std::to_string(stamp));

    cps::Config config = fluid_config();
    config.snapshot_dir = directory.string();
    {
        cps::SimulationEngine engine(config);
        engine.initialize();
        for (int i = 0; i < 10; ++i) {
            engine.step();
        }
        engine.write_snapshot_now();
    }
    const cps::Snapshot stored = cps::read_snapshot(cps::latest_snapshot(directory));
    EXPECT_FALSE(slots_by_id(stored.state).empty());
    std::size_t displaced = 0;
    for (std::size_t i = 0; i < stored.state.size(); ++i) {
        displaced += stored.state.id[i] != i ? 1u : 0u;
    }
    EXPECT_GT(displaced, 0u) << "a fluid's sort should have moved particles by now";

    cps::SimulationEngine resumed(config);
    resumed.resume();
    cps::HostParticleState state;
    resumed.particles().download(state, resumed.stream());
    EXPECT_EQ(state.id, stored.state.id);

    std::error_code ignored;
    fs::remove_all(directory, ignored);
}

// N-body never permutes: fused swaps position buffers and nothing else. Nothing but
// the stored ids can show that.
TEST(ParticleIds, NBodyIdsStayInOrderUnderEveryVariant) {
    SKIP_WITHOUT_GPU();
    for (const cps::NBodyVariant variant :
         {cps::NBodyVariant::Naive, cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
        cps::Config config;
        config.variant = variant;
        config.num_particles = 256;
        config.initial_condition = cps::InitialCondition::Plummer;
        config.softening = 0.05f;
        config.block_size = 128;
        config.diagnostics_interval = 0;
        config.diagnostics_path.clear();
        config.snapshot_interval = 0;
        cps::SimulationEngine engine(config);
        engine.initialize();
        for (int i = 0; i < 7; ++i) {
            engine.step();
        }
        cps::HostParticleState state;
        engine.particles().download(state, engine.stream());
        for (std::size_t i = 0; i < state.size(); ++i) {
            ASSERT_EQ(state.id[i], i);
        }
    }
}

}  // namespace
