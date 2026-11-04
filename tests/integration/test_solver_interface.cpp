#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cps/core/initial_conditions.hpp"
#include "cps/core/particle_system.hpp"
#include "cps/core/simulation_engine.hpp"
#include "cps/solvers/nbody_solver.hpp"
#include "test_support_gpu.hpp"

namespace fs = std::filesystem;

namespace {

using Calls = std::vector<std::string>;

// Particles coast: the acceleration array is left alone, so a step is
// x <- x + v dt through the real integrator launcher on the engine's stream. What
// it adds is a record of every hook the engine called, which outlives the solver,
// because the engine owns it.
class RecordingSolver final : public cps::Solver {
   public:
    struct Options {
        cps::VelocityConvention convention = cps::VelocityConvention::Synchronised;
        bool potential = false;
        // Written to every acceleration component by initialize and on_resume, so a
        // test can see whether the engine's convention shift ran after them.
        float acceleration = 0.0f;
        // What check_device_errors reports, if anything.
        std::string device_error;
    };

    RecordingSolver(Calls& calls, float dt, int block_size, Options options)
        : calls_(calls), dt_(dt), block_size_(block_size), options_(std::move(options)) {}

    void initialize(cps::ParticleSystem& particles, cudaStream_t stream) override {
        calls_.emplace_back("initialize");
        fill_acceleration(particles, stream);
    }

    void on_resume(cps::ParticleSystem& particles, cudaStream_t stream) override {
        calls_.emplace_back("on_resume");
        fill_acceleration(particles, stream);
    }

    void step(cps::ParticleSystem& particles, cudaStream_t stream) override {
        calls_.emplace_back("step");
        cps::integrate_verlet_position(particles.x(), particles.y(), particles.z(), particles.vx(),
                                       particles.vy(), particles.vz(), particles.ax(),
                                       particles.ay(), particles.az(), particles.size(), dt_,
                                       block_size_, stream);
    }

    cps::VelocityConvention velocity_convention() const override { return options_.convention; }

    void prepare_diagnostics(cps::ParticleSystem&, cudaStream_t) override {
        calls_.emplace_back("prepare_diagnostics");
    }

    bool has_potential_energy() const override { return options_.potential; }

    void check_device_errors(long long step, cudaStream_t) const override {
        calls_.emplace_back("check_device_errors");
        if (!options_.device_error.empty()) {
            throw std::runtime_error(options_.device_error + " at step " + std::to_string(step));
        }
    }

    std::string startup_description() const override { return "  recording     : yes\n"; }

   private:
    void fill_acceleration(cps::ParticleSystem& particles, cudaStream_t stream) {
        const std::size_t n = static_cast<std::size_t>(particles.size());
        acceleration_.assign(n, options_.acceleration);
        cps::copy_to_device_async(particles.ax(), acceleration_.data(), n, stream);
        cps::copy_to_device_async(particles.ay(), acceleration_.data(), n, stream);
        cps::copy_to_device_async(particles.az(), acceleration_.data(), n, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    Calls& calls_;
    float dt_;
    int block_size_;
    Options options_;
    std::vector<float> acceleration_;
};

class SolverInterfaceTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = fs::temp_directory_path() / ("cps_solver_" + std::to_string(stamp));
        fs::create_directories(directory_);
    }

    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(directory_, ignored);
    }

    cps::Config config() const {
        cps::Config config;
        config.solver = cps::SolverType::NBody;
        config.variant = cps::NBodyVariant::Tiled;
        config.initial_condition = cps::InitialCondition::Plummer;
        config.num_particles = 300;
        config.dt = 1.0e-3f;
        config.softening = 0.05f;
        config.block_size = 128;
        config.snapshot_dir = (directory_ / "snapshots").string();
        config.snapshot_interval = 0;
        config.diagnostics_interval = 0;
        config.diagnostics_path.clear();
        return config;
    }

    std::unique_ptr<cps::Solver> recording(const cps::Config& config, Calls& calls,
                                           RecordingSolver::Options options = {}) const {
        return std::make_unique<RecordingSolver>(calls, config.dt, config.block_size,
                                                 std::move(options));
    }

    fs::path directory_;
};

// Takes what was recorded so far, so each test can assert one stretch at a time.
Calls take(Calls& calls) {
    Calls taken;
    std::swap(taken, calls);
    return taken;
}

void expect_same_bits(const std::vector<float>& a, const std::vector<float>& b,
                      const std::string& label) {
    ASSERT_EQ(a.size(), b.size()) << label;
    for (std::size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(std::bit_cast<uint32_t>(a[i]), std::bit_cast<uint32_t>(b[i]))
            << label << " at index " << i;
    }
}

void expect_same_state(const cps::HostParticleState& a, const cps::HostParticleState& b,
                       const std::string& label) {
    expect_same_bits(a.x, b.x, label + " x");
    expect_same_bits(a.y, b.y, label + " y");
    expect_same_bits(a.z, b.z, label + " z");
    expect_same_bits(a.vx, b.vx, label + " vx");
    expect_same_bits(a.vy, b.vy, label + " vy");
    expect_same_bits(a.vz, b.vz, label + " vz");
    expect_same_bits(a.mass, b.mass, label + " mass");
}

TEST_F(SolverInterfaceTest, InitializeCallsItsHookOnceAndNothingElse) {
    SKIP_WITHOUT_GPU();
    Calls calls;
    const cps::Config c = config();
    cps::SimulationEngine engine(c, recording(c, calls));
    EXPECT_TRUE(calls.empty()) << "construction runs no hook";
    engine.initialize();
    EXPECT_EQ(take(calls), Calls{"initialize"});
    EXPECT_EQ(engine.current_step(), 0);
}

TEST_F(SolverInterfaceTest, TheEngineRefusesToStepBeforeInitialisation) {
    SKIP_WITHOUT_GPU();
    Calls calls;
    const cps::Config c = config();
    cps::SimulationEngine engine(c, recording(c, calls));
    EXPECT_THROW(engine.step(), std::runtime_error);
    EXPECT_TRUE(calls.empty());
}

TEST_F(SolverInterfaceTest, ANullSolverIsRefusedByName) {
    SKIP_WITHOUT_GPU();
    try {
        cps::SimulationEngine engine(config(), nullptr);
        FAIL() << "a null solver was accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("null solver"), std::string::npos) << error.what();
    }
}

// The engine moves the step count and the device work happens on its own stream,
// so draining that stream is what makes the coasting visible.
TEST_F(SolverInterfaceTest, StepRunsTheSolverOnTheEnginesStream) {
    SKIP_WITHOUT_GPU();
    Calls calls;
    const cps::Config c = config();
    cps::SimulationEngine engine(c, recording(c, calls));
    engine.initialize();
    take(calls);
    cps::HostParticleState before;
    engine.particles().download(before, engine.stream());

    engine.step();
    engine.step();
    engine.step();
    EXPECT_EQ(take(calls), (Calls{"step", "step", "step"}));
    EXPECT_EQ(engine.current_step(), 3);

    cps::HostParticleState after;
    engine.particles().download(after, engine.stream());
    for (std::size_t i = 0; i < after.size(); ++i) {
        ASSERT_NEAR(after.x[i], before.x[i] + 3.0f * c.dt * before.vx[i], 1.0e-5f) << i;
        ASSERT_EQ(std::bit_cast<uint32_t>(after.vx[i]), std::bit_cast<uint32_t>(before.vx[i])) << i;
    }
}

// A diagnostic checks the device first, so a blown-up run does no more work, and
// then gives the solver its chance to prepare before anything is reduced.
TEST_F(SolverInterfaceTest, ADiagnosticChecksThenPreparesThenReduces) {
    SKIP_WITHOUT_GPU();
    Calls calls;
    const cps::Config c = config();
    cps::SimulationEngine engine(c, recording(c, calls));
    engine.initialize();
    engine.step();
    take(calls);

    const cps::Diagnostics row = engine.diagnostics(true);
    EXPECT_EQ(take(calls), (Calls{"check_device_errors", "prepare_diagnostics"}));
    EXPECT_EQ(row.step, 1);
    EXPECT_GT(row.kinetic, 0.0);
    EXPECT_EQ(row.potential, 0.0) << "this solver has no potential energy pass";
}

TEST_F(SolverInterfaceTest, ThePotentialPassFollowsTheSolver) {
    SKIP_WITHOUT_GPU();
    Calls calls;
    const cps::Config c = config();
    RecordingSolver::Options options;
    options.potential = true;
    cps::SimulationEngine engine(c, recording(c, calls, options));
    engine.initialize();
    EXPECT_LT(engine.diagnostics(true).potential, 0.0) << "a bound cloud has negative potential";
    EXPECT_EQ(engine.diagnostics(false).potential, 0.0) << "the caller can still turn it off";
}

TEST_F(SolverInterfaceTest, ASnapshotChecksTheDeviceAndRecordsTheConvention) {
    SKIP_WITHOUT_GPU();
    for (const cps::VelocityConvention convention :
         {cps::VelocityConvention::Synchronised, cps::VelocityConvention::HalfStep}) {
        const bool half_step = convention == cps::VelocityConvention::HalfStep;
        SCOPED_TRACE(half_step ? "half step" : "synchronised");
        Calls calls;
        cps::Config c = config();
        c.snapshot_dir = (directory_ / (half_step ? "half" : "sync")).string();
        RecordingSolver::Options options;
        options.convention = convention;
        cps::SimulationEngine engine(c, recording(c, calls, options));
        engine.initialize();
        engine.step();
        take(calls);

        engine.write_snapshot_now();
        EXPECT_EQ(take(calls), Calls{"check_device_errors"});
        const cps::Snapshot snapshot = cps::read_snapshot(cps::latest_snapshot(c.snapshot_dir));
        EXPECT_EQ(snapshot.step, 1u);
        EXPECT_EQ((snapshot.flags & cps::SNAPSHOT_FLAG_HALF_STEP_VELOCITY) != 0u, half_step);
    }
}

// The opening row, then the steps with the rows and the snapshots on their
// cadences, then the closing checks. Step 7 is not on the diagnostics cadence, so
// it gets a closing row.
TEST_F(SolverInterfaceTest, RunCallsTheHooksInTheOrderOfTheLoop) {
    SKIP_WITHOUT_GPU();
    Calls calls;
    cps::Config c = config();
    c.steps = 7;
    c.diagnostics_interval = 3;
    c.diagnostics_path = (directory_ / "diagnostics.csv").string();
    c.snapshot_interval = 4;
    cps::SimulationEngine engine(c, recording(c, calls));
    engine.run();

    const Calls expected = {
        "initialize",                                    // run() initialises if nothing has
        "check_device_errors", "prepare_diagnostics",    // the opening row
        "step",                "step",
        "step",                                          // step 3
        "check_device_errors", "prepare_diagnostics",    // its row
        "step",                                          // step 4
        "check_device_errors",                           // its snapshot
        "step",                "step",                   // step 6
        "check_device_errors", "prepare_diagnostics",    // its row
        "step",                                          // step 7
        "check_device_errors",                           // the end of the run
        "check_device_errors", "prepare_diagnostics",    // the closing row
    };
    EXPECT_EQ(take(calls), expected);
    EXPECT_EQ(engine.current_step(), 7);
    EXPECT_EQ(engine.steps_run(), 7);

    std::ifstream file(c.diagnostics_path);
    int lines = 0;
    for (std::string line; std::getline(file, line);) {
        ++lines;
    }
    EXPECT_EQ(lines, 1 + 4) << "a header and the rows at steps 0, 3, 6 and 7";
    EXPECT_EQ(cps::read_snapshot(cps::latest_snapshot(c.snapshot_dir)).step, 4u);
}

// A resume is not a fresh start, so it runs its own hook and not initialize, and
// the coasting state carries on bit for bit.
TEST_F(SolverInterfaceTest, AResumeCallsOnResumeAndCarriesOn) {
    SKIP_WITHOUT_GPU();
    const cps::Config c = config();
    Calls straight_calls;
    cps::SimulationEngine straight(c, recording(c, straight_calls));
    straight.initialize();
    for (int i = 0; i < 6; ++i) {
        straight.step();
    }
    cps::HostParticleState expected;
    straight.particles().download(expected, straight.stream());

    {
        Calls calls;
        cps::SimulationEngine first(c, recording(c, calls));
        first.initialize();
        for (int i = 0; i < 3; ++i) {
            first.step();
        }
        first.write_snapshot_now();
    }

    Calls calls;
    cps::SimulationEngine resumed(c, recording(c, calls));
    EXPECT_EQ(resumed.resume(), 3);
    EXPECT_EQ(take(calls), Calls{"on_resume"});
    for (int i = 0; i < 3; ++i) {
        resumed.step();
    }
    EXPECT_EQ(take(calls), (Calls{"step", "step", "step"}));
    EXPECT_EQ(resumed.current_step(), 6);

    cps::HostParticleState state;
    resumed.particles().download(state, resumed.stream());
    expect_same_state(expected, state, "resumed");
}

// Fresh run: the shift to the half step reads what initialize left in the
// acceleration array. If the engine shifted first it would read zero.
TEST_F(SolverInterfaceTest, TheConventionShiftOnInitializeSeesWhatTheHookLeft) {
    SKIP_WITHOUT_GPU();
    Calls calls;
    const cps::Config c = config();
    RecordingSolver::Options options;
    options.convention = cps::VelocityConvention::HalfStep;
    options.acceleration = 2.0f;
    cps::SimulationEngine engine(c, recording(c, calls, options));
    engine.initialize();

    const cps::HostParticleState initial = cps::make_initial_state(c);
    cps::HostParticleState state;
    engine.particles().download(state, engine.stream());
    for (std::size_t i = 0; i < state.size(); ++i) {
        ASSERT_FLOAT_EQ(state.vx[i], initial.vx[i] - 2.0f * c.dt / 2.0f) << i;
    }
}

// Resume across conventions: a snapshot with synchronised velocities into a solver
// that carries the half step. The shift reads what on_resume left, and the flag it
// then writes says the array has moved.
TEST_F(SolverInterfaceTest, TheConventionShiftOnResumeSeesWhatTheHookLeft) {
    SKIP_WITHOUT_GPU();
    const cps::Config c = config();
    cps::HostParticleState written;
    {
        Calls calls;
        cps::SimulationEngine first(c, recording(c, calls));
        first.initialize();
        first.step();
        first.write_snapshot_now();
        first.particles().download(written, first.stream());
    }

    Calls calls;
    RecordingSolver::Options options;
    options.convention = cps::VelocityConvention::HalfStep;
    options.acceleration = 2.0f;
    cps::SimulationEngine resumed(c, recording(c, calls, options));
    resumed.resume();

    cps::HostParticleState state;
    resumed.particles().download(state, resumed.stream());
    for (std::size_t i = 0; i < state.size(); ++i) {
        ASSERT_FLOAT_EQ(state.vx[i], written.vx[i] - 2.0f * c.dt / 2.0f) << i;
    }

    resumed.write_snapshot_now();
    EXPECT_NE(cps::read_snapshot(cps::latest_snapshot(c.snapshot_dir)).flags &
                  cps::SNAPSHOT_FLAG_HALF_STEP_VELOCITY,
              0u);
}

// And when the convention already matches, the loaded velocity is not touched at
// all, whatever the hook left in the acceleration array.
TEST_F(SolverInterfaceTest, AResumeInTheSameConventionLeavesTheVelocityAlone) {
    SKIP_WITHOUT_GPU();
    const cps::Config c = config();
    RecordingSolver::Options options;
    options.convention = cps::VelocityConvention::HalfStep;
    options.acceleration = 2.0f;
    cps::HostParticleState written;
    {
        Calls calls;
        cps::SimulationEngine first(c, recording(c, calls, options));
        first.initialize();
        first.step();
        first.write_snapshot_now();
        first.particles().download(written, first.stream());
    }

    Calls calls;
    cps::SimulationEngine resumed(c, recording(c, calls, options));
    resumed.resume();
    cps::HostParticleState state;
    resumed.particles().download(state, resumed.stream());
    expect_same_state(written, state, "loaded");
}

// A failure the solver reports stops a diagnostic before it prepares anything, and
// stops a snapshot before it writes anything, with the step named.
TEST_F(SolverInterfaceTest, ADeviceErrorFromTheSolverStopsTheEngine) {
    SKIP_WITHOUT_GPU();
    Calls calls;
    const cps::Config c = config();
    RecordingSolver::Options options;
    options.device_error = "the recording solver failed";
    cps::SimulationEngine engine(c, recording(c, calls, options));
    engine.initialize();
    engine.step();
    take(calls);

    const auto error_from = [](auto&& action) {
        try {
            action();
        } catch (const std::runtime_error& error) {
            return std::string(error.what());
        }
        return std::string();
    };
    EXPECT_EQ(error_from([&] { engine.check_device_errors(); }),
              "the recording solver failed at step 1");
    EXPECT_EQ(error_from([&] { engine.diagnostics(); }), "the recording solver failed at step 1");
    EXPECT_EQ(take(calls), (Calls{"check_device_errors", "check_device_errors"}));

    EXPECT_EQ(error_from([&] { engine.write_snapshot_now(); }),
              "the recording solver failed at step 1");
    EXPECT_TRUE(cps::latest_snapshot(c.snapshot_dir).empty());
}

TEST_F(SolverInterfaceTest, TheEngineExposesTheSolverItHolds) {
    SKIP_WITHOUT_GPU();
    Calls calls;
    const cps::Config c = config();
    cps::SimulationEngine engine(c, recording(c, calls));
    EXPECT_EQ(engine.solver().startup_description(), "  recording     : yes\n");
}

// The configured solver is what the engine builds when it is given none.
TEST_F(SolverInterfaceTest, TheEngineBuildsTheConfiguredSolverOtherwise) {
    SKIP_WITHOUT_GPU();
    cps::Config c = config();
    c.variant = cps::NBodyVariant::Fused;
    c.nbody.tile_size = 64;
    cps::SimulationEngine engine(c);
    const auto* solver = dynamic_cast<const cps::NBodySolver*>(&engine.solver());
    ASSERT_NE(solver, nullptr);
    EXPECT_EQ(solver->variant(), cps::NBodyVariant::Fused);
    EXPECT_EQ(solver->tile_size(), 64);
    EXPECT_EQ(solver->velocity_convention(), cps::VelocityConvention::HalfStep);
}

// One step at each tile width, compared by bit pattern with the first. The tiled
// force sum visits j in the same order at every width, and a padded slot adds +0.0,
// so the width must not reach the result. The documentation says so, and bench_all
// compares widths on the strength of it. This binary links the strict library; the
// fast math library is measured separately and not asserted.
class TileWidthTest : public ::testing::Test {
   protected:
    struct Accelerations {
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> z;
    };

    // Safe after synchronize(), by the contract in simulation_engine.hpp.
    static Accelerations read_accelerations(cps::SimulationEngine& engine) {
        engine.synchronize();
        cps::ParticleSystem& particles = engine.particles();
        const std::size_t n = static_cast<std::size_t>(particles.size());
        Accelerations out{std::vector<float>(n), std::vector<float>(n), std::vector<float>(n)};
        CUDA_CHECK(cudaMemcpy(out.x.data(), particles.ax(), n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out.y.data(), particles.ay(), n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out.z.data(), particles.az(), n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return out;
    }

    static void expect_same_accelerations(const Accelerations& a, const Accelerations& b,
                                          const std::string& label) {
        expect_same_bits(a.x, b.x, label + " ax");
        expect_same_bits(a.y, b.y, label + " ay");
        expect_same_bits(a.z, b.z, label + " az");
    }
};

TEST_F(TileWidthTest, OneStepIsBitIdenticalAtEveryWidth) {
    SKIP_WITHOUT_GPU();
    for (const cps::NBodyVariant variant : {cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
        // 1000 leaves a partly filled last tile at every width, and is not a
        // multiple of the block size either.
        for (const int count : {1000, 4096}) {
            cps::Config base;
            base.solver = cps::SolverType::NBody;
            base.variant = variant;
            base.initial_condition = cps::InitialCondition::Plummer;
            base.num_particles = count;
            base.dt = 1.0e-3f;
            base.softening = 0.05f;
            base.block_size = 128;
            base.snapshot_interval = 0;
            base.diagnostics_interval = 0;
            base.diagnostics_path.clear();

            const auto run_at = [&](int width, Accelerations& seed, Accelerations& after,
                                    cps::HostParticleState& state) {
                cps::Config config = base;
                config.nbody.tile_size = width;
                cps::SimulationEngine engine(config);
                engine.initialize();
                seed = read_accelerations(engine);
                engine.step();
                after = read_accelerations(engine);
                engine.particles().download(state, engine.stream());
            };

            Accelerations reference_seed;
            Accelerations reference_after;
            cps::HostParticleState reference_state;
            run_at(32, reference_seed, reference_after, reference_state);

            for (const int width : {64, 128, 256}) {
                const std::string label = cps::to_string(variant) + " N=" + std::to_string(count) +
                                          " width " + std::to_string(width);
                SCOPED_TRACE(label);
                Accelerations seed;
                Accelerations after;
                cps::HostParticleState state;
                run_at(width, seed, after, state);
                expect_same_accelerations(reference_seed, seed, label + " seed");
                expect_same_accelerations(reference_after, after, label + " after the step");
                expect_same_state(reference_state, state, label);
            }
        }
    }
}

}  // namespace
