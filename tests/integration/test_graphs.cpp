#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/core/simulation_engine.hpp"
#include "cps/solvers/nbody_solver.hpp"
#include "test_support_gpu.hpp"

namespace fs = std::filesystem;

namespace {

// Odd on purpose: a run of the fused variant ends with its buffers the other way
// round from where they began, which it has to do the same under either setting.
constexpr long long STEPS = 101;

struct Case {
    const char* label;
    cps::SolverType solver;
    cps::NBodyVariant variant;
    int count;
    // Whether this solver's step is captured when graphs are on.
    bool captured;
};

constexpr Case EVERY_SOLVER[] = {
    {"naive", cps::SolverType::NBody, cps::NBodyVariant::Naive, 1024, true},
    {"tiled", cps::SolverType::NBody, cps::NBodyVariant::Tiled, 2048, true},
    {"fused", cps::SolverType::NBody, cps::NBodyVariant::Fused, 2048, false},
    {"sph", cps::SolverType::Sph, cps::NBodyVariant::Tiled, 4096, true},
};

constexpr Case FUSED = EVERY_SOLVER[2];

// Compares bit patterns, not values: float equality would accept -0 for +0.
void expect_bit_identical(const std::vector<float>& a, const std::vector<float>& b,
                          const std::string& label) {
    ASSERT_EQ(a.size(), b.size()) << label;
    for (std::size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(std::bit_cast<uint32_t>(a[i]), std::bit_cast<uint32_t>(b[i]))
            << label << " at index " << i;
    }
}

void expect_bit_identical(const cps::HostParticleState& a, const cps::HostParticleState& b,
                          const std::string& label) {
    expect_bit_identical(a.x, b.x, label + " x");
    expect_bit_identical(a.y, b.y, label + " y");
    expect_bit_identical(a.z, b.z, label + " z");
    expect_bit_identical(a.vx, b.vx, label + " vx");
    expect_bit_identical(a.vy, b.vy, label + " vy");
    expect_bit_identical(a.vz, b.vz, label + " vz");
    expect_bit_identical(a.mass, b.mass, label + " mass");
    EXPECT_EQ(a.id, b.id) << label;
}

bool all_bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](float left, float right) {
        return std::bit_cast<uint32_t>(left) == std::bit_cast<uint32_t>(right);
    });
}

void expect_same_row(const cps::Diagnostics& a, const cps::Diagnostics& b,
                     const std::string& label) {
    EXPECT_EQ(a.step, b.step) << label;
    const auto bits = [](double value) { return std::bit_cast<uint64_t>(value); };
    EXPECT_EQ(bits(a.time), bits(b.time)) << label;
    EXPECT_EQ(bits(a.kinetic), bits(b.kinetic)) << label;
    EXPECT_EQ(bits(a.potential), bits(b.potential)) << label;
    EXPECT_EQ(bits(a.total), bits(b.total)) << label;
    EXPECT_EQ(bits(a.px), bits(b.px)) << label;
    EXPECT_EQ(bits(a.py), bits(b.py)) << label;
    EXPECT_EQ(bits(a.pz), bits(b.pz)) << label;
    EXPECT_EQ(bits(a.momentum_magnitude), bits(b.momentum_magnitude)) << label;
}

std::string file_bytes(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream) << path;
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

cps::HostParticleState download(cps::SimulationEngine& engine) {
    cps::HostParticleState state;
    engine.particles().download(state, engine.stream());
    return state;
}

// Coasts on the integrator launcher, as a solver written with no thought of graphs
// would. Only what a test overrides differs from the defaults.
class CoastingSolver : public cps::Solver {
   public:
    CoastingSolver(float dt, int block_size) : dt_(dt), block_size_(block_size) {}

    void initialize(cps::ParticleSystem&, cudaStream_t) override {}
    void on_resume(cps::ParticleSystem&, cudaStream_t) override {}

    void step(cps::ParticleSystem& particles, cudaStream_t stream) override {
        cps::integrate_verlet_position(particles.x(), particles.y(), particles.z(), particles.vx(),
                                       particles.vy(), particles.vz(), particles.ax(),
                                       particles.ay(), particles.az(), particles.size(), dt_,
                                       block_size_, stream);
    }

    cps::VelocityConvention velocity_convention() const override {
        return cps::VelocityConvention::Synchronised;
    }

    void prepare_diagnostics(cps::ParticleSystem&, cudaStream_t) override {}

    bool has_potential_energy() const override { return false; }

   private:
    float dt_;
    int block_size_;
};

// Says it can be captured, and then waits on the stream in its step.
class WaitingSolver final : public CoastingSolver {
   public:
    using CoastingSolver::CoastingSolver;

    void step(cps::ParticleSystem& particles, cudaStream_t stream) override {
        CoastingSolver::step(particles, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    bool graph_capturable() const override { return true; }
};

class Graphs : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = fs::temp_directory_path() / ("cps_graphs_" + std::to_string(stamp));
        fs::create_directories(directory_);
    }

    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(directory_, ignored);
    }

    cps::Config config_for(const Case& c, bool graphs, const std::string& name = "run") const {
        cps::Config config;
        config.name = name;
        config.solver = c.solver;
        config.variant = c.variant;
        config.num_particles = c.count;
        config.steps = STEPS;
        config.dt = 1.0e-3f;
        config.block_size = 128;
        config.seed = 20260802u;
        config.use_graphs = graphs;
        config.diagnostics_interval = 0;
        config.diagnostics_path.clear();
        config.snapshot_dir = (directory_ / name / "snapshots").string();
        config.snapshot_interval = 0;
        if (c.solver == cps::SolverType::Sph) {
            config.initial_condition = cps::InitialCondition::DamBreak;
            config.dt = 1.0e-4f;
            config.external_gravity = {0.0f, -9.81f, 0.0f};
            config.sph.domain_min = {0.0f, 0.0f, 0.0f};
            config.sph.domain_max = {0.8f, 0.5f, 0.4f};
            config.sph.fluid_min = {0.0f, 0.0f, 0.0f};
            config.sph.fluid_max = {0.3f, 0.4f, 0.4f};
        } else {
            config.initial_condition = cps::InitialCondition::Plummer;
            config.softening = 0.05f;
            config.gravitational_constant = 1.0f;
        }
        return config;
    }

    // Steps from a fresh start and asserts which path actually ran, so a change that
    // quietly stopped using graphs could not pass as agreement.
    cps::HostParticleState run_steps(const Case& c, bool graphs, long long steps) const {
        cps::SimulationEngine engine(config_for(c, graphs));
        engine.initialize();
        EXPECT_EQ(engine.using_graphs(), graphs && c.captured);
        for (long long i = 0; i < steps; ++i) {
            engine.step();
        }
        return download(engine);
    }

    fs::path directory_;
};

TEST_F(Graphs, GraphAndStreamRunsAreBitIdentical) {
    SKIP_WITHOUT_GPU();
    for (const Case& c : EVERY_SOLVER) {
        SCOPED_TRACE(c.label);
        const cps::HostParticleState stream = run_steps(c, false, STEPS);
        const cps::HostParticleState graph = run_steps(c, true, STEPS);
        expect_bit_identical(stream, graph, c.label);
    }
}

// It launches one kernel, so a capture would have nothing to remove. Asking for graphs
// is not an error: the run goes on the stream, with nothing captured.
TEST_F(Graphs, TheFusedVariantRunsOnTheStreamWhenGraphsAreOn) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(config_for(FUSED, true));
    engine.initialize();
    EXPECT_FALSE(engine.solver().graph_capturable());
    EXPECT_FALSE(engine.using_graphs());
    EXPECT_EQ(engine.graph_capture_milliseconds(), 0.0);
    engine.step();
    EXPECT_EQ(engine.current_step(), 1);
}

// Capture is only a recording: the step counter and the state do not move, and no
// device work has run.
TEST_F(Graphs, CapturingRunsNothing) {
    SKIP_WITHOUT_GPU();
    for (const Case& c : EVERY_SOLVER) {
        SCOPED_TRACE(c.label);
        cps::SimulationEngine stream(config_for(c, false));
        stream.initialize();
        cps::SimulationEngine graph(config_for(c, true));
        graph.initialize();
        EXPECT_EQ(graph.current_step(), 0);
        EXPECT_EQ(graph.graph_capture_milliseconds() > 0.0, c.captured);
        expect_bit_identical(download(stream), download(graph), c.label);
    }
}

// A diagnostic runs on the stream between launches. It has to see the same state as
// on the stream path, and must not disturb what follows.
TEST_F(Graphs, DiagnosticsTakenMidRunMatchTheStreamPath) {
    SKIP_WITHOUT_GPU();
    for (const Case& c : EVERY_SOLVER) {
        SCOPED_TRACE(c.label);
        cps::SimulationEngine stream(config_for(c, false));
        cps::SimulationEngine graph(config_for(c, true));
        stream.initialize();
        graph.initialize();
        ASSERT_EQ(graph.using_graphs(), c.captured);
        for (long long i = 1; i <= STEPS; ++i) {
            stream.step();
            graph.step();
            // Prime, so the rows land on odd and even steps alike.
            if (i % 7 == 0) {
                const std::string label = std::string(c.label) + " step " + std::to_string(i);
                expect_same_row(stream.diagnostics(), graph.diagnostics(), label);
            }
        }
        expect_bit_identical(download(stream), download(graph), c.label);
    }
}

// The whole of run(), with its solver clock, every cadence and the final checks.
TEST_F(Graphs, ARunWritesTheSameFilesEitherWay) {
    SKIP_WITHOUT_GPU();
    for (const Case& c : EVERY_SOLVER) {
        SCOPED_TRACE(c.label);
        std::vector<fs::path> diagnostics;
        std::vector<fs::path> snapshots;
        for (const bool graphs : {false, true}) {
            const std::string name = std::string(c.label) + (graphs ? "_on" : "_off");
            cps::Config config = config_for(c, graphs, name);
            config.snapshot_interval = 25;
            config.diagnostics_interval = 11;
            config.diagnostics_path = (directory_ / name / "diagnostics.csv").string();
            cps::SimulationEngine engine(config);
            engine.run();
            EXPECT_EQ(engine.using_graphs(), graphs && c.captured);
            EXPECT_EQ(engine.current_step(), STEPS);
            EXPECT_GT(engine.solver_milliseconds(), 0.0);
            diagnostics.emplace_back(config.diagnostics_path);
            snapshots.push_back(cps::latest_snapshot(config.snapshot_dir));
        }
        EXPECT_EQ(file_bytes(diagnostics[0]), file_bytes(diagnostics[1]));
        EXPECT_EQ(snapshots[0].filename(), snapshots[1].filename());
        EXPECT_EQ(file_bytes(snapshots[0]), file_bytes(snapshots[1]));
    }
}

// dt is a scalar in the capture. A resume forced across a change of dt builds a new
// engine, and its capture has to hold the new value: the stream path, which reads dt
// on every launch, is the reference, and the old dt must give something different.
TEST_F(Graphs, AForcedResumeAcrossADtChangeUsesTheNewDt) {
    SKIP_WITHOUT_GPU();
    for (const Case& c : EVERY_SOLVER) {
        SCOPED_TRACE(c.label);
        const std::string label = c.label;
        cps::Config original = config_for(c, true, label);
        {
            cps::SimulationEngine engine(original);
            engine.initialize();
            for (int i = 0; i < 21; ++i) {
                engine.step();
            }
            engine.write_snapshot_now();
        }

        const auto resumed_under = [&](float dt, bool graphs) {
            cps::Config config = original;
            config.dt = dt;
            config.use_graphs = graphs;
            cps::SimulationEngine engine(config);
            ::testing::internal::CaptureStderr();
            const long long at = engine.resume(true);
            static_cast<void>(::testing::internal::GetCapturedStderr());
            EXPECT_EQ(at, 21);
            EXPECT_EQ(engine.using_graphs(), graphs && c.captured);
            for (int i = 0; i < 40; ++i) {
                engine.step();
            }
            return download(engine);
        };

        const float new_dt = 0.5f * original.dt;
        const cps::HostParticleState stream = resumed_under(new_dt, false);
        const cps::HostParticleState graph = resumed_under(new_dt, true);
        expect_bit_identical(stream, graph, label);

        const cps::HostParticleState old_dt = resumed_under(original.dt, true);
        EXPECT_FALSE(all_bits_equal(old_dt.x, graph.x)) << "the new dt made no difference";
    }
}

// The device flag and both state scans sit outside the graph, so a bad state still
// stops the run, at the step it reaches them.
TEST_F(Graphs, ANonFiniteStateStillStopsARunNamingTheStep) {
    SKIP_WITHOUT_GPU();
    const float not_a_number = std::numeric_limits<float>::quiet_NaN();
    for (const Case& c : EVERY_SOLVER) {
        SCOPED_TRACE(c.label);
        cps::Config config = config_for(c, true);
        config.steps = 3;
        cps::SimulationEngine engine(config);
        engine.initialize();
        ASSERT_EQ(engine.using_graphs(), c.captured);
        cps::HostParticleState state = download(engine);
        state.x[40] = not_a_number;
        engine.particles().upload(state, engine.stream());
        std::string message;
        try {
            engine.run();
        } catch (const std::runtime_error& error) {
            message = error.what();
        }
        EXPECT_NE(message.find("non-finite"), std::string::npos) << message;
        EXPECT_NE(message.find("step 3"), std::string::npos) << message;
    }
}

// The fluid's own flag is written by a captured hash pass and read outside the graph.
TEST_F(Graphs, TheDeviceErrorFlagIsSetFromInsideTheGraph) {
    SKIP_WITHOUT_GPU();
    const Case& sph = EVERY_SOLVER[3];
    cps::SimulationEngine engine(config_for(sph, true));
    engine.initialize();
    ASSERT_TRUE(engine.using_graphs());
    cps::HostParticleState state = download(engine);
    state.y[117] = std::numeric_limits<float>::quiet_NaN();
    engine.particles().upload(state, engine.stream());
    EXPECT_NO_THROW(engine.check_device_errors());

    engine.step();
    std::string message;
    try {
        engine.check_device_errors();
    } catch (const std::runtime_error& error) {
        message = error.what();
    }
    EXPECT_NE(message.find("non-finite positions detected by step 1:"), std::string::npos)
        << message;
}

// A solver that says nothing about graphs runs on the stream, whatever is configured.
TEST_F(Graphs, ACustomSolverRunsWithNoGraph) {
    SKIP_WITHOUT_GPU();
    cps::Config config = config_for(EVERY_SOLVER[1], true);
    config.steps = 5;
    const auto run_with = [&](bool graphs) {
        cps::Config local = config;
        local.use_graphs = graphs;
        cps::SimulationEngine engine(
            local, std::make_unique<CoastingSolver>(config.dt, config.block_size));
        engine.run();
        EXPECT_FALSE(engine.using_graphs());
        EXPECT_EQ(engine.graph_capture_milliseconds(), 0.0);
        EXPECT_EQ(engine.current_step(), 5);
        return download(engine);
    };
    expect_bit_identical(run_with(false), run_with(true), "coasting");
}

// A wait inside a capture would otherwise surface as a graph that does something else.
//
// A suite of its own, because the failure it provokes is a CUDA API error, which
// compute-sanitizer counts. That leaves Graphs.* clean under memcheck.
using GraphRefusal = Graphs;

TEST_F(GraphRefusal, AStepThatWaitsOnTheStreamIsRefusedAtCapture) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = config_for(EVERY_SOLVER[1], true);
    cps::SimulationEngine engine(config,
                                 std::make_unique<WaitingSolver>(config.dt, config.block_size));
    EXPECT_THROW(engine.initialize(), std::runtime_error);
    EXPECT_FALSE(engine.using_graphs());

    // The stream is not left in capture.
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusActive;
    CUDA_CHECK(cudaStreamIsCapturing(engine.stream(), &status));
    EXPECT_EQ(status, cudaStreamCaptureStatusNone);
}

}  // namespace
