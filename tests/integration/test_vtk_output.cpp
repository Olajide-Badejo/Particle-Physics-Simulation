#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cps/core/simulation_engine.hpp"
#include "cps/io/vtk.hpp"
#include "cps/solvers/nbody_solver.hpp"
#include "test_support_gpu.hpp"

namespace fs = std::filesystem;

namespace {

using Bytes = std::vector<char>;

Bytes read_all(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// Coasts, with a position that turns non-finite at one chosen step. The real solvers
// do not blow up on demand, and an edit from outside could not land between two
// steps of a run.
class BlowsUpAtStep final : public cps::Solver {
   public:
    BlowsUpAtStep(float dt, int block_size, int bad_step)
        : dt_(dt), block_size_(block_size), bad_step_(bad_step) {}

    void initialize(cps::ParticleSystem&, cudaStream_t) override {}
    void on_resume(cps::ParticleSystem&, cudaStream_t) override {}

    void step(cps::ParticleSystem& particles, cudaStream_t stream) override {
        cps::integrate_verlet_position(particles.x(), particles.y(), particles.z(), particles.vx(),
                                       particles.vy(), particles.vz(), particles.ax(),
                                       particles.ay(), particles.az(), particles.size(), dt_,
                                       block_size_, stream);
        if (++steps_ == bad_step_) {
            cps::HostParticleState state;
            particles.download(state, stream);
            state.y[3] = std::numeric_limits<float>::quiet_NaN();
            particles.upload(state, stream);
        }
    }

    cps::VelocityConvention velocity_convention() const override {
        return cps::VelocityConvention::Synchronised;
    }
    void prepare_diagnostics(cps::ParticleSystem&, cudaStream_t) override {}
    bool has_potential_energy() const override { return false; }

   private:
    float dt_;
    int block_size_;
    int bad_step_;
    int steps_ = 0;
};

class VtkOutputTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = fs::temp_directory_path() / ("cps_vtk_output_" + std::to_string(stamp));
        fs::create_directories(directory_);
    }

    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(directory_, ignored);
    }

    cps::Config config(cps::NBodyVariant variant = cps::NBodyVariant::Tiled) const {
        cps::Config c;
        c.variant = variant;
        c.num_particles = 128;
        c.initial_condition = cps::InitialCondition::Plummer;
        c.softening = 0.05f;
        c.dt = 1.0e-3f;
        c.block_size = 128;
        c.steps = 25;
        c.diagnostics_interval = 0;
        c.diagnostics_path.clear();
        c.snapshot_interval = 0;
        c.snapshot_dir = (directory_ / "snapshots").string();
        c.vtk_interval = 10;
        c.vtk_dir = (directory_ / "vtk").string();
        return c;
    }

    cps::Config fluid() const {
        cps::Config c = config();
        c.solver = cps::SolverType::Sph;
        c.initial_condition = cps::InitialCondition::DamBreak;
        c.num_particles = 4096;
        c.dt = 1.0e-4f;
        c.external_gravity = {0.0f, -9.81f, 0.0f};
        c.sph.domain_min = {0.0f, 0.0f, 0.0f};
        c.sph.domain_max = {0.8f, 0.5f, 0.4f};
        c.sph.fluid_min = {0.0f, 0.0f, 0.0f};
        c.sph.fluid_max = {0.3f, 0.4f, 0.4f};
        return c;
    }

    static std::vector<uint64_t> steps_in_series(const cps::Config& c) {
        std::vector<uint64_t> steps;
        for (const cps::VtkSeriesEntry& entry :
             cps::read_vtk_series_index(fs::path(c.vtk_dir) / cps::VTK_SERIES_FILE_NAME)) {
            steps.push_back(entry.step);
            EXPECT_TRUE(fs::exists(fs::path(c.vtk_dir) / entry.file)) << entry.file;
        }
        return steps;
    }

    static std::string text_of(const Bytes& bytes) { return std::string(bytes.begin(), bytes.end()); }

    fs::path directory_;
};

// Frames fall on multiples of the cadence and on step 0, and step 25 is off it.
TEST_F(VtkOutputTest, WritesStepZeroAndEveryMultipleOfTheCadence) {
    SKIP_WITHOUT_GPU();
    const cps::Config c = config();
    cps::SimulationEngine engine(c);
    engine.run();

    EXPECT_EQ(steps_in_series(c), (std::vector<uint64_t>{0u, 10u, 20u}));
    const std::vector<cps::VtkSeriesEntry> entries =
        cps::read_vtk_series_index(fs::path(c.vtk_dir) / cps::VTK_SERIES_FILE_NAME);
    for (const cps::VtkSeriesEntry& entry : entries) {
        EXPECT_EQ(entry.time, static_cast<double>(entry.step) * static_cast<double>(c.dt));
    }
    EXPECT_EQ(engine.current_step(), 25);
}

TEST_F(VtkOutputTest, FramesOffTouchNothing) {
    SKIP_WITHOUT_GPU();
    cps::Config c = config();
    c.vtk_interval = 0;
    const fs::path keep = fs::path(c.vtk_dir) / cps::VTK_SERIES_FILE_NAME;
    fs::create_directories(c.vtk_dir);
    std::ofstream(keep) << "not touched\n";
    cps::SimulationEngine engine(c);
    engine.run();
    EXPECT_EQ(text_of(read_all(keep)), "not touched\n");
}

// What a frame holds is what the device holds. Written directly from the state the
// engine ends on, the file comes out the same to the byte.
TEST_F(VtkOutputTest, AFrameHoldsTheDeviceStateAndTheVelocityConvention) {
    SKIP_WITHOUT_GPU();
    for (const auto& [c, half_step] :
         {std::pair<cps::Config, bool>{config(cps::NBodyVariant::Tiled), false},
          std::pair<cps::Config, bool>{config(cps::NBodyVariant::Fused), true},
          std::pair<cps::Config, bool>{fluid(), true}}) {
        SCOPED_TRACE(half_step ? "half step" : "synchronised");
        cps::Config local = c;
        local.steps = 20;
        cps::SimulationEngine engine(local);
        engine.run();
        cps::HostParticleState state;
        engine.particles().download(state, engine.stream());

        const fs::path framed = cps::vtk_frame_path(local.vtk_dir, 20);
        const fs::path direct = directory_ / "direct.vtp";
        cps::write_vtk_frame(direct, state, cps::VtkFrameInfo{20u, engine.simulated_time(), half_step});
        EXPECT_EQ(read_all(framed), read_all(direct));
    }
}

TEST_F(VtkOutputTest, AFreshRunReplacesAnEarlierSeries) {
    SKIP_WITHOUT_GPU();
    cps::Config c = config();
    {
        cps::SimulationEngine engine(c);
        engine.run();
    }
    ASSERT_TRUE(fs::exists(cps::vtk_frame_path(c.vtk_dir, 20)));
    c.steps = 10;
    {
        cps::SimulationEngine engine(c);
        engine.run();
    }
    EXPECT_EQ(steps_in_series(c), (std::vector<uint64_t>{0u, 10u}));
    EXPECT_FALSE(fs::exists(cps::vtk_frame_path(c.vtk_dir, 20)));
}

// A blown-up state never reaches a frame. Frames before it stay, the series stays
// valid, and the error names the step.
TEST_F(VtkOutputTest, ANonFiniteStateIsRefusedAndTheSeriesStaysValid) {
    SKIP_WITHOUT_GPU();
    const cps::Config c = config();
    cps::SimulationEngine engine(c, std::make_unique<BlowsUpAtStep>(c.dt, c.block_size, 15));
    try {
        engine.run();
        FAIL() << "the run should have stopped at the frame due at step 20";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("non-finite state at step 20"), std::string::npos) << message;
    }
    EXPECT_EQ(steps_in_series(c), (std::vector<uint64_t>{0u, 10u}));
    EXPECT_FALSE(fs::exists(cps::vtk_frame_path(c.vtk_dir, 20)));
    EXPECT_FALSE(fs::exists(cps::vtk_frame_path(c.vtk_dir, 20).string() + ".tmp"));
}

// The same bad state at step 0, before any frame has been written: the series is
// there and empty.
TEST_F(VtkOutputTest, ANonFiniteInitialStateWritesNoFrame) {
    SKIP_WITHOUT_GPU();
    const cps::Config c = config();
    cps::SimulationEngine engine(c);
    engine.initialize();
    cps::HostParticleState state;
    engine.particles().download(state, engine.stream());
    state.vx[1] = std::numeric_limits<float>::infinity();
    engine.particles().upload(state, engine.stream());
    EXPECT_THROW(engine.run(), std::runtime_error);
    EXPECT_TRUE(steps_in_series(c).empty());
}

// The whole point: killed and resumed, the series has the frames of a run that never
// stopped, and the same bytes in them, including for the fluid, whose slot order the
// sort keeps changing.
TEST_F(VtkOutputTest, AResumedRunContinuesTheSeriesAndMatchesAStraightRun) {
    SKIP_WITHOUT_GPU();
    for (const cps::Config& base : {config(cps::NBodyVariant::Fused), fluid()}) {
        cps::Config straight = base;
        straight.vtk_dir = (directory_ / "straight").string();
        straight.snapshot_dir = (directory_ / "straight_snapshots").string();
        straight.steps = 40;
        {
            cps::SimulationEngine engine(straight);
            engine.run();
        }

        cps::Config split = base;
        split.vtk_dir = (directory_ / "split").string();
        split.snapshot_dir = (directory_ / "split_snapshots").string();
        split.snapshot_interval = 15;
        // The first leg overruns its last snapshot, at step 15, the way a run that
        // was killed does: the frame at 20 has to go.
        split.steps = 25;
        {
            cps::SimulationEngine engine(split);
            engine.run();
        }
        EXPECT_EQ(steps_in_series(split), (std::vector<uint64_t>{0u, 10u, 20u}));

        split.steps = 40;
        split.snapshot_interval = 0;
        {
            cps::SimulationEngine engine(split);
            EXPECT_EQ(engine.resume(), 15);
            EXPECT_EQ(steps_in_series(split), (std::vector<uint64_t>{0u, 10u}));
            EXPECT_FALSE(fs::exists(cps::vtk_frame_path(split.vtk_dir, 20)));
            engine.run();
        }

        EXPECT_EQ(steps_in_series(split), (std::vector<uint64_t>{0u, 10u, 20u, 30u, 40u}));
        EXPECT_EQ(read_all(fs::path(split.vtk_dir) / cps::VTK_SERIES_FILE_NAME),
                  read_all(fs::path(straight.vtk_dir) / cps::VTK_SERIES_FILE_NAME));
        for (const uint64_t step : {0u, 10u, 20u, 30u, 40u}) {
            SCOPED_TRACE(step);
            EXPECT_EQ(read_all(cps::vtk_frame_path(split.vtk_dir, step)),
                      read_all(cps::vtk_frame_path(straight.vtk_dir, step)));
        }
    }
}

// A resume landing exactly on a frame keeps it rather than dropping and rewriting.
TEST_F(VtkOutputTest, AResumeOnAFrameStepKeepsThatFrame) {
    SKIP_WITHOUT_GPU();
    cps::Config c = config();
    c.steps = 20;
    c.snapshot_interval = 20;
    {
        cps::SimulationEngine engine(c);
        engine.run();
    }
    const fs::path at_twenty = cps::vtk_frame_path(c.vtk_dir, 20);
    const auto written = fs::last_write_time(at_twenty);

    c.steps = 30;
    {
        cps::SimulationEngine engine(c);
        engine.resume();
        engine.run();
    }
    EXPECT_EQ(fs::last_write_time(at_twenty), written);
    EXPECT_EQ(steps_in_series(c), (std::vector<uint64_t>{0u, 10u, 20u, 30u}));
}

// Killed after the snapshot and before the frame at the same step, a run has a state
// to resume and no frame of it. The resume supplies it.
TEST_F(VtkOutputTest, AResumeOnACadenceStepWritesAFrameTheSeriesLacks) {
    SKIP_WITHOUT_GPU();
    cps::Config c = config();
    c.steps = 20;
    c.snapshot_interval = 20;
    {
        cps::SimulationEngine engine(c);
        engine.run();
    }
    const fs::path index = fs::path(c.vtk_dir) / cps::VTK_SERIES_FILE_NAME;
    std::vector<cps::VtkSeriesEntry> entries = cps::read_vtk_series_index(index);
    entries.pop_back();
    cps::write_vtk_series_index(index, entries);
    fs::remove(cps::vtk_frame_path(c.vtk_dir, 20));

    {
        cps::SimulationEngine engine(c);
        engine.resume();
        engine.run();
    }
    EXPECT_EQ(steps_in_series(c), (std::vector<uint64_t>{0u, 10u, 20u}));
}

// A refused resume changes nothing, the same promise the diagnostics file gets.
TEST_F(VtkOutputTest, ARefusedResumeLeavesTheSeriesAlone) {
    SKIP_WITHOUT_GPU();
    cps::Config c = config();
    c.steps = 20;
    c.snapshot_interval = 15;
    {
        cps::SimulationEngine engine(c);
        engine.run();
    }
    cps::Config different = c;
    different.dt = 2.0e-3f;
    {
        cps::SimulationEngine engine(different);
        EXPECT_THROW(engine.resume(), std::runtime_error);
    }
    EXPECT_EQ(steps_in_series(c), (std::vector<uint64_t>{0u, 10u, 20u}));
}

// Frames are run control: they change no trajectory, and they are not on the solver
// clock.
TEST_F(VtkOutputTest, FramesDoNotPerturbTheTrajectory) {
    SKIP_WITHOUT_GPU();
    for (const cps::Config& base : {config(cps::NBodyVariant::Fused), fluid()}) {
        cps::Config with_frames = base;
        with_frames.vtk_interval = 3;
        cps::Config without = base;
        without.vtk_interval = 0;

        cps::SimulationEngine a(with_frames);
        a.run();
        cps::SimulationEngine b(without);
        b.run();

        cps::HostParticleState sa;
        cps::HostParticleState sb;
        a.particles().download(sa, a.stream());
        b.particles().download(sb, b.stream());
        ASSERT_EQ(sa.size(), sb.size());
        for (std::size_t i = 0; i < sa.size(); ++i) {
            ASSERT_EQ(std::bit_cast<uint32_t>(sa.x[i]), std::bit_cast<uint32_t>(sb.x[i])) << i;
            ASSERT_EQ(std::bit_cast<uint32_t>(sa.vy[i]), std::bit_cast<uint32_t>(sb.vy[i])) << i;
            ASSERT_EQ(sa.id[i], sb.id[i]) << i;
        }
        EXPECT_LE(a.solver_milliseconds(), a.wall_milliseconds());
    }
}

}  // namespace
