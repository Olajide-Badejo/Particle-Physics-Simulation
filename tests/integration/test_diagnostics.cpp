#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/core/simulation_engine.hpp"
#include "test_support_gpu.hpp"

namespace fs = std::filesystem;

namespace {

// Tiled and fused start from one initial condition and follow the same orbit,
// so once the fused velocity is properly synchronised their total energies
// differ only by rounding. In single precision that rounding is a random walk
// that reaches about 4e-6 over a thousand steps, so the 1e-6 first predicted for
// this comparison cannot be met by any correct code. In double precision the
// same two integrators agree to about 1e-14.
//
// At dt = 1e-3 the old stale force bias was only about 2e-6, inside that noise,
// so this limit bounds the rounding and cannot see that bias.
constexpr double TILED_VS_FUSED_ENERGY_LIMIT = 1.0e-5;

// The bias grows as dt squared and the noise does not, so a larger step is what
// separates them: at dt = 1e-2 the bias is many times this limit.
constexpr double LARGE_STEP_DT = 1.0e-2;
constexpr double LARGE_STEP_ENERGY_LIMIT = 2.0e-5;

class DiagnosticsTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = fs::temp_directory_path() / ("cps_diagnostics_" + std::to_string(stamp));
        fs::create_directories(directory_);
    }

    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(directory_, ignored);
    }

    fs::path csv_path() const { return directory_ / "diagnostics.csv"; }

    cps::Config nbody_config(cps::NBodyVariant variant, int count = 64) const {
        cps::Config config;
        config.solver = cps::SolverType::NBody;
        config.variant = variant;
        config.initial_condition = cps::InitialCondition::Plummer;
        config.num_particles = count;
        config.dt = 1.0e-3f;
        config.softening = 0.05f;
        config.gravitational_constant = 1.0f;
        config.block_size = 32;
        config.seed = 20260802u;
        config.snapshot_dir = (directory_ / "snapshots").string();
        config.snapshot_interval = 0;
        config.diagnostics_interval = 10;
        config.diagnostics_path = csv_path().string();
        return config;
    }

    // A run from the start to total steps, with a snapshot every snapshot_every.
    static void run_fresh(cps::Config config, long long total, long long snapshot_every) {
        config.steps = total;
        config.snapshot_interval = snapshot_every;
        cps::SimulationEngine engine(config);
        engine.initialize();
        engine.run();
    }

    // A resume from the newest snapshot, carried on to total steps.
    static void run_resumed(cps::Config config, long long total) {
        config.steps = total;
        config.snapshot_interval = 0;
        cps::SimulationEngine engine(config);
        engine.resume();
        engine.run();
    }

    static std::string read_text(const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        std::ostringstream buffer;
        buffer << in.rdbuf();
        return buffer.str();
    }

    static void write_text(const fs::path& path, const std::string& text) {
        std::ofstream out(path, std::ios::binary);
        out << text;
    }

    static std::vector<std::string> split_lines(const std::string& text) {
        std::vector<std::string> lines;
        std::istringstream in(text);
        for (std::string line; std::getline(in, line);) {
            lines.push_back(line);
        }
        return lines;
    }

    // Relative gap in final total energy between tiled and fused two body runs
    // of a thousand steps from one initial condition. Prints what it measured.
    double tiled_versus_fused_energy(float dt) const {
        const auto final_total = [&](cps::NBodyVariant variant) {
            cps::Config config = nbody_config(variant, 2);
            config.initial_condition = cps::InitialCondition::TwoBody;
            config.dt = dt;
            config.softening = 0.01f;
            config.total_mass = 2.0f;
            config.two_body_separation = 1.0f;
            config.steps = 1000;
            config.diagnostics_interval = 0;
            config.diagnostics_path.clear();
            cps::SimulationEngine engine(config);
            engine.initialize();
            for (long long i = 0; i < config.steps; ++i) {
                engine.step();
            }
            return engine.diagnostics(true).total;
        };
        const double tiled = final_total(cps::NBodyVariant::Tiled);
        const double fused = final_total(cps::NBodyVariant::Fused);
        const double difference = std::abs(tiled - fused) / std::abs(tiled);
        std::printf("two body total energy after 1000 steps at dt = %g: tiled %.12g, fused %.12g, "
                    "relative difference %.3e\n",
                    static_cast<double>(dt), tiled, fused, difference);
        return difference;
    }

    // One header line, then each listed step exactly once and in order.
    static void expect_header_and_steps(const std::string& text, long long first, long long last,
                                        long long every) {
        const std::vector<std::string> lines = split_lines(text);
        ASSERT_FALSE(lines.empty());
        EXPECT_EQ(lines.front(), cps::DIAGNOSTICS_HEADER);
        std::vector<long long> found;
        for (std::size_t i = 1; i < lines.size(); ++i) {
            found.push_back(std::stoll(lines[i]));
        }
        std::vector<long long> expected;
        for (long long step = first; step <= last; step += every) {
            expected.push_back(step);
        }
        EXPECT_EQ(found, expected);
    }

    fs::path directory_;
};

// The full history survives the resume, and it is the history an uninterrupted
// run writes: same rows, same time column, same bits. That holds for the fused
// variant only because its diagnostics recompute the force, so a state reports
// the same energy whether or not the process restarted just before it.
TEST_F(DiagnosticsTest, AResumedRunKeepsItsHistoryAndWritesEveryStepOnce) {
    SKIP_WITHOUT_GPU();
    for (cps::NBodyVariant variant :
         {cps::NBodyVariant::Naive, cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
        SCOPED_TRACE(cps::to_string(variant));
        const cps::Config config = nbody_config(variant);

        run_fresh(config, 100, 0);
        const std::string straight = read_text(csv_path());
        expect_header_and_steps(straight, 0, 100, 10);

        run_fresh(config, 50, 50);
        run_resumed(config, 100);
        const std::string resumed = read_text(csv_path());
        expect_header_and_steps(resumed, 0, 100, 10);
        EXPECT_EQ(resumed, straight);

        fs::remove_all(config.snapshot_dir);
    }
}

// A run that dies after its last snapshot has diagnostics ahead of the state a
// resume restores, and possibly a row cut off in the middle of a write.
TEST_F(DiagnosticsTest, RowsPastTheSnapshotAreDroppedAndWrittenOnce) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = nbody_config(cps::NBodyVariant::Tiled);

    run_fresh(config, 100, 0);
    const std::string straight = read_text(csv_path());

    run_fresh(config, 70, 50);
    const std::string crashed = read_text(csv_path()) + "80,0.08,1.25";
    write_text(csv_path(), crashed);
    expect_header_and_steps(read_text(csv_path()).substr(0, crashed.rfind('\n') + 1), 0, 70, 10);

    run_resumed(config, 100);
    const std::string resumed = read_text(csv_path());
    expect_header_and_steps(resumed, 0, 100, 10);
    EXPECT_EQ(resumed, straight);
    EXPECT_FALSE(fs::exists(csv_path().string() + ".tmp"));
}

// The rows that come back are the resumed run's, not whatever was in the file.
TEST_F(DiagnosticsTest, StaleRowsPastTheSnapshotDoNotSurvive) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = nbody_config(cps::NBodyVariant::Tiled);
    run_fresh(config, 50, 50);
    write_text(csv_path(), read_text(csv_path()) + "60,-1,-1,-1,-1,-1,-1,-1,-1\n");

    run_resumed(config, 60);
    const std::string resumed = read_text(csv_path());
    expect_header_and_steps(resumed, 0, 60, 10);
    EXPECT_EQ(resumed.find(",-1,-1,-1,"), std::string::npos) << resumed;
}

TEST_F(DiagnosticsTest, AResumeWithNoFileStartsOneWithAHeader) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = nbody_config(cps::NBodyVariant::Tiled);
    run_fresh(config, 50, 50);

    fs::remove(csv_path());
    run_resumed(config, 70);
    expect_header_and_steps(read_text(csv_path()), 50, 70, 10);

    write_text(csv_path(), "");
    run_resumed(config, 70);
    expect_header_and_steps(read_text(csv_path()), 50, 70, 10);
}

TEST_F(DiagnosticsTest, AResumeRefusesAFileInAnotherFormat) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = nbody_config(cps::NBodyVariant::Tiled);
    run_fresh(config, 50, 50);

    const std::string foreign = "step,energy\n0,1.5\n";
    write_text(csv_path(), foreign);
    cps::SimulationEngine engine(config);
    try {
        engine.resume();
        FAIL() << "the resume should have refused the file";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(csv_path().string()), std::string::npos) << message;
        EXPECT_NE(message.find("expected header"), std::string::npos) << message;
    }
    EXPECT_EQ(read_text(csv_path()), foreign);
}

TEST_F(DiagnosticsTest, AResumeRefusesARowWhoseStepCannotBeRead) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = nbody_config(cps::NBodyVariant::Tiled);
    run_fresh(config, 50, 50);

    const std::string damaged = read_text(csv_path()) + "not a row\n";
    write_text(csv_path(), damaged);
    cps::SimulationEngine engine(config);
    EXPECT_THROW(engine.resume(), std::runtime_error);
    EXPECT_EQ(read_text(csv_path()), damaged);
}

// A resume refused over its physics must not cost the original run its file.
TEST_F(DiagnosticsTest, ARefusedResumeLeavesTheFileAlone) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = nbody_config(cps::NBodyVariant::Tiled);
    run_fresh(config, 50, 50);
    const std::string before = read_text(csv_path());

    cps::Config changed = config;
    changed.dt = 2.0e-3f;
    cps::SimulationEngine engine(changed);
    EXPECT_THROW(engine.resume(), std::runtime_error);
    EXPECT_EQ(read_text(csv_path()), before);
}

TEST_F(DiagnosticsTest, AFreshRunStillTruncates) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = nbody_config(cps::NBodyVariant::Tiled);
    write_text(csv_path(), "something else entirely\n");
    run_fresh(config, 20, 0);
    expect_header_and_steps(read_text(csv_path()), 0, 20, 10);
}

// Both paths conserve energy and follow the same orbit, so a real gap here
// means the fused path is measuring something other than its own state.
TEST_F(DiagnosticsTest, TiledAndFusedTotalEnergyAgreeOverAThousandSteps) {
    SKIP_WITHOUT_GPU();
    const double difference = tiled_versus_fused_energy(1.0e-3f);
    EXPECT_LT(difference, TILED_VS_FUSED_ENERGY_LIMIT);
}

// The same comparison where a stale force in the fused velocity would show.
TEST_F(DiagnosticsTest, TiledAndFusedTotalEnergyAgreeAtALargeStep) {
    SKIP_WITHOUT_GPU();
    const double difference = tiled_versus_fused_energy(static_cast<float>(LARGE_STEP_DT));
    EXPECT_LT(difference, LARGE_STEP_ENERGY_LIMIT);
}

// The fused diagnostic used to read the force at the previous positions until a
// restart recomputed it, so one state could report two energies.
TEST_F(DiagnosticsTest, AFusedStateReportsOneEnergyAcrossAResume) {
    SKIP_WITHOUT_GPU();
    cps::Config config = nbody_config(cps::NBodyVariant::Fused);
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    config.steps = 25;

    cps::Diagnostics before;
    {
        cps::SimulationEngine engine(config);
        engine.initialize();
        for (int i = 0; i < 25; ++i) {
            engine.step();
        }
        before = engine.diagnostics(true);
        engine.write_snapshot_now();
    }
    cps::SimulationEngine engine(config);
    engine.resume();
    const cps::Diagnostics after = engine.diagnostics(true);
    EXPECT_EQ(std::bit_cast<uint64_t>(before.kinetic), std::bit_cast<uint64_t>(after.kinetic));
    EXPECT_EQ(std::bit_cast<uint64_t>(before.total), std::bit_cast<uint64_t>(after.total));
}

TEST_F(DiagnosticsTest, AForcedResumeAcrossADtChangeKeepsTheSnapshotClock) {
    SKIP_WITHOUT_GPU();
    cps::Config config = nbody_config(cps::NBodyVariant::Tiled);
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    run_fresh(config, 30, 30);
    const double t0 = cps::read_snapshot(cps::latest_snapshot(config.snapshot_dir)).time;
    EXPECT_EQ(std::bit_cast<uint64_t>(t0),
              std::bit_cast<uint64_t>(30.0 * static_cast<double>(config.dt)));

    cps::Config changed = config;
    changed.dt = 2.5e-3f;
    cps::SimulationEngine engine(changed);
    ::testing::internal::CaptureStderr();
    engine.resume(true);
    ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(std::bit_cast<uint64_t>(engine.simulated_time()), std::bit_cast<uint64_t>(t0));

    const int k = 7;
    for (int i = 0; i < k; ++i) {
        engine.step();
    }
    const double expected = t0 + static_cast<double>(k) * static_cast<double>(changed.dt);
    EXPECT_EQ(std::bit_cast<uint64_t>(engine.simulated_time()), std::bit_cast<uint64_t>(expected));
    EXPECT_EQ(std::bit_cast<uint64_t>(engine.diagnostics(true).time),
              std::bit_cast<uint64_t>(expected));

    engine.write_snapshot_now();
    const cps::Snapshot written = cps::read_snapshot(cps::latest_snapshot(config.snapshot_dir));
    EXPECT_EQ(written.step, 37u);
    EXPECT_EQ(std::bit_cast<uint64_t>(written.time), std::bit_cast<uint64_t>(expected));
}

TEST_F(DiagnosticsTest, AnOrdinaryResumeReportsTheStraightRunsTime) {
    SKIP_WITHOUT_GPU();
    cps::Config config = nbody_config(cps::NBodyVariant::Tiled);
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    run_fresh(config, 30, 30);

    cps::SimulationEngine engine(config);
    engine.resume();
    for (int i = 0; i < 7; ++i) {
        engine.step();
    }
    EXPECT_EQ(std::bit_cast<uint64_t>(engine.simulated_time()),
              std::bit_cast<uint64_t>(37.0 * static_cast<double>(config.dt)));
}

// CPS1 stored no time, so the only clock available is the step count.
TEST_F(DiagnosticsTest, ACps1ResumeTakesItsClockFromTheStep) {
    SKIP_WITHOUT_GPU();
    const fs::path fixture = fs::path(CPS_PROJECT_ROOT) / "tests" / "data" / "snapshot_v1_4p.bin";
    ASSERT_TRUE(fs::exists(fixture)) << fixture;
    cps::Config config = nbody_config(cps::NBodyVariant::Tiled, 4);
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    fs::create_directories(config.snapshot_dir);
    fs::copy_file(fixture, cps::snapshot_path_for_step(config.snapshot_dir, 42u));

    cps::SimulationEngine engine(config);
    ::testing::internal::CaptureStderr();
    engine.resume();
    ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(std::bit_cast<uint64_t>(engine.simulated_time()),
              std::bit_cast<uint64_t>(42.0 * static_cast<double>(config.dt)));
}

// Diagnostics every step are slow enough that the two figures cannot be confused.
TEST_F(DiagnosticsTest, SolverTimeIsReportedSeparatelyAndNeverExceedsWallTime) {
    SKIP_WITHOUT_GPU();
    cps::Config config = nbody_config(cps::NBodyVariant::Tiled, 1024);
    config.diagnostics_interval = 1;
    config.steps = 40;
    cps::SimulationEngine engine(config);
    engine.initialize();
    engine.run();
    EXPECT_GT(engine.solver_milliseconds(), 0.0);
    EXPECT_GT(engine.wall_milliseconds(), 0.0);
    EXPECT_LE(engine.solver_milliseconds(), engine.wall_milliseconds());
}

// A step count that included the steps before the resume would overstate the
// speed of every resumed run, the longer the history the worse.
TEST_F(DiagnosticsTest, OnlyTheStepsOfThisProcessAreCounted) {
    SKIP_WITHOUT_GPU();
    cps::Config config = nbody_config(cps::NBodyVariant::Tiled);
    run_fresh(config, 50, 50);

    config.steps = 80;
    config.snapshot_interval = 0;
    cps::SimulationEngine engine(config);
    engine.resume();
    engine.run();
    EXPECT_EQ(engine.steps_run(), 30);
    EXPECT_EQ(engine.current_step(), 80);
}

}  // namespace
