#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/core/simulation_engine.hpp"
#include "test_support_gpu.hpp"

namespace fs = std::filesystem;

namespace {

class ResumeTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = fs::temp_directory_path() / ("cps_resume_" + std::to_string(stamp));
        fs::create_directories(directory_);
    }

    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(directory_, ignored);
    }

    cps::Config base_config(cps::SolverType solver, cps::NBodyVariant variant, int count) {
        cps::Config config;
        config.solver = solver;
        config.variant = variant;
        config.num_particles = count;
        config.dt = 1.0e-3f;
        config.block_size = 128;
        config.seed = 20260802u;
        config.diagnostics_interval = 0;
        config.diagnostics_path.clear();
        config.snapshot_dir = directory_.string();
        config.snapshot_interval = 0;

        if (solver == cps::SolverType::Sph) {
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

    // Runs straight through for total steps and returns the final state.
    cps::HostParticleState run_straight(const cps::Config& config, long long total) {
        cps::Config local = config;
        local.steps = total;
        cps::SimulationEngine engine(local);
        engine.initialize();
        for (long long i = 0; i < total; ++i) {
            engine.step();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cps::HostParticleState state;
        engine.particles().download(state, engine.stream());
        return state;
    }

    // Runs `first` steps, snapshots, then resumes into a fresh engine and runs
    // the remainder.
    cps::HostParticleState run_with_resume(const cps::Config& config, long long first,
                                           long long second) {
        {
            cps::Config local = config;
            local.steps = first;
            cps::SimulationEngine engine(local);
            engine.initialize();
            for (long long i = 0; i < first; ++i) {
                engine.step();
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            engine.write_snapshot_now();
        }
        cps::Config local = config;
        local.steps = first + second;
        cps::SimulationEngine engine(local);
        const long long at = engine.resume();
        EXPECT_EQ(at, first);
        for (long long i = 0; i < second; ++i) {
            engine.step();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cps::HostParticleState state;
        engine.particles().download(state, engine.stream());
        return state;
    }

    // Compares bit patterns, not values. Float equality would accept -0 for
    // +0 and reject a NaN that came back exactly as it went in, and neither
    // is what "bit identical" means.
    static void expect_bit_identical(const cps::HostParticleState& a,
                                     const cps::HostParticleState& b, const std::string& label) {
        ASSERT_EQ(a.size(), b.size()) << label;
        auto bits = [](float value) { return std::bit_cast<uint32_t>(value); };
        for (std::size_t i = 0; i < a.size(); ++i) {
            ASSERT_EQ(bits(a.x[i]), bits(b.x[i])) << label << " position x at index " << i;
            ASSERT_EQ(bits(a.y[i]), bits(b.y[i])) << label << " position y at index " << i;
            ASSERT_EQ(bits(a.z[i]), bits(b.z[i])) << label << " position z at index " << i;
            ASSERT_EQ(bits(a.vx[i]), bits(b.vx[i])) << label << " velocity x at index " << i;
            ASSERT_EQ(bits(a.vy[i]), bits(b.vy[i])) << label << " velocity y at index " << i;
            ASSERT_EQ(bits(a.vz[i]), bits(b.vz[i])) << label << " velocity z at index " << i;
            ASSERT_EQ(bits(a.mass[i]), bits(b.mass[i])) << label << " mass at index " << i;
        }
    }

    // A few steps under config, then a snapshot in the test directory.
    void write_snapshot_under(const cps::Config& config, int steps) {
        cps::Config local = config;
        local.steps = steps;
        cps::SimulationEngine engine(local);
        engine.initialize();
        for (int i = 0; i < steps; ++i) {
            engine.step();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        engine.write_snapshot_now();
    }

    // The error a resume under config raises, or "" if it succeeds.
    static std::string resume_error(const cps::Config& config) {
        cps::SimulationEngine engine(config);
        try {
            engine.resume();
        } catch (const std::runtime_error& error) {
            return error.what();
        }
        return "";
    }

    fs::path directory_;
};

// The central claim of the snapshot design: stopping and resuming is not
// merely close to running straight through, it is the same run. Anything less
// makes a snapshot useless for a long job, because the results would depend on
// how many times it was interrupted.
TEST_F(ResumeTest, SplitVariantResumesBitIdentically) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Tiled, 2048);
    const cps::HostParticleState straight = run_straight(config, 120);
    const cps::HostParticleState resumed = run_with_resume(config, 50, 70);
    expect_bit_identical(straight, resumed, "tiled");
}

TEST_F(ResumeTest, NaiveVariantResumesBitIdentically) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Naive, 1024);
    const cps::HostParticleState straight = run_straight(config, 100);
    const cps::HostParticleState resumed = run_with_resume(config, 37, 63);
    expect_bit_identical(straight, resumed, "naive");
}

// The fused variant is the interesting case, because its velocity array holds
// the leapfrog half step velocity rather than the synchronised one. The flag
// in the snapshot header is what carries that distinction across the restart.
TEST_F(ResumeTest, FusedVariantResumesBitIdentically) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Fused, 2048);
    const cps::HostParticleState straight = run_straight(config, 120);
    const cps::HostParticleState resumed = run_with_resume(config, 50, 70);
    expect_bit_identical(straight, resumed, "fused");
}

TEST_F(ResumeTest, FluidSolverResumesBitIdentically) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::Sph, cps::NBodyVariant::Tiled, 4096);
    const cps::HostParticleState straight = run_straight(config, 100);
    const cps::HostParticleState resumed = run_with_resume(config, 40, 60);
    expect_bit_identical(straight, resumed, "sph");
}

TEST_F(ResumeTest, ResumingTwiceStillLandsOnTheSameState) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Fused, 1024);
    const cps::HostParticleState straight = run_straight(config, 90);

    {
        cps::Config local = config;
        local.steps = 30;
        cps::SimulationEngine engine(local);
        engine.initialize();
        for (int i = 0; i < 30; ++i) {
            engine.step();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        engine.write_snapshot_now();
    }
    {
        cps::Config local = config;
        local.steps = 60;
        cps::SimulationEngine engine(local);
        engine.resume();
        for (int i = 0; i < 30; ++i) {
            engine.step();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        engine.write_snapshot_now();
    }
    cps::Config local = config;
    local.steps = 90;
    cps::SimulationEngine engine(local);
    engine.resume();
    for (int i = 0; i < 30; ++i) {
        engine.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cps::HostParticleState state;
    engine.particles().download(state, engine.stream());
    expect_bit_identical(straight, state, "twice resumed");
}

// A diagnostic reads the state and must not perturb it. If it did, a run that
// happened to take a diagnostic would diverge from one that did not, and the
// energy curve would be measuring its own observer.
TEST_F(ResumeTest, TakingDiagnosticsDoesNotPerturbTheTrajectory) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Fused, 1024);
    const cps::HostParticleState quiet = run_straight(config, 80);

    cps::Config local = config;
    local.steps = 80;
    cps::SimulationEngine engine(local);
    engine.initialize();
    for (int i = 0; i < 80; ++i) {
        engine.step();
        engine.diagnostics(true);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cps::HostParticleState observed;
    engine.particles().download(observed, engine.stream());
    expect_bit_identical(quiet, observed, "observed");
}

TEST_F(ResumeTest, TheSnapshotOnDiskCarriesTheStepAndTheVelocityConvention) {
    SKIP_WITHOUT_GPU();
    cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Fused, 256);
    config.steps = 25;
    cps::SimulationEngine engine(config);
    engine.initialize();
    for (int i = 0; i < 25; ++i) {
        engine.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    engine.write_snapshot_now();

    const fs::path path = cps::latest_snapshot(directory_);
    ASSERT_FALSE(path.empty());
    const cps::Snapshot snapshot = cps::read_snapshot(path);
    EXPECT_EQ(snapshot.step, 25u);
    EXPECT_TRUE((snapshot.flags & cps::SNAPSHOT_FLAG_HALF_STEP_VELOCITY) != 0u);
    EXPECT_EQ(snapshot.state.size(), 256u);
}

TEST_F(ResumeTest, RefusesASnapshotWithTheWrongParticleCount) {
    SKIP_WITHOUT_GPU();
    cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Tiled, 256);
    config.steps = 5;
    {
        cps::SimulationEngine engine(config);
        engine.initialize();
        engine.step();
        CUDA_CHECK(cudaDeviceSynchronize());
        engine.write_snapshot_now();
    }
    cps::Config mismatched = config;
    mismatched.num_particles = 512;
    cps::SimulationEngine engine(mismatched);
    EXPECT_THROW(engine.resume(), std::runtime_error);
}

TEST_F(ResumeTest, ReportsAMissingSnapshotDirectoryClearly) {
    SKIP_WITHOUT_GPU();
    cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Tiled, 128);
    config.snapshot_dir = (directory_ / "does_not_exist").string();
    cps::SimulationEngine engine(config);
    EXPECT_THROW(engine.resume(), std::runtime_error);
}

// Resuming under different physics continues a different simulation from a
// state the new one never produced. Each of these must be refused, and the
// message must say which field moved and from what to what.
TEST_F(ResumeTest, RefusesADifferentTimeStep) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Tiled, 256);
    write_snapshot_under(config, 3);
    cps::Config changed = config;
    changed.dt = 2.0e-3f;
    const std::string message = resume_error(changed);
    EXPECT_NE(message.find("dt is 0.001 in the snapshot and 0.002 in the configuration"),
              std::string::npos)
        << message;
    EXPECT_NE(message.find("--force-resume"), std::string::npos) << message;
}

TEST_F(ResumeTest, RefusesADifferentSolver) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Tiled, 512);
    write_snapshot_under(config, 3);
    const cps::Config changed = base_config(cps::SolverType::Sph, cps::NBodyVariant::Tiled, 512);
    const std::string message = resume_error(changed);
    EXPECT_NE(message.find("solver is nbody in the snapshot and sph in the configuration"),
              std::string::npos)
        << message;
}

TEST_F(ResumeTest, RefusesADifferentVariant) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Fused, 256);
    write_snapshot_under(config, 3);
    cps::Config changed = config;
    changed.variant = cps::NBodyVariant::Naive;
    const std::string message = resume_error(changed);
    EXPECT_NE(message.find("variant is fused in the snapshot and naive in the configuration"),
              std::string::npos)
        << message;
}

TEST_F(ResumeTest, RefusesADifferentSoftening) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Tiled, 256);
    write_snapshot_under(config, 3);
    cps::Config changed = config;
    changed.softening = 0.01f;
    const std::string message = resume_error(changed);
    EXPECT_NE(message.find("softening is 0.05 in the snapshot and 0.01 in the configuration"),
              std::string::npos)
        << message;
}

TEST_F(ResumeTest, RefusesADifferentFluidParameter) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::Sph, cps::NBodyVariant::Tiled, 512);
    write_snapshot_under(config, 3);
    cps::Config changed = config;
    changed.sph.viscosity = 0.1f;
    const std::string message = resume_error(changed);
    EXPECT_NE(message.find("sph.viscosity is 0.2 in the snapshot and 0.1 in the configuration"),
              std::string::npos)
        << message;
}

// Run control is not physics: a longer run, another cadence or another
// diagnostics file must resume without complaint.
TEST_F(ResumeTest, AcceptsAChangeToRunControlOnly) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Tiled, 256);
    write_snapshot_under(config, 3);
    cps::Config changed = config;
    changed.steps = 1000;
    changed.name = "longer";
    changed.snapshot_interval = 17;
    changed.block_size = 64;
    changed.seed = 7u;
    EXPECT_EQ(resume_error(changed), "");
}

TEST_F(ResumeTest, ForceResumeAcceptsADifferentTimeStepWithAWarning) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Tiled, 256);
    write_snapshot_under(config, 3);
    cps::Config changed = config;
    changed.dt = 2.0e-3f;
    cps::SimulationEngine engine(changed);
    long long at = -1;
    ::testing::internal::CaptureStderr();
    EXPECT_NO_THROW(at = engine.resume(true));
    const std::string warning = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(at, 3);
    EXPECT_NE(warning.find("warning:"), std::string::npos) << warning;
    EXPECT_NE(warning.find("dt is 0.001 in the snapshot and 0.002 in the configuration"),
              std::string::npos)
        << warning;
    EXPECT_NO_THROW(engine.step());
}

// The committed CPS1 fixture (see tests/unit/test_snapshot_format.cpp for how
// it is made) has no physics record. It must still load, with a warning that
// nothing could be checked, and the run must carry on from its step.
TEST_F(ResumeTest, ResumesTheCps1FixtureWithAWarning) {
    SKIP_WITHOUT_GPU();
    const fs::path fixture =
        fs::path(CPS_PROJECT_ROOT) / "tests" / "data" / "snapshot_v1_4p.bin";
    ASSERT_TRUE(fs::exists(fixture)) << fixture;
    fs::copy_file(fixture, cps::snapshot_path_for_step(directory_, 42u));

    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Tiled, 4);
    cps::SimulationEngine engine(config);
    long long at = -1;
    ::testing::internal::CaptureStderr();
    EXPECT_NO_THROW(at = engine.resume());
    const std::string warning = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(at, 42);
    EXPECT_NE(warning.find("CPS1"), std::string::npos) << warning;
    EXPECT_NE(warning.find("cannot be verified"), std::string::npos) << warning;

    for (int i = 0; i < 5; ++i) {
        engine.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(engine.current_step(), 47);
    cps::HostParticleState state;
    engine.particles().download(state, engine.stream());
    ASSERT_EQ(state.size(), 4u);
    for (std::size_t i = 0; i < state.size(); ++i) {
        EXPECT_TRUE(std::isfinite(state.x[i]) && std::isfinite(state.vx[i])) << "index " << i;
    }
}

// Version 2 stores the engine's simulated time next to the step.
TEST_F(ResumeTest, TheSnapshotRecordsTheSimulatedTime) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = base_config(cps::SolverType::NBody, cps::NBodyVariant::Tiled, 128);
    write_snapshot_under(config, 7);
    const cps::Snapshot snapshot = cps::read_snapshot(cps::latest_snapshot(directory_));
    EXPECT_EQ(snapshot.format_version, 2u);
    EXPECT_EQ(std::bit_cast<uint64_t>(snapshot.time),
              std::bit_cast<uint64_t>(7.0 * static_cast<double>(config.dt)));
}

}  // namespace
