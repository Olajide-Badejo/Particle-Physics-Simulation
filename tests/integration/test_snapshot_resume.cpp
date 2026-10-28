#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "core/simulation_engine.hpp"
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
        engine.particles().download(state);
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
        engine.particles().download(state);
        return state;
    }

    static void expect_bit_identical(const cps::HostParticleState& a,
                                     const cps::HostParticleState& b, const std::string& label) {
        ASSERT_EQ(a.size(), b.size()) << label;
        for (std::size_t i = 0; i < a.size(); ++i) {
            ASSERT_EQ(a.x[i], b.x[i]) << label << " position x at index " << i;
            ASSERT_EQ(a.y[i], b.y[i]) << label << " position y at index " << i;
            ASSERT_EQ(a.z[i], b.z[i]) << label << " position z at index " << i;
            ASSERT_EQ(a.vx[i], b.vx[i]) << label << " velocity x at index " << i;
            ASSERT_EQ(a.vy[i], b.vy[i]) << label << " velocity y at index " << i;
            ASSERT_EQ(a.vz[i], b.vz[i]) << label << " velocity z at index " << i;
            ASSERT_EQ(a.mass[i], b.mass[i]) << label << " mass at index " << i;
        }
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
    engine.particles().download(state);
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
    engine.particles().download(observed);
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

}  // namespace
