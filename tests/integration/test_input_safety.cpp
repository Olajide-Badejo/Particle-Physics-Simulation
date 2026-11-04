#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/core/simulation_engine.hpp"
#include "cps/physics/uniform_grid.hpp"
#include "cps/solvers/sph_passes.hpp"
#include "cps/solvers/sph_solver.hpp"
#include "test_support_gpu.hpp"

namespace fs = std::filesystem;

namespace {

constexpr float NOT_A_NUMBER = std::numeric_limits<float>::quiet_NaN();
constexpr float INFINITY_F = std::numeric_limits<float>::infinity();

cps::Config sph_config(int count, long long steps) {
    cps::Config config;
    config.name = "input_safety_sph";
    config.solver = cps::SolverType::Sph;
    config.initial_condition = cps::InitialCondition::DamBreak;
    config.num_particles = count;
    config.steps = steps;
    config.dt = 1.0e-4f;
    config.block_size = 128;
    config.external_gravity = {0.0f, -9.81f, 0.0f};
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    config.sph.domain_min = {0.0f, 0.0f, 0.0f};
    config.sph.domain_max = {0.8f, 0.5f, 0.4f};
    config.sph.fluid_min = {0.0f, 0.0f, 0.0f};
    config.sph.fluid_max = {0.3f, 0.4f, 0.4f};
    return config;
}

cps::Config nbody_config(cps::NBodyVariant variant, int count, long long steps) {
    cps::Config config;
    config.name = "input_safety_" + cps::to_string(variant);
    config.solver = cps::SolverType::NBody;
    config.variant = variant;
    config.initial_condition = cps::InitialCondition::Plummer;
    config.num_particles = count;
    config.steps = steps;
    config.dt = 1.0e-3f;
    config.softening = 0.05f;
    config.gravitational_constant = 1.0f;
    config.block_size = 128;
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    return config;
}

// Overwrites one value in the device state. edit gets the host copy.
void edit_state(cps::SimulationEngine& engine,
                const std::function<void(cps::HostParticleState&)>& edit) {
    cps::HostParticleState state;
    engine.particles().download(state, engine.stream());
    edit(state);
    engine.particles().upload(state, engine.stream());
}

// The message of the std::runtime_error the call has to raise.
std::string error_from(const std::function<void()>& call) {
    try {
        call();
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    ADD_FAILURE() << "expected a std::runtime_error";
    return "";
}

bool has_text(const std::string& message, const std::string& text) {
    return message.find(text) != std::string::npos;
}

int count_non_finite_particles(const cps::HostParticleState& state) {
    int bad = 0;
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (!(std::isfinite(state.x[i]) && std::isfinite(state.y[i]) &&
              std::isfinite(state.z[i]) && std::isfinite(state.vx[i]) &&
              std::isfinite(state.vy[i]) && std::isfinite(state.vz[i]))) {
            ++bad;
        }
    }
    return bad;
}

std::string read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

int count_files(const fs::path& directory) {
    int files = 0;
    for ([[maybe_unused]] const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        ++files;
    }
    return files;
}

class InputSafetyTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = fs::temp_directory_path() / ("cps_input_safety_" + std::to_string(stamp));
        fs::create_directories(directory_);
    }

    void TearDown() override {
        std::error_code ignored;
        fs::remove_all(directory_, ignored);
    }

    std::vector<fs::path> files_in_directory() const {
        std::vector<fs::path> files;
        for (const fs::directory_entry& entry : fs::directory_iterator(directory_)) {
            files.push_back(entry.path());
        }
        return files;
    }

    fs::path directory_;
};

// One cell pass with bad positions planted among good ones: the good ones get
// exactly the cell the shared formulas give, the bad ones get the reserved slot.
TEST(InputSafetyGrid, NonFinitePositionsGetTheReservedSlotAndFinitePositionsDoNotChange) {
    SKIP_WITHOUT_GPU();
    const int count = 1024;
    const cps::UniformGrid grid =
        cps::make_uniform_grid(-1.0f, -1.0f, 0.0f, 2.0f, 2.5f, 5.0f, 0.037f);

    std::vector<float> x(count), y(count), z(count);
    for (int i = 0; i < count; ++i) {
        x[i] = -1.0f + 0.0021f * static_cast<float>(i);
        y[i] = 2.0f - 0.0013f * static_cast<float>(i);
        z[i] = 0.0007f * static_cast<float>(i * (i % 7));
    }
    x[3] = NOT_A_NUMBER;
    y[500] = INFINITY_F;
    z[1023] = -INFINITY_F;

    cps::DeviceBuffer<float> dx(count), dy(count), dz(count);
    cps::DeviceBuffer<uint32_t> cell(count), index(count);
    cps::DeviceBuffer<int> flag(1);
    flag.zero();
    CUDA_CHECK(cudaMemcpy(dx.get(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy.get(), y.data(), dy.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz.get(), z.data(), dz.bytes(), cudaMemcpyHostToDevice));

    cps::sph_assign_cells(dx.get(), dy.get(), dz.get(), cell.get(), index.get(), flag.get(), count,
                          grid, 128, nullptr);

    std::vector<uint32_t> host_cell(count), host_index(count);
    int host_flag = 0;
    CUDA_CHECK(cudaMemcpy(host_cell.data(), cell.get(), cell.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_index.data(), index.get(), index.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&host_flag, flag.get(), sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(host_flag, 1);
    for (int i = 0; i < count; ++i) {
        ASSERT_EQ(host_index[i], static_cast<uint32_t>(i));
        if (i == 3 || i == 500 || i == 1023) {
            EXPECT_EQ(host_cell[i], cps::reserved_slot(grid)) << "index " << i;
            continue;
        }
        ASSERT_LT(host_cell[i], grid.num_cells) << "index " << i;
        ASSERT_EQ(host_cell[i], cps::cell_index(grid, x[i], y[i], z[i])) << "index " << i;
    }
}

TEST(InputSafetySph, OneBadPositionIsReportedAfterTheNextStep) {
    SKIP_WITHOUT_GPU();
    struct Case {
        const char* label;
        float value;
    };
    for (const Case& bad : {Case{"NaN", NOT_A_NUMBER}, Case{"Inf", INFINITY_F},
                            Case{"minus Inf", -INFINITY_F}}) {
        SCOPED_TRACE(bad.label);
        cps::SimulationEngine engine(sph_config(2000, 0));
        engine.initialize();
        edit_state(engine, [&](cps::HostParticleState& state) { state.y[117] = bad.value; });
        EXPECT_NO_THROW(engine.check_device_errors()) << "nothing has hashed the position yet";

        engine.step();
        const std::string message = error_from([&] { engine.check_device_errors(); });
        EXPECT_TRUE(has_text(message, "non-finite positions")) << message;
        EXPECT_TRUE(has_text(message, "by step 1:")) << message;
        EXPECT_TRUE(has_text(message, "too large a time step")) << message;
    }
}

// Every pass of the step has to survive the sentinel. Stepping on and then
// reaching a device synchronisation shows that no launch faulted, and the
// step named follows the engine.
TEST(InputSafetySph, TheStepKeepsRunningSafelyWithAParticleInTheSentinelCell) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(sph_config(2000, 0));
    engine.initialize();
    edit_state(engine, [](cps::HostParticleState& state) {
        state.x[0] = NOT_A_NUMBER;
        state.y[1] = INFINITY_F;
        state.z[2] = -INFINITY_F;
    });
    for (int i = 0; i < 25; ++i) {
        engine.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::string message = error_from([&] { engine.check_device_errors(); });
    EXPECT_TRUE(has_text(message, "by step 25:")) << message;
}

// Every position bad at once puts the whole system into the reserved slot, which
// is the longest run the cell bounds pass can see, and leaves every real cell empty.
TEST(InputSafetySph, EveryPositionNonFiniteStaysInBounds) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(sph_config(2000, 0));
    engine.initialize();
    edit_state(engine, [](cps::HostParticleState& state) {
        for (std::size_t i = 0; i < state.size(); ++i) {
            state.x[i] = NOT_A_NUMBER;
        }
    });
    engine.step();
    engine.step();
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_FALSE(error_from([&] { engine.check_device_errors(); }).empty());
}

// The reserved slot is what stops the failure spreading: no finite particle
// looks in it, so its neighbours keep a finite state.
TEST(InputSafetySph, ANaNParticleDoesNotContaminateItsNeighbours) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(sph_config(2000, 0));
    engine.initialize();
    edit_state(engine, [](cps::HostParticleState& state) { state.x[1000] = NOT_A_NUMBER; });
    engine.step();
    CUDA_CHECK(cudaDeviceSynchronize());
    cps::HostParticleState state;
    engine.particles().download(state, engine.stream());
    EXPECT_EQ(count_non_finite_particles(state), 1);
}

// The wall clamp puts an infinite position straight back on the wall, so one
// step later nothing in the state shows it happened. Only the flag remembers,
// which is why it stays set.
TEST(InputSafetySph, TheFlagRemembersAnInfinityTheWallsHaveAlreadyHealed) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(sph_config(2000, 0));
    engine.initialize();
    edit_state(engine, [](cps::HostParticleState& state) { state.x[1000] = INFINITY_F; });
    engine.step();
    CUDA_CHECK(cudaDeviceSynchronize());
    cps::HostParticleState state;
    engine.particles().download(state, engine.stream());
    EXPECT_EQ(count_non_finite_particles(state), 0);
    EXPECT_TRUE(has_text(error_from([&] { engine.check_device_errors(); }), "by step 1:"));
    EXPECT_TRUE(has_text(error_from([&] { engine.diagnostics(); }), "by step 1:"));
    EXPECT_TRUE(has_text(error_from([&] { engine.write_snapshot_now(); }), "by step 1:"));
}

TEST(InputSafetySph, AHealthyRunLeavesTheFlagClear) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = sph_config(2000, 200);
    cps::SimulationEngine engine(config);
    engine.initialize();
    for (int i = 0; i < 200; ++i) {
        engine.step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_NO_THROW(engine.check_device_errors());
    EXPECT_NO_THROW(engine.diagnostics());
}

// The solver on its own: clear from the start, clear after healthy steps, and
// set by the first bad position it meets.
TEST(InputSafetySph, TheSolverFlagReadsClearUntilABadPositionIsHashed) {
    SKIP_WITHOUT_GPU();
    const cps::Config config = sph_config(2000, 0);
    cps::HostParticleState state = cps::make_initial_state(config);
    const cps::Stream stream;
    cps::ParticleSystem particles(config.num_particles);
    particles.upload(state, stream.get());

    cps::SphSolver solver(config, config.num_particles);
    EXPECT_FALSE(solver.non_finite_position_seen(stream.get()));
    for (int i = 0; i < 200; ++i) {
        solver.step(particles, stream.get());
    }
    EXPECT_FALSE(solver.non_finite_position_seen(stream.get()));

    particles.download(state, stream.get());
    state.z[3] = NOT_A_NUMBER;
    particles.upload(state, stream.get());
    solver.step(particles, stream.get());
    EXPECT_TRUE(solver.non_finite_position_seen(stream.get()));
}

TEST(InputSafetySph, TheSolverRefusesADomainTooLargeForItsCellSize) {
    SKIP_WITHOUT_GPU();
    cps::Config config = sph_config(500, 0);
    config.sph.domain_max.z = 1.0e9f;
    const std::string message = error_from([&] { cps::SimulationEngine engine(config); });
    EXPECT_TRUE(has_text(message, "axis z")) << message;
    EXPECT_TRUE(has_text(message, "raise the smoothing radius")) << message;
}

// The check itself is tested on the host. These show the solver applies it.
TEST(InputSafetySph, TheSolverRefusesATooLargeTimeStep) {
    SKIP_WITHOUT_GPU();
    cps::Config config = sph_config(50000, 0);
    config.sph.gas_constant = 1.0e4f;
    const std::string message = error_from([&] { cps::SimulationEngine engine(config); });
    EXPECT_TRUE(has_text(message, "acoustic limit")) << message;
    EXPECT_TRUE(has_text(message, "--allow-unstable-dt")) << message;
}

TEST(InputSafetySph, TheFlagLetsTheSolverRunATooLargeTimeStep) {
    SKIP_WITHOUT_GPU();
    cps::Config config = sph_config(50000, 3);
    config.sph.gas_constant = 1.0e4f;
    config.allow_unstable_dt = true;
    ::testing::internal::CaptureStderr();
    cps::SimulationEngine engine(config);
    const std::string warning = ::testing::internal::GetCapturedStderr();
    EXPECT_TRUE(has_text(warning, "acoustic limit")) << warning;
    engine.initialize();
    EXPECT_NO_THROW(engine.step());
}

// With both cadences off the only checks left are the ones at the end of run().
TEST(InputSafetySph, RunChecksTheFlagAtTheEndWhenNothingElseDoes) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(sph_config(2000, 12));
    engine.initialize();
    edit_state(engine, [](cps::HostParticleState& state) { state.x[5] = NOT_A_NUMBER; });
    const std::string message = error_from([&] { engine.run(); });
    EXPECT_TRUE(has_text(message, "by step 12:")) << message;
}

// The cell pass reads a finite position, and only the update at the end of the
// step turns it into a NaN, so the flag stays clear and the state scan is the
// one check that can see it.
TEST(InputSafetyEndOfRun, ABadStateBornInTheFinalStepIsCaught) {
    SKIP_WITHOUT_GPU();
    const cps::Config configs[] = {
        sph_config(2000, 1),
        nbody_config(cps::NBodyVariant::Naive, 256, 1),
        nbody_config(cps::NBodyVariant::Tiled, 256, 1),
        nbody_config(cps::NBodyVariant::Fused, 256, 1),
    };
    for (const cps::Config& config : configs) {
        SCOPED_TRACE(config.name);
        cps::SimulationEngine engine(config);
        engine.initialize();
        edit_state(engine, [](cps::HostParticleState& state) { state.vx[11] = NOT_A_NUMBER; });
        const std::string message = error_from([&] { engine.run(); });
        EXPECT_TRUE(has_text(message, "at step 1:")) << message;
        EXPECT_TRUE(has_text(message, "particle ")) << message;
        EXPECT_TRUE(has_text(message, "non-finite")) << message;
    }
}

// N-body has no flag, so with both cadences off this scan is its only check.
TEST(InputSafetyEndOfRun, AnNBodyRunWithABadPositionThrows) {
    SKIP_WITHOUT_GPU();
    for (const cps::NBodyVariant variant :
         {cps::NBodyVariant::Naive, cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
        SCOPED_TRACE(cps::to_string(variant));
        cps::SimulationEngine engine(nbody_config(variant, 256, 3));
        engine.initialize();
        edit_state(engine, [](cps::HostParticleState& state) { state.x[40] = NOT_A_NUMBER; });
        const std::string message = error_from([&] { engine.run(); });
        EXPECT_TRUE(has_text(message, "at step 3:")) << message;
        EXPECT_TRUE(has_text(message, "non-finite")) << message;
    }
}

TEST(InputSafetyEndOfRun, AHealthyRunWithBothCadencesOffCompletes) {
    SKIP_WITHOUT_GPU();
    const cps::Config configs[] = {
        sph_config(2000, 20),
        nbody_config(cps::NBodyVariant::Naive, 256, 20),
        nbody_config(cps::NBodyVariant::Tiled, 256, 20),
        nbody_config(cps::NBodyVariant::Fused, 256, 20),
    };
    for (const cps::Config& config : configs) {
        SCOPED_TRACE(config.name);
        cps::SimulationEngine engine(config);
        engine.initialize();
        EXPECT_NO_THROW(engine.run());
        EXPECT_EQ(engine.current_step(), 20);
        EXPECT_EQ(engine.steps_run(), 20);
        EXPECT_GE(engine.wall_milliseconds(), engine.solver_milliseconds());
    }
}

TEST_F(InputSafetyTest, RunStopsAtTheDiagnosticsIntervalAndWritesNoBadRow) {
    SKIP_WITHOUT_GPU();
    const fs::path csv = directory_ / "diagnostics.csv";
    {
        cps::Config config = sph_config(2000, 50);
        config.diagnostics_interval = 5;
        config.diagnostics_path = csv.string();
        cps::SimulationEngine engine(config);
        engine.initialize();
        edit_state(engine, [](cps::HostParticleState& state) { state.x[5] = NOT_A_NUMBER; });
        const std::string message = error_from([&] { engine.run(); });
        EXPECT_TRUE(has_text(message, "by step 5:")) << message;
    }
    const std::string text = read_file(csv);
    EXPECT_TRUE(has_text(text, "\n0,")) << text;
    EXPECT_FALSE(has_text(text, "\n5,")) << text;
    EXPECT_FALSE(has_text(text, "nan")) << text;
}

TEST(InputSafetySph, DiagnosticsCatchABadVelocityThatTheHashCannotSee) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(sph_config(2000, 0));
    engine.initialize();
    engine.step();
    edit_state(engine, [](cps::HostParticleState& state) { state.vy[9] = NOT_A_NUMBER; });
    const std::string message = error_from([&] { engine.diagnostics(false); });
    EXPECT_TRUE(has_text(message, "non-finite diagnostics at step 1:")) << message;
}

TEST(InputSafetyNBody, DiagnosticsThrowNamingTheStep) {
    SKIP_WITHOUT_GPU();
    for (const cps::NBodyVariant variant :
         {cps::NBodyVariant::Naive, cps::NBodyVariant::Tiled, cps::NBodyVariant::Fused}) {
        SCOPED_TRACE(cps::to_string(variant));
        cps::SimulationEngine engine(nbody_config(variant, 256, 0));
        engine.initialize();
        edit_state(engine, [](cps::HostParticleState& state) { state.x[40] = NOT_A_NUMBER; });

        // The potential sum reads the positions directly.
        std::string message = error_from([&] { engine.diagnostics(); });
        EXPECT_TRUE(has_text(message, "non-finite diagnostics at step 0:")) << message;

        // After a step the force pass has carried it into the velocities, so
        // the row is bad without the potential too.
        engine.step();
        message = error_from([&] { engine.diagnostics(false); });
        EXPECT_TRUE(has_text(message, "non-finite diagnostics at step 1:")) << message;
    }
}

TEST(InputSafetyNBody, AHealthyStateRaisesNothing) {
    SKIP_WITHOUT_GPU();
    cps::SimulationEngine engine(nbody_config(cps::NBodyVariant::Tiled, 256, 0));
    engine.initialize();
    for (int i = 0; i < 20; ++i) {
        engine.step();
    }
    EXPECT_NO_THROW(engine.check_device_errors());
    EXPECT_NO_THROW(engine.diagnostics());
}

// Both solvers: a healthy snapshot goes first, then a snapshot falls due on a
// blown-up state. The run stops, the directory still holds exactly the one
// file, unchanged, and a resume can go back to it.
TEST_F(InputSafetyTest, ASnapshotDueOnABadStateThrowsAndLeavesTheOlderOneAlone) {
    SKIP_WITHOUT_GPU();
    struct Case {
        const char* label;
        cps::Config config;
    };
    const Case cases[] = {
        {"sph", sph_config(2000, 50)},
        {"tiled", nbody_config(cps::NBodyVariant::Tiled, 256, 50)},
        {"fused", nbody_config(cps::NBodyVariant::Fused, 256, 50)},
    };
    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.label);
        const fs::path run_directory = directory_ / test_case.label;
        cps::Config config = test_case.config;
        config.snapshot_dir = run_directory.string();
        config.snapshot_interval = 5;

        cps::SimulationEngine engine(config);
        engine.initialize();
        engine.write_snapshot_now();
        ASSERT_EQ(count_files(run_directory), 1);
        const fs::path older = fs::directory_iterator(run_directory)->path();
        const std::string older_bytes = read_file(older);
        const fs::file_time_type older_time = fs::last_write_time(older);

        edit_state(engine, [](cps::HostParticleState& state) { state.x[7] = NOT_A_NUMBER; });
        const std::string message = error_from([&] { engine.run(); });
        EXPECT_TRUE(has_text(message, "non-finite")) << message;
        EXPECT_TRUE(has_text(message, "step 5")) << message;

        EXPECT_EQ(count_files(run_directory), 1);
        EXPECT_EQ(fs::directory_iterator(run_directory)->path(), older);
        EXPECT_EQ(read_file(older), older_bytes);
        EXPECT_EQ(fs::last_write_time(older), older_time);

        cps::SimulationEngine resumed(config);
        EXPECT_EQ(resumed.resume(), 0);
    }
}

// The scan is the check that names the particle, so it is driven directly,
// on a state where the fluid solver's flag has nothing to say.
TEST_F(InputSafetyTest, TheSnapshotScanNamesTheStepAndTheFirstBadParticle) {
    SKIP_WITHOUT_GPU();
    struct Case {
        const char* label;
        std::function<void(cps::HostParticleState&)> edit;
    };
    const Case cases[] = {
        {"x", [](cps::HostParticleState& state) { state.x[17] = NOT_A_NUMBER; }},
        {"y", [](cps::HostParticleState& state) { state.y[17] = INFINITY_F; }},
        {"z", [](cps::HostParticleState& state) { state.z[17] = -INFINITY_F; }},
        {"vx", [](cps::HostParticleState& state) { state.vx[17] = NOT_A_NUMBER; }},
        {"vy", [](cps::HostParticleState& state) { state.vy[17] = INFINITY_F; }},
        {"vz", [](cps::HostParticleState& state) { state.vz[17] = -INFINITY_F; }},
    };
    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.label);
        for (const cps::SolverType solver : {cps::SolverType::NBody, cps::SolverType::Sph}) {
            cps::Config config = solver == cps::SolverType::Sph
                                     ? sph_config(2000, 0)
                                     : nbody_config(cps::NBodyVariant::Tiled, 256, 0);
            config.snapshot_dir = directory_.string();
            cps::SimulationEngine engine(config);
            engine.initialize();
            edit_state(engine, [&](cps::HostParticleState& state) {
                test_case.edit(state);
                state.vz[200] = NOT_A_NUMBER;  // a later bad particle must not win
            });
            const std::string message = error_from([&] { engine.write_snapshot_now(); });
            EXPECT_TRUE(has_text(message, "at step 0:")) << message;
            EXPECT_TRUE(has_text(message, "particle 17 ")) << message;
            EXPECT_TRUE(has_text(message, "non-finite")) << message;
            EXPECT_TRUE(files_in_directory().empty());
        }
    }
}

}  // namespace
