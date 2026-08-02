#include "core/simulation_engine.hpp"

#include <cstdio>
#include <stdexcept>

#include "core/initial_conditions.hpp"
#include "core/snapshot.hpp"
#include "solvers/nbody_solver.hpp"
#include "utils/progress.hpp"

namespace fs = std::filesystem;

namespace cps {
namespace {

// Which velocity convention a solver carries between steps. The split velocity
// Verlet keeps the velocity synchronised with the positions; the fused kernel
// and the fluid solver both advance a leapfrog half step velocity. Resuming a
// snapshot written under one convention into a solver that wants the other is
// the one case where the loaded velocity has to be shifted, and getting that
// wrong offsets the whole trajectory by half a step in a way that looks like
// a subtly wrong force.
bool solver_wants_half_step(const Config& config) {
    if (config.solver == SolverType::Sph) {
        return true;
    }
    return config.variant == NBodyVariant::Fused;
}

}  // namespace

SimulationEngine::SimulationEngine(Config config) : config_(std::move(config)) {
    config_.validate();
    particles_ = std::make_unique<ParticleSystem>(config_.num_particles);
    reducer_ = std::make_unique<DiagnosticsReducer>(config_.num_particles);
    if (config_.solver == SolverType::Sph) {
        sph_ = std::make_unique<SphSolver>(config_, config_.num_particles);
    }
    if (!config_.diagnostics_path.empty() && config_.diagnostics_interval > 0) {
        const fs::path path(config_.diagnostics_path);
        if (!path.parent_path().empty()) {
            fs::create_directories(path.parent_path());
        }
        writer_ = std::make_unique<DiagnosticsWriter>(config_.diagnostics_path);
    }
}

SimulationEngine::~SimulationEngine() = default;

void SimulationEngine::initialize() {
    const HostParticleState state = make_initial_state(config_);
    particles_->upload(state);
    current_step_ = 0;
    half_step_velocity_ = false;

    if (config_.solver == SolverType::NBody) {
        // Seed the acceleration array. Velocity Verlet needs a_0 before its
        // first half kick, and the fused leapfrog needs it to step the
        // velocity back by half a step.
        compute_forces_tiled(particles_->x(), particles_->y(), particles_->z(), particles_->mass(),
                             particles_->ax(), particles_->ay(), particles_->az(),
                             particles_->size(), config_.softening, config_.gravitational_constant,
                             config_.block_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // For the fluid case the acceleration array is still zero here, so this
    // records the convention without touching a single velocity.
    ensure_velocity_convention(solver_wants_half_step(config_));
    initialized_ = true;
}

void SimulationEngine::ensure_velocity_convention(bool want_half_step) {
    if (want_half_step == half_step_velocity_) {
        return;
    }
    // Stepping to the half step velocity means going backwards half a step;
    // coming back out of it means going forwards.
    const float dt = want_half_step ? -config_.dt : config_.dt;
    offset_velocity_half_step(particles_->vx(), particles_->vy(), particles_->vz(),
                              particles_->ax(), particles_->ay(), particles_->az(),
                              particles_->size(), dt, config_.block_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    half_step_velocity_ = want_half_step;
}

long long SimulationEngine::resume() {
    const fs::path path = latest_snapshot(config_.snapshot_dir);
    if (path.empty()) {
        throw std::runtime_error("no snapshot found in " + config_.snapshot_dir);
    }
    const Snapshot snapshot = read_snapshot(path);
    if (snapshot.state.size() != static_cast<std::size_t>(config_.num_particles)) {
        throw std::runtime_error("snapshot " + path.string() + " holds " +
                                 std::to_string(snapshot.state.size()) +
                                 " particles but the configuration asks for " +
                                 std::to_string(config_.num_particles));
    }

    particles_->upload(snapshot.state);
    current_step_ = static_cast<long long>(snapshot.step);
    half_step_velocity_ = (snapshot.flags & SNAPSHOT_FLAG_HALF_STEP_VELOCITY) != 0u;

    if (config_.solver == SolverType::NBody) {
        // Accelerations are a deterministic function of the positions, so
        // recomputing them here reproduces exactly the array the original run
        // held at this step. That is what makes the resumed trajectory bit
        // identical rather than merely close.
        compute_forces_tiled(particles_->x(), particles_->y(), particles_->z(), particles_->mass(),
                             particles_->ax(), particles_->ay(), particles_->az(),
                             particles_->size(), config_.softening, config_.gravitational_constant,
                             config_.block_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ensure_velocity_convention(solver_wants_half_step(config_));
    initialized_ = true;
    std::fprintf(stderr, "resumed from %s at step %lld\n", path.string().c_str(), current_step_);
    return current_step_;
}

void SimulationEngine::step_nbody_split() {
    integrate_verlet_position(particles_->x(), particles_->y(), particles_->z(), particles_->vx(),
                              particles_->vy(), particles_->vz(), particles_->ax(),
                              particles_->ay(), particles_->az(), particles_->size(), config_.dt,
                              config_.block_size);

    if (config_.variant == NBodyVariant::Naive) {
        compute_forces_naive(particles_->x(), particles_->y(), particles_->z(), particles_->mass(),
                             particles_->ax(), particles_->ay(), particles_->az(),
                             particles_->size(), config_.softening, config_.gravitational_constant,
                             config_.block_size);
    } else {
        compute_forces_tiled(particles_->x(), particles_->y(), particles_->z(), particles_->mass(),
                             particles_->ax(), particles_->ay(), particles_->az(),
                             particles_->size(), config_.softening, config_.gravitational_constant,
                             config_.block_size);
    }

    integrate_verlet_velocity(particles_->vx(), particles_->vy(), particles_->vz(),
                              particles_->ax(), particles_->ay(), particles_->az(),
                              particles_->size(), config_.dt, config_.block_size);
}

void SimulationEngine::step_nbody_fused() {
    nbody_fused_step(particles_->x(), particles_->y(), particles_->z(), particles_->x_next(),
                     particles_->y_next(), particles_->z_next(), particles_->vx(),
                     particles_->vy(), particles_->vz(), particles_->ax(), particles_->ay(),
                     particles_->az(), particles_->mass(), particles_->size(), config_.softening,
                     config_.gravitational_constant, config_.dt, config_.block_size);
    particles_->swap_position_buffers();
}

void SimulationEngine::step() {
    if (!initialized_) {
        throw std::runtime_error("the engine must be initialised or resumed before stepping");
    }
    if (config_.solver == SolverType::Sph) {
        sph_->step(*particles_, config_.dt);
    } else if (config_.variant == NBodyVariant::Fused) {
        step_nbody_fused();
    } else {
        step_nbody_split();
    }
    ++current_step_;
}

Diagnostics SimulationEngine::diagnostics(bool include_potential) {
    const float half_dt = half_step_velocity_ ? 0.5f * config_.dt : 0.0f;
    const bool with_potential = include_potential && config_.solver == SolverType::NBody;
    return reducer_->compute(*particles_, current_step_,
                             static_cast<double>(current_step_) * static_cast<double>(config_.dt),
                             config_.softening, config_.gravitational_constant, with_potential,
                             half_dt);
}

void SimulationEngine::write_snapshot_now() const {
    Snapshot snapshot;
    snapshot.step = static_cast<uint32_t>(current_step_);
    snapshot.flags = half_step_velocity_ ? SNAPSHOT_FLAG_HALF_STEP_VELOCITY : 0u;
    particles_->download(snapshot.state);
    write_snapshot(snapshot_path_for_step(config_.snapshot_dir, snapshot.step), snapshot);
}

void SimulationEngine::run() {
    if (!initialized_) {
        initialize();
    }

    if (writer_ && writer_->is_open()) {
        writer_->write(diagnostics());
    }

    ProgressBar bar(config_.name.empty() ? std::string("simulation") : config_.name, config_.steps);
    bar.update(current_step_);

    CudaTimer timer;
    timer.start();

    for (long long remaining = config_.steps - current_step_; remaining > 0; --remaining) {
        step();

        if (config_.diagnostics_interval > 0 && writer_ && writer_->is_open() &&
            current_step_ % config_.diagnostics_interval == 0) {
            writer_->write(diagnostics());
        }
        if (config_.snapshot_interval > 0 && current_step_ % config_.snapshot_interval == 0) {
            write_snapshot_now();
        }
        bar.update(current_step_);
    }

    solver_ms_ += timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    bar.finish();

    if (writer_ && writer_->is_open() && config_.diagnostics_interval > 0 &&
        current_step_ % config_.diagnostics_interval != 0) {
        writer_->write(diagnostics());
    }
}

}  // namespace cps
