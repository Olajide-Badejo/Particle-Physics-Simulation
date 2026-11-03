#include "core/simulation_engine.hpp"

#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/initial_conditions.hpp"
#include "core/physics_fingerprint.hpp"
#include "core/snapshot.hpp"
#include "solvers/nbody_solver.hpp"
#include "utils/progress.hpp"

namespace fs = std::filesystem;

namespace cps {
namespace {

// Which velocity convention a solver carries between steps. The split velocity
// Verlet keeps the velocity synchronised with the positions; the fused kernel
// advances a leapfrog half step velocity. Resuming a snapshot written under one
// convention into a solver that wants the other is the one case where the
// loaded velocity has to be shifted, and getting that wrong offsets the whole
// trajectory by half a step in a way that looks like a subtly wrong force.
//
// The fluid solver runs the same kick-drift leapfrog update as the fused
// kernel, but its initial offset is not applied: the acceleration is still
// zero when the offset runs, so the initial velocity is taken as v_{-1/2} as
// it stands. Its viscous term is evaluated with that half step velocity, which
// makes it first order. The flag stays true for the fluid because it decides
// how the diagnostics read the velocity, and v + a dt/2 is a better estimate
// of the synchronised velocity than v alone.
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
}

SimulationEngine::~SimulationEngine() = default;

// The file is opened only once the engine knows whether this is a fresh run or
// a resume, and for a resume only after every check has passed, so a refused
// resume leaves the original run's diagnostics untouched.
void SimulationEngine::open_diagnostics(long long resume_step) {
    writer_.reset();
    if (config_.diagnostics_path.empty() || config_.diagnostics_interval <= 0) {
        return;
    }
    const fs::path path(config_.diagnostics_path);
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }
    if (resume_step < 0) {
        writer_ = std::make_unique<DiagnosticsWriter>(config_.diagnostics_path);
    } else {
        writer_ = std::make_unique<DiagnosticsWriter>(config_.diagnostics_path, resume_step);
    }
}

void SimulationEngine::initialize() {
    const HostParticleState state = make_initial_state(config_);
    particles_->upload(state);
    current_step_ = 0;
    time_origin_step_ = 0;
    time_origin_ = 0.0;
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
    // records the convention without touching a single velocity: the initial
    // half step offset is not applied, and the initial velocity is used as
    // v_{-1/2} as it stands.
    ensure_velocity_convention(solver_wants_half_step(config_));
    open_diagnostics(-1);
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

// A count mismatch cannot be forced past, because the state would not fit,
// but a physics mismatch can: the state loads fine and only the meaning of
// the continued run is in doubt, which the person resuming may know is fine.
void SimulationEngine::check_physics(const Snapshot& snapshot, const fs::path& path,
                                     bool force) const {
    if (snapshot.format_version == 1) {
        std::fprintf(stderr,
                     "warning: snapshot %s is in the old CPS1 format, which records no "
                     "configuration, so its physics cannot be verified against this one\n",
                     path.string().c_str());
        return;
    }

    const std::vector<PhysicsField> fields = physics_fields(config_);
    const uint64_t fingerprint = physics_fingerprint(fields);
    if (snapshot.fingerprint == fingerprint &&
        snapshot.fingerprint_version == PHYSICS_FINGERPRINT_VERSION) {
        return;
    }

    std::string detail;
    try {
        detail = describe_differences(compare_physics(snapshot.physics, fields));
    } catch (const std::runtime_error& error) {
        detail = error.what();
    }
    if (detail.empty()) {
        char buffer[192];
        std::snprintf(buffer, sizeof(buffer),
                      "every recorded field agrees, but the fingerprint is 0x%016" PRIx64
                      " (version %u) in the snapshot and 0x%016" PRIx64
                      " (version %u) in the configuration",
                      snapshot.fingerprint, snapshot.fingerprint_version, fingerprint,
                      PHYSICS_FINGERPRINT_VERSION);
        detail = buffer;
    }
    const std::string message =
        "snapshot " + path.string() + " was written under different physics: " + detail;
    if (!force) {
        throw std::runtime_error(message + ". Resuming would continue a different simulation; " +
                                 "pass --force-resume to do it anyway");
    }
    std::fprintf(stderr, "warning: %s; resuming anyway because --force-resume was given\n",
                 message.c_str());
}

long long SimulationEngine::resume(bool force) {
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
    if (snapshot.step > static_cast<uint64_t>(std::numeric_limits<long long>::max())) {
        throw std::runtime_error("snapshot " + path.string() + " is at step " +
                                 std::to_string(snapshot.step) + ", beyond what a run can reach");
    }
    check_physics(snapshot, path, force);

    particles_->upload(snapshot.state);
    current_step_ = static_cast<long long>(snapshot.step);
    half_step_velocity_ = (snapshot.flags & SNAPSHOT_FLAG_HALF_STEP_VELOCITY) != 0u;

    // Time carries on from the snapshot's own clock. When that clock is
    // exactly what step * dt gives under this configuration, which is every
    // resume that did not force a change of dt, the origin stays at zero so
    // the resumed run reports bit for bit the times the straight run would
    // have. A CPS1 snapshot has no clock, so it gets step * dt as well.
    const double resumed_at =
        static_cast<double>(current_step_) * static_cast<double>(config_.dt);
    time_origin_step_ = 0;
    time_origin_ = 0.0;
    if (std::isfinite(snapshot.time) &&
        std::bit_cast<uint64_t>(snapshot.time) != std::bit_cast<uint64_t>(resumed_at)) {
        time_origin_step_ = current_step_;
        time_origin_ = snapshot.time;
    }

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

    // After a fluid resume the acceleration array is zero, because the fluid
    // solver keeps no force pass outside its step, so the first diagnostic row
    // after the resume reads the velocity with a zero acceleration. The state
    // itself is unaffected and the trajectory stays bit identical.
    ensure_velocity_convention(solver_wants_half_step(config_));
    open_diagnostics(current_step_);
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
    if (config_.solver == SolverType::NBody && config_.variant == NBodyVariant::Fused) {
        // The fused kernel leaves a_n in the acceleration array, the force at
        // the positions it started from, but the synchronised velocity is
        // v_{n+1/2} + a_{n+1} dt/2. Recomputing the force at the current
        // positions gives a_{n+1}. The fused kernel never reads this array,
        // so overwriting it changes nothing the trajectory depends on.
        compute_forces_tiled(particles_->x(), particles_->y(), particles_->z(), particles_->mass(),
                             particles_->ax(), particles_->ay(), particles_->az(),
                             particles_->size(), config_.softening, config_.gravitational_constant,
                             config_.block_size);
    }
    const float half_dt = half_step_velocity_ ? 0.5f * config_.dt : 0.0f;
    const bool with_potential = include_potential && config_.solver == SolverType::NBody;
    return reducer_->compute(*particles_, current_step_, simulated_time(), config_.softening,
                             config_.gravitational_constant, with_potential, half_dt);
}

void SimulationEngine::write_snapshot_now() const {
    Snapshot snapshot;
    snapshot.step = static_cast<uint64_t>(current_step_);
    snapshot.time = simulated_time();
    snapshot.flags = half_step_velocity_ ? SNAPSHOT_FLAG_HALF_STEP_VELOCITY : 0u;
    const std::vector<PhysicsField> fields = physics_fields(config_);
    snapshot.fingerprint = physics_fingerprint(fields);
    snapshot.fingerprint_version = PHYSICS_FINGERPRINT_VERSION;
    snapshot.physics = physics_json(fields);
    particles_->download(snapshot.state);
    write_snapshot(snapshot_path_for_step(config_.snapshot_dir, snapshot.step), snapshot);
}

void SimulationEngine::run() {
    using Clock = std::chrono::steady_clock;
    if (!initialized_) {
        initialize();
    }
    const Clock::time_point wall_start = Clock::now();

    if (writer_ && writer_->is_open()) {
        writer_->write(diagnostics());
    }

    ProgressBar bar(config_.name.empty() ? std::string("simulation") : config_.name, config_.steps);
    bar.update(current_step_);

    // The solver clock runs on device events around each unbroken stretch of
    // step() calls and stops just before a diagnostic or a snapshot. Both of
    // those copy from the device and so synchronise anyway; stopping the
    // clock there adds no synchronisation that would not have happened.
    CudaTimer timer;
    bool timing = false;

    for (long long remaining = config_.steps - current_step_; remaining > 0; --remaining) {
        if (!timing) {
            timer.start();
            timing = true;
        }
        step();
        ++steps_run_;

        const bool diagnostic_due = config_.diagnostics_interval > 0 && writer_ &&
                                    writer_->is_open() &&
                                    current_step_ % config_.diagnostics_interval == 0;
        const bool snapshot_due =
            config_.snapshot_interval > 0 && current_step_ % config_.snapshot_interval == 0;
        if (diagnostic_due || snapshot_due) {
            solver_ms_ += timer.stop();
            timing = false;
        }
        if (diagnostic_due) {
            writer_->write(diagnostics());
        }
        if (snapshot_due) {
            if (writer_) {
                writer_->flush();
            }
            write_snapshot_now();
        }
        bar.update(current_step_);
    }

    if (timing) {
        solver_ms_ += timer.stop();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    bar.finish();

    if (writer_ && writer_->is_open() && config_.diagnostics_interval > 0 &&
        current_step_ % config_.diagnostics_interval != 0) {
        writer_->write(diagnostics());
    }
    if (writer_) {
        writer_->flush();
    }

    wall_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - wall_start).count();
}

}  // namespace cps
