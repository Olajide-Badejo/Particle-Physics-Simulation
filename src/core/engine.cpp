#include "cps/core/simulation_engine.hpp"

#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cps/core/initial_conditions.hpp"
#include "cps/core/physics_fingerprint.hpp"
#include "cps/core/snapshot.hpp"
#include "cps/solvers/solver_factory.hpp"
#include "cps/utils/progress.hpp"

namespace fs = std::filesystem;

namespace cps {
namespace {

Config validated(Config config) {
    config.validate();
    return config;
}

}  // namespace

SimulationEngine::SimulationEngine(Config config) : config_(validated(std::move(config))) {
    particles_ = std::make_unique<ParticleSystem>(config_.num_particles);
    reducer_ = std::make_unique<DiagnosticsReducer>(config_.num_particles);
    solver_ = make_solver(config_);
}

SimulationEngine::SimulationEngine(Config config, std::unique_ptr<Solver> solver)
    : config_(validated(std::move(config))), solver_(std::move(solver)) {
    if (!solver_) {
        throw std::runtime_error("the engine was given a null solver");
    }
    particles_ = std::make_unique<ParticleSystem>(config_.num_particles);
    reducer_ = std::make_unique<DiagnosticsReducer>(config_.num_particles);
}

// The members are about to free device memory the stream may still be using. A
// destructor has nowhere to send a failure, so the status is dropped here; a run
// that reaches its end has already been through a checked synchronisation.
SimulationEngine::~SimulationEngine() { static_cast<void>(cudaStreamSynchronize(stream_.get())); }

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
    particles_->upload(state, stream_.get());
    current_step_ = 0;
    time_origin_step_ = 0;
    time_origin_ = 0.0;
    half_step_velocity_ = false;

    solver_->initialize(*particles_, stream_.get());
    ensure_velocity_convention();
    stream_.synchronize();
    open_diagnostics(-1);
    initialized_ = true;
}

void SimulationEngine::ensure_velocity_convention() {
    const bool want_half_step = solver_->velocity_convention() == VelocityConvention::HalfStep;
    if (want_half_step == half_step_velocity_) {
        return;
    }
    // Stepping to the half step velocity means going backwards half a step;
    // coming back out of it means going forwards.
    const float dt = want_half_step ? -config_.dt : config_.dt;
    offset_velocity_half_step(particles_->vx(), particles_->vy(), particles_->vz(),
                              particles_->ax(), particles_->ay(), particles_->az(),
                              particles_->size(), dt, config_.block_size, stream_.get());
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

    particles_->upload(snapshot.state, stream_.get());
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

    solver_->on_resume(*particles_, stream_.get());
    ensure_velocity_convention();
    stream_.synchronize();
    open_diagnostics(current_step_);
    initialized_ = true;
    std::fprintf(stderr, "resumed from %s at step %lld\n", path.string().c_str(), current_step_);
    return current_step_;
}

void SimulationEngine::step() {
    if (!initialized_) {
        throw std::runtime_error("the engine must be initialised or resumed before stepping");
    }
    solver_->step(*particles_, stream_.get());
    ++current_step_;
}

void SimulationEngine::check_device_errors() const {
    solver_->check_device_errors(current_step_, stream_.get());
}

void SimulationEngine::check_state_finite(const HostParticleState& state) const {
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (!(std::isfinite(state.x[i]) && std::isfinite(state.y[i]) &&
              std::isfinite(state.z[i]) && std::isfinite(state.vx[i]) &&
              std::isfinite(state.vy[i]) && std::isfinite(state.vz[i]))) {
            throw std::runtime_error(
                "non-finite state at step " + std::to_string(current_step_) + ": particle " +
                std::to_string(i) +
                " has a non-finite position or velocity. The state has blown up, typically from " +
                "too large a time step");
        }
    }
}

Diagnostics SimulationEngine::diagnostics(bool include_potential) {
    check_device_errors();
    solver_->prepare_diagnostics(*particles_, stream_.get());
    const float half_dt = half_step_velocity_ ? 0.5f * config_.dt : 0.0f;
    const bool with_potential = include_potential && solver_->has_potential_energy();
    const Diagnostics row =
        reducer_->compute(*particles_, current_step_, simulated_time(), config_.softening,
                          config_.gravitational_constant, with_potential, half_dt, stream_.get());
    if (!(std::isfinite(row.kinetic) && std::isfinite(row.potential) && std::isfinite(row.px) &&
          std::isfinite(row.py) && std::isfinite(row.pz))) {
        char buffer[192];
        std::snprintf(buffer, sizeof(buffer),
                      "kinetic energy %g, potential energy %g, momentum (%g, %g, %g)", row.kinetic,
                      row.potential, row.px, row.py, row.pz);
        throw std::runtime_error("non-finite diagnostics at step " +
                                 std::to_string(current_step_) + ": " + buffer +
                                 ". The state has blown up, typically from too large a time step");
    }
    return row;
}

void SimulationEngine::write_snapshot_now() const {
    check_device_errors();
    Snapshot snapshot;
    snapshot.step = static_cast<uint64_t>(current_step_);
    snapshot.time = simulated_time();
    snapshot.flags = half_step_velocity_ ? SNAPSHOT_FLAG_HALF_STEP_VELOCITY : 0u;
    const std::vector<PhysicsField> fields = physics_fields(config_);
    snapshot.fingerprint = physics_fingerprint(fields);
    snapshot.fingerprint_version = PHYSICS_FINGERPRINT_VERSION;
    snapshot.physics = physics_json(fields);
    particles_->download(snapshot.state, stream_.get());

    // A resume trusts what it loads, so a blown-up state must never reach a
    // checkpoint. Refusing here, before the writer has created anything, also
    // leaves any older snapshot in place as the one to go back to.
    check_state_finite(snapshot.state);
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
            timer.start(stream_.get());
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
            solver_ms_ += timer.stop(stream_.get());
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
        solver_ms_ += timer.stop(stream_.get());
    }
    stream_.synchronize();
    bar.finish();
    check_device_errors();

    // The flag only sees positions at the start of a step, so it misses a NaN
    // born in the last step, and N-body has no flag at all. Both clocks are
    // accounted for here: the solver clock has stopped, and the wall clock,
    // which covers everything run() does, is still running.
    HostParticleState final_state;
    particles_->download(final_state, stream_.get());
    check_state_finite(final_state);

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
