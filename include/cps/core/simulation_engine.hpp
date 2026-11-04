#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "cps/core/config.hpp"
#include "cps/core/diagnostics.hpp"
#include "cps/core/particle_system.hpp"
#include "cps/core/snapshot.hpp"
#include "cps/solvers/sph_solver.hpp"

namespace cps {

// Owns the device state and drives the loop.
//
// The engine holds no particle data on the host. Configuration, diagnostics
// accumulators and the staging buffer are the only host allocations that scale
// with the particle count, and the staging buffer is only touched when a
// snapshot is written and once at the end of run().
//
// Everything the engine does on the device goes through one stream of its own,
// which is non-blocking, so the host and the device run apart until something
// waits. What is safe when, for code holding the engine:
//
//   - initialize() and resume() return with the stream drained, so the arrays
//     behind particles() can be read or written directly straight after them.
//
//   - step() only enqueues work. After it, those arrays are the device's: reading
//     or writing them from the host, on the default stream or on any stream but
//     this one, races with the steps still running.
//
//   - synchronize() takes them back. It blocks until every step so far has
//     finished, and they are then safe until the next step(). cudaDeviceSynchronize
//     has the same effect, because it waits on every stream.
//
//   - particles().upload() and download() are safe at any time if they are given
//     stream(), which orders them after the steps and returns once they are done.
//
//   - diagnostics(), write_snapshot_now() and check_device_errors() wait for the
//     steps themselves and are safe at any time. So is run(), which returns with
//     the stream drained.
//
//   - The destructor waits for the stream before freeing anything. Device buffers
//     outside the engine that stream() work touches have to be kept alive until
//     a synchronize().
class SimulationEngine {
   public:
    explicit SimulationEngine(Config config);
    ~SimulationEngine();

    SimulationEngine(const SimulationEngine&) = delete;
    SimulationEngine& operator=(const SimulationEngine&) = delete;

    // Builds the initial condition and computes the opening force evaluation.
    // Returns with the stream drained.
    void initialize();

    // Loads the newest snapshot in the configured directory and continues from
    // it. Returns the step the state was written at. Refuses a snapshot whose
    // physics fingerprint differs from this configuration's, naming every
    // field that differs, unless force is set, in which case it warns on
    // stderr and continues. A CPS1 snapshot records no physics, so it loads
    // with a warning that it cannot be verified. Returns with the stream drained.
    long long resume(bool force = false);

    // Advances one step with the configured solver and variant. This enqueues
    // the step and returns without waiting for the device, so a launch failure
    // may only surface at the next synchronisation.
    void step();

    // Blocks until everything enqueued on the stream has finished. See the top
    // of this class for what that makes safe.
    void synchronize() const { stream_.synchronize(); }

    // The stream all of this engine's device work runs on, for code that has to
    // order its own work with it: downloads, a timer, a direct kernel launch.
    cudaStream_t stream() const { return stream_.get(); }

    // Runs to the configured step count, writing snapshots and diagnostics on
    // their configured cadences and drawing a progress bar. After initialize()
    // the diagnostics file starts afresh; after resume() it is continued (see
    // the resuming DiagnosticsWriter constructor).
    //
    // A blown-up state stops the run with std::runtime_error naming the step,
    // through the checks in diagnostics() and write_snapshot_now(), and once
    // the steps are done through check_device_errors() and a host scan of the
    // downloaded state. They run only on those cadences, because a check on
    // every step would add a device sync to the solver path. A run with
    // diagnostics and snapshots both disabled therefore has no periodic check,
    // but it is checked once at the end.
    void run();

    // Throws std::runtime_error naming the current step if the row would hold a
    // non-finite kinetic energy, potential energy or momentum, or if
    // check_device_errors() would throw.
    Diagnostics diagnostics(bool include_potential = true);

    // Throws std::runtime_error, leaving no new file and any older snapshot
    // untouched, if check_device_errors() would throw or if any position or
    // velocity is non-finite, naming the step and the first bad particle.
    void write_snapshot_now() const;

    // Reads the fluid solver's non-finite position flag and throws
    // std::runtime_error naming the current step if it is set. N-body has no
    // such flag, so this does nothing there. Drains the stream if there is one.
    void check_device_errors() const;

    ParticleSystem& particles() { return *particles_; }
    const ParticleSystem& particles() const { return *particles_; }
    const Config& config() const { return config_; }
    long long current_step() const { return current_step_; }

    // The engine's simulated time, t0 + (step - s0) dt. A fresh run has s0 = 0
    // and t0 = 0, which is exactly step * dt. A resume takes s0 and t0 from
    // the snapshot, so a resume forced across a change of dt keeps the time
    // the snapshot was written at and advances it with the new dt from there.
    // When the snapshot's time is exactly step * dt under this configuration,
    // as it is for every resume that did not change dt, and for CPS1, which
    // stores no time, the origin is left at zero so the reported times are bit
    // for bit those of a run that never stopped.
    double simulated_time() const {
        return time_origin_ + static_cast<double>(current_step_ - time_origin_step_) *
                                  static_cast<double>(config_.dt);
    }

    // Timing of run(), accumulated over every call in this process.
    //
    // solver_milliseconds() is device time from CUDA events spanning only the
    // step() calls. Each unbroken stretch of steps between diagnostics and
    // snapshots is timed as one segment, so the figure excludes diagnostics,
    // snapshot writes and the progress bar. Host time spent launching steps
    // inside a segment counts only where it leaves the device idle.
    //
    // wall_milliseconds() is host wall clock time from the start of run() to
    // its return, not counting an initialize() or resume() done beforehand:
    // the steps, every diagnostic (the opening and closing rows included),
    // snapshot writes, the progress bar and the state check at the end. It is
    // never less than the solver time.
    //
    // steps_run() counts the steps run() advanced in this process. After a
    // resume it excludes the steps before the snapshot, so dividing either
    // time by it gives a per step cost; current_step() is the step index.
    double solver_milliseconds() const { return solver_ms_; }
    double wall_milliseconds() const { return wall_ms_; }
    long long steps_run() const { return steps_run_; }

   private:
    void step_nbody_split();
    void step_nbody_fused();
    void ensure_velocity_convention(bool want_half_step);
    // Throws std::runtime_error naming the current step and the first particle
    // with a non-finite position or velocity.
    void check_state_finite(const HostParticleState& state) const;
    void check_physics(const Snapshot& snapshot, const std::filesystem::path& path,
                       bool force) const;
    // A negative resume_step starts the file afresh; otherwise the file is
    // continued from that step.
    void open_diagnostics(long long resume_step);

    Config config_;

    // Ahead of everything that holds device memory, so it outlives all of it. It
    // comes after config_ so a configuration is refused before the device is
    // touched at all.
    Stream stream_;
    std::unique_ptr<ParticleSystem> particles_;
    std::unique_ptr<SphSolver> sph_;
    std::unique_ptr<DiagnosticsReducer> reducer_;
    std::unique_ptr<DiagnosticsWriter> writer_;
    long long current_step_ = 0;
    long long time_origin_step_ = 0;  // s0 in simulated_time()
    double time_origin_ = 0.0;        // t0 in simulated_time()
    double solver_ms_ = 0.0;
    double wall_ms_ = 0.0;
    long long steps_run_ = 0;

    // True when the velocity array holds the leapfrog half step velocity
    // rather than the velocity synchronised with the positions.
    bool half_step_velocity_ = false;
    bool initialized_ = false;
};

}  // namespace cps
