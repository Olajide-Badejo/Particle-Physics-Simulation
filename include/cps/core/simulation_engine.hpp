#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "cps/core/config.hpp"
#include "cps/core/diagnostics.hpp"
#include "cps/core/particle_system.hpp"
#include "cps/core/snapshot.hpp"
#include "cps/io/vtk.hpp"
#include "cps/solvers/solver.hpp"
#include "cps/utils/cuda_helpers.hpp"

namespace cps {

// Owns the device state and drives the loop. What a step does belongs to the
// Solver it holds: solver.hpp says which hook runs when.
//
// The engine holds no particle data on the host. Configuration, diagnostics
// accumulators and the staging buffer are the only host allocations that scale
// with the particle count, and the staging buffer is only touched when a
// snapshot or a VTK frame is written and once at the end of run().
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
//
// With run.use_graphs on and a solver that can be captured, step() launches a graph
// captured at the end of initialize() or resume() and everything above still holds.
// The arithmetic and its order are the same, so a run gives the same bits either way.
class SimulationEngine {
   public:
    // Builds the solver the configuration asks for.
    explicit SimulationEngine(Config config);

    // Runs the solver it is given, which may be any Solver at all. The configuration
    // still sets the particle count and the initial state, the time step the engine
    // shifts velocities and reports time with, and what snapshots record as the
    // physics.
    SimulationEngine(Config config, std::unique_ptr<Solver> solver);

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

    // Advances one step with the solver. This enqueues the step and returns without
    // waiting for the device, so a launch failure may only surface at the next
    // synchronisation.
    void step();

    // Blocks until everything enqueued on the stream has finished. See the top
    // of this class for what that makes safe.
    void synchronize() const { stream_.synchronize(); }

    // The stream all of this engine's device work runs on, for code that has to
    // order its own work with it: downloads, a timer, a direct kernel launch.
    cudaStream_t stream() const { return stream_.get(); }

    // Runs to the configured step count, writing snapshots, diagnostics and VTK
    // frames on their configured cadences and drawing a progress bar. After
    // initialize() the diagnostics file and the VTK series start afresh; after
    // resume() they are continued (see the resuming DiagnosticsWriter constructor
    // and VtkSeries::resumed).
    //
    // Frames fall on multiples of the VTK cadence, step 0 included, and nowhere
    // else, so a resumed run ends up with the frames of a run that never stopped. A
    // run whose last step is off the cadence gets no extra frame.
    //
    // A blown-up state stops the run with std::runtime_error naming the step,
    // through the checks in diagnostics(), write_snapshot_now() and the frame
    // writer, and once the steps are done through check_device_errors() and a host
    // scan of the downloaded state. They run only on those cadences, because a
    // check on every step would add a device sync to the solver path. A run with
    // all three disabled therefore has no periodic check, but it is checked once
    // at the end.
    void run();

    // Throws std::runtime_error naming the current step if the row would hold a
    // non-finite kinetic energy, potential energy or momentum, or if
    // check_device_errors() would throw.
    Diagnostics diagnostics(bool include_potential = true);

    // Throws std::runtime_error, leaving no new file and any older snapshot
    // untouched, if check_device_errors() would throw or if any position or
    // velocity is non-finite, naming the step and the first bad particle.
    void write_snapshot_now() const;

    // Asks the solver whether the device has flagged a failure, and throws
    // std::runtime_error naming the current step if so. The fluid solver reads its
    // non-finite position flag, which drains the stream. N-body has no such flag,
    // so this does nothing there.
    void check_device_errors() const;

    ParticleSystem& particles() { return *particles_; }
    const ParticleSystem& particles() const { return *particles_; }
    const Solver& solver() const { return *solver_; }
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
    // step() calls. Each unbroken stretch of steps between diagnostics, snapshots
    // and frames is timed as one segment, so the figure excludes all three and
    // the progress bar. Host time spent launching steps
    // inside a segment counts only where it leaves the device idle.
    //
    // wall_milliseconds() is host wall clock time from the start of run() to
    // its return, not counting an initialize() or resume() done beforehand:
    // the steps, every diagnostic (the opening and closing rows included),
    // snapshot and frame writes, the progress bar and the state check at the end. It is
    // never less than the solver time.
    //
    // steps_run() counts the steps run() advanced in this process. After a
    // resume it excludes the steps before the snapshot, so dividing either
    // time by it gives a per step cost; current_step() is the step index.
    double solver_milliseconds() const { return solver_ms_; }
    double wall_milliseconds() const { return wall_ms_; }
    long long steps_run() const { return steps_run_; }

    // Whether step() is launching captured graphs: run.use_graphs is on, the solver
    // can be captured and a capture has happened.
    bool using_graphs() const { return step_graph_.has_value(); }

    // Host time spent capturing, over every call in this process. Capture is part of
    // initialize() and resume(), so like them it is not in either time above.
    double graph_capture_milliseconds() const { return graph_capture_ms_; }

   private:
    // Shifts the velocity array into the solver's convention if it is not there
    // already. Reads the acceleration array.
    void ensure_velocity_convention();
    // Captures the solver's step into a graph, if graphs are in use.
    void capture_step_graph();
    // Throws std::runtime_error naming the current step and the first particle
    // with a non-finite position or velocity.
    void check_state_finite(const HostParticleState& state) const;
    void check_physics(const Snapshot& snapshot, const std::filesystem::path& path,
                       bool force) const;
    // A negative resume_step starts the file afresh; otherwise the file is
    // continued from that step.
    void open_diagnostics(long long resume_step);
    // The same choice for the VTK series. Does nothing when frames are off, so a
    // run with them off leaves an existing series alone.
    void open_vtk_series(long long resume_step);
    // Refuses a blown-up state the way write_snapshot_now() does, leaving the series
    // as it was.
    void write_vtk_frame_now();

    Config config_;

    // Ahead of everything that holds device memory, so it outlives all of it. It
    // comes after config_ so a configuration is refused before the device is
    // touched at all.
    Stream stream_;
    std::unique_ptr<ParticleSystem> particles_;
    std::unique_ptr<Solver> solver_;
    // Empty on the stream path. After the particles and the solver, so it goes before
    // the buffers its capture names.
    std::optional<CapturedGraph> step_graph_;
    std::unique_ptr<DiagnosticsReducer> reducer_;
    std::unique_ptr<DiagnosticsWriter> writer_;
    std::optional<VtkSeries> vtk_series_;
    long long current_step_ = 0;
    long long time_origin_step_ = 0;  // s0 in simulated_time()
    double time_origin_ = 0.0;        // t0 in simulated_time()
    double solver_ms_ = 0.0;
    double wall_ms_ = 0.0;
    double graph_capture_ms_ = 0.0;
    long long steps_run_ = 0;

    // True when the velocity array holds the leapfrog half step velocity
    // rather than the velocity synchronised with the positions.
    bool half_step_velocity_ = false;
    bool initialized_ = false;
};

}  // namespace cps
