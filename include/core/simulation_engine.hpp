#pragma once

#include <memory>
#include <string>

#include "core/config.hpp"
#include "core/diagnostics.hpp"
#include "core/particle_system.hpp"
#include "solvers/sph_solver.hpp"

namespace cps {

// Owns the device state and drives the loop.
//
// The engine holds no particle data on the host. Configuration, diagnostics
// accumulators and the snapshot staging buffer are the only host allocations
// that scale with the particle count, and the staging buffer is only touched
// when a snapshot is actually written.
class SimulationEngine {
   public:
    explicit SimulationEngine(Config config);
    ~SimulationEngine();

    SimulationEngine(const SimulationEngine&) = delete;
    SimulationEngine& operator=(const SimulationEngine&) = delete;

    // Builds the initial condition and computes the opening force evaluation.
    void initialize();

    // Loads the newest snapshot in the configured directory and continues from
    // it. Returns the step the state was written at.
    long long resume();

    // Advances one step with the configured solver and variant.
    void step();

    // Runs to the configured step count, writing snapshots and diagnostics on
    // their configured cadences and drawing a progress bar.
    void run();

    Diagnostics diagnostics(bool include_potential = true);

    void write_snapshot_now() const;

    ParticleSystem& particles() { return *particles_; }
    const ParticleSystem& particles() const { return *particles_; }
    const Config& config() const { return config_; }
    long long current_step() const { return current_step_; }

    // Milliseconds of device time spent inside step(), accumulated.
    double solver_milliseconds() const { return solver_ms_; }

   private:
    void step_nbody_split();
    void step_nbody_fused();
    void ensure_velocity_convention(bool want_half_step);

    Config config_;
    std::unique_ptr<ParticleSystem> particles_;
    std::unique_ptr<SphSolver> sph_;
    std::unique_ptr<DiagnosticsReducer> reducer_;
    std::unique_ptr<DiagnosticsWriter> writer_;
    long long current_step_ = 0;
    double solver_ms_ = 0.0;

    // True when the velocity array holds the leapfrog half step velocity
    // rather than the velocity synchronised with the positions.
    bool half_step_velocity_ = false;
    bool initialized_ = false;
};

}  // namespace cps
