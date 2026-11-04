#pragma once

#include <cuda_runtime.h>

#include <string>

namespace cps {

class ParticleSystem;

// What the velocity array holds between steps.
//
//   Synchronised  v_n, at the same time as the positions
//   HalfStep      the leapfrog v_{n-1/2}, half a step behind them
//
// Resuming a snapshot written under one convention into a solver that wants the
// other is the one case where the loaded velocity has to be shifted, and getting
// that wrong offsets the whole trajectory by half a step in a way that looks like
// a subtly wrong force. A solver only states its convention. The engine records
// what the array holds, does the shifting and writes the snapshot flag.
enum class VelocityConvention { Synchronised, HalfStep };

// One way of advancing the particle state, which the engine drives without
// knowing which one it holds. The engine keeps the loop, the file output, the
// timing, the reductions and the velocity convention; everything that depends on
// the physics is here.
//
// The engine calls these in a fixed order:
//
//   initialize()  upload, initialize, shift to the convention, drain
//   resume()      upload, on_resume, shift to the convention, drain
//   step()        step
//   diagnostics() check_device_errors, prepare_diagnostics, then the reduction,
//                 with a potential energy pass if has_potential_energy
//   snapshot      check_device_errors, then the download
//
// The shift reads the acceleration array, so what initialize and on_resume leave
// there is what a change of convention uses. Everything takes the stream the
// device work has to go on and only enqueues on it, unless it says otherwise.
//
// The hooks that decide the physics have no default, so a new solver cannot leave
// out a resume fixup by accident: that would only show up as a resumed run
// drifting from a straight one.
class Solver {
   public:
    Solver() = default;
    virtual ~Solver() = default;

    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    // A fresh run: the initial state has just been uploaded. Seeds whatever the
    // first step or the convention shift reads, such as a_0.
    virtual void initialize(ParticleSystem& particles, cudaStream_t stream) = 0;

    // A resumed run: the snapshot's state has just been uploaded, and only the
    // positions, velocities and masses survive in one. Rebuilds whatever else the
    // next step or the convention shift reads, so that the resumed run enters the
    // loop in the state the original run was in.
    virtual void on_resume(ParticleSystem& particles, cudaStream_t stream) = 0;

    // Advances one step, leaving the velocity array in velocity_convention().
    virtual void step(ParticleSystem& particles, cudaStream_t stream) = 0;

    // The convention this solver carries between steps. It must not change over
    // the life of the solver.
    virtual VelocityConvention velocity_convention() const = 0;

    // Runs before a diagnostic reduction, which reads a half step velocity as
    // v + a dt/2. Leaves in the acceleration array what that formula needs. It
    // must not touch anything the trajectory depends on, because a diagnostic
    // must not perturb the run it measures.
    virtual void prepare_diagnostics(ParticleSystem& particles, cudaStream_t stream) = 0;

    // Whether the softened gravitational potential energy pass applies. When it
    // does not, the diagnostics report zero potential energy.
    virtual bool has_potential_energy() const = 0;

    // Throws std::runtime_error naming the step if the device has flagged a failure
    // this solver watches for. A solver with no such flag does nothing here and
    // does not touch the stream.
    virtual void check_device_errors(long long /*step*/, cudaStream_t /*stream*/) const {}

    // Lines for the configuration block printed at start up, each ending in a
    // newline, for what this solver has worked out for itself. May be empty.
    virtual std::string startup_description() const { return {}; }
};

// Converts a synchronised velocity into the half step velocity, and back again:
// v + a dt/2, called with a negative dt to go in and a positive dt to come out.
// Applying it before a leapfrog loop starts is what makes the trajectory agree
// with velocity Verlet rather than being offset by half a step. The stream has no
// default: see nbody_solver.hpp.
void offset_velocity_half_step(float* vx, float* vy, float* vz, const float* ax, const float* ay,
                               const float* az, int count, float dt, int block_size,
                               cudaStream_t stream);

}  // namespace cps
