#include "cps/solvers/nbody_solver.hpp"

#include <stdexcept>
#include <string>

#include "cps/core/particle_system.hpp"

namespace cps {

NBodySolver::NBodySolver(const Config& config) : config_(config) {
    switch (config_.variant) {
        case NBodyVariant::Naive:
        case NBodyVariant::Tiled:
        case NBodyVariant::Fused:
            return;
    }
    throw std::runtime_error("unknown N-body variant " +
                             std::to_string(static_cast<int>(config_.variant)) +
                             ", expected naive, tiled or fused");
}

void NBodySolver::compute_forces(ParticleSystem& particles, cudaStream_t stream) {
    compute_forces_tiled(particles.x(), particles.y(), particles.z(), particles.mass(),
                         particles.ax(), particles.ay(), particles.az(), particles.size(),
                         config_.softening, config_.gravitational_constant, config_.block_size,
                         config_.nbody.tile_size, stream);
}

// Velocity Verlet needs a_0 before its first half kick, and the fused leapfrog
// needs it to step the velocity back by half a step. Every variant seeds with the
// tiled kernel, the naive one included.
void NBodySolver::initialize(ParticleSystem& particles, cudaStream_t stream) {
    compute_forces(particles, stream);
}

// Accelerations are a deterministic function of the positions, so recomputing them
// here reproduces exactly the array the original run held at this step. That is
// what makes the resumed trajectory bit identical rather than merely close.
void NBodySolver::on_resume(ParticleSystem& particles, cudaStream_t stream) {
    compute_forces(particles, stream);
}

void NBodySolver::step_split(ParticleSystem& particles, cudaStream_t stream) {
    integrate_verlet_position(particles.x(), particles.y(), particles.z(), particles.vx(),
                              particles.vy(), particles.vz(), particles.ax(), particles.ay(),
                              particles.az(), particles.size(), config_.dt, config_.block_size,
                              stream);

    if (config_.variant == NBodyVariant::Naive) {
        compute_forces_naive(particles.x(), particles.y(), particles.z(), particles.mass(),
                             particles.ax(), particles.ay(), particles.az(), particles.size(),
                             config_.softening, config_.gravitational_constant, config_.block_size,
                             stream);
    } else {
        compute_forces(particles, stream);
    }

    integrate_verlet_velocity(particles.vx(), particles.vy(), particles.vz(), particles.ax(),
                              particles.ay(), particles.az(), particles.size(), config_.dt,
                              config_.block_size, stream);
}

void NBodySolver::step_fused(ParticleSystem& particles, cudaStream_t stream) {
    nbody_fused_step(particles.x(), particles.y(), particles.z(), particles.x_next(),
                     particles.y_next(), particles.z_next(), particles.vx(), particles.vy(),
                     particles.vz(), particles.ax(), particles.ay(), particles.az(),
                     particles.mass(), particles.size(), config_.softening,
                     config_.gravitational_constant, config_.dt, config_.block_size,
                     config_.nbody.tile_size, stream);
    particles.swap_position_buffers();
}

void NBodySolver::step(ParticleSystem& particles, cudaStream_t stream) {
    if (config_.variant == NBodyVariant::Fused) {
        step_fused(particles, stream);
    } else {
        step_split(particles, stream);
    }
}

// The split velocity Verlet keeps the velocity synchronised with the positions;
// the fused kernel advances a leapfrog half step velocity.
VelocityConvention NBodySolver::velocity_convention() const {
    return config_.variant == NBodyVariant::Fused ? VelocityConvention::HalfStep
                                                  : VelocityConvention::Synchronised;
}

// The fused kernel leaves a_n in the acceleration array, the force at the positions
// it started from, but the synchronised velocity is v_{n+1/2} + a_{n+1} dt/2.
// Recomputing the force at the current positions gives a_{n+1}. The fused kernel
// never reads this array, so overwriting it changes nothing the trajectory depends
// on. The split variants read it, and it already holds a_{n+1} there.
void NBodySolver::prepare_diagnostics(ParticleSystem& particles, cudaStream_t stream) {
    if (config_.variant == NBodyVariant::Fused) {
        compute_forces(particles, stream);
    }
}

}  // namespace cps
