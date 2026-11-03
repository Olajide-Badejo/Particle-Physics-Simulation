#pragma once

#include "cps/core/config.hpp"

namespace cps_test {

// The block of fluid the SPH validation cases run on, shared by the device
// cases in test_sph_validation.cpp and the host cases in
// host/test_sph_validation_host.cpp so the two halves cannot drift apart.
inline cps::Config fluid_config(int count, long long steps) {
    cps::Config config;
    config.name = "sph_validation";
    config.solver = cps::SolverType::Sph;
    config.initial_condition = cps::InitialCondition::DamBreak;
    config.num_particles = count;
    config.steps = steps;
    config.dt = 1.0e-4f;
    config.block_size = 256;
    config.seed = 20260802u;
    config.external_gravity = {0.0f, -9.81f, 0.0f};
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();
    config.sph.domain_min = {0.0f, 0.0f, 0.0f};
    config.sph.domain_max = {0.8f, 0.5f, 0.4f};
    config.sph.fluid_min = {0.0f, 0.0f, 0.0f};
    config.sph.fluid_max = {0.3f, 0.4f, 0.4f};
    config.sph.rest_density = 1000.0f;
    config.sph.gas_constant = 1000.0f;
    config.sph.viscosity = 0.2f;
    config.sph.boundary_damping = 0.4f;
    return config;
}

}  // namespace cps_test
