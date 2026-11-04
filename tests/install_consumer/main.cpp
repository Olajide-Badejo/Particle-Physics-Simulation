// Proves an installed cps tree is usable on its own: its headers compile, its
// targets link, and a small N-body run works. Exits 77 with no usable GPU, the
// code ctest treats as skipped elsewhere in this project.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <exception>

#include "cps/core/simulation_engine.hpp"
#include "cps/physics/gravity.hpp"
#include "cps/version.hpp"

int main() {
    std::printf("cps %s\n", CPS_VERSION_STRING);

    // A header only formula, on the host: two unit masses a unit apart.
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    cps::accumulate_pair_acceleration(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, ax, ay, az);
    if (std::abs(ax - 1.0f) > 1.0e-3f) {
        std::fprintf(stderr, "unexpected pair acceleration %g\n", static_cast<double>(ax));
        return 1;
    }

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "no usable CUDA device, skipping the engine run\n");
        return 77;
    }

    cps::Config config;
    config.name = "install_consumer";
    config.solver = cps::SolverType::NBody;
    config.variant = cps::NBodyVariant::Tiled;
    config.initial_condition = cps::InitialCondition::Plummer;
    config.num_particles = 256;
    config.steps = 10;
    config.dt = 1.0e-3f;
    config.softening = 0.05f;
    config.gravitational_constant = 1.0f;
    config.block_size = 128;
    config.snapshot_interval = 0;
    config.diagnostics_interval = 0;
    config.diagnostics_path.clear();

    try {
        cps::SimulationEngine engine(config);
        engine.initialize();
        for (long long i = 0; i < config.steps; ++i) {
            engine.step();
        }
        const cps::Diagnostics row = engine.diagnostics();
        std::printf("ran %lld steps of %d bodies, total energy %.6f\n", engine.current_step(),
                    config.num_particles, row.total);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "engine run failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
