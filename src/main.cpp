#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "core/initial_conditions.hpp"
#include "core/simulation_engine.hpp"
#include "utils/cuda_helpers.hpp"
#include "utils/progress.hpp"

namespace {

void print_usage() {
    std::printf(
        "cuda_physics_sim: CUDA particle physics engine\n"
        "\n"
        "usage: cuda_physics_sim --config <file.json> [options]\n"
        "\n"
        "options:\n"
        "  --config <path>       simulation configuration in JSON\n"
        "  --solver <name>       override the solver: nbody or sph\n"
        "  --variant <name>      override the N-body variant: naive, tiled or fused\n"
        "  --steps <n>           override the step count\n"
        "  --particles <n>       override the particle count\n"
        "  --dt <value>          override the time step\n"
        "  --seed <n>            override the random seed\n"
        "  --snapshot-dir <path> override the snapshot directory\n"
        "  --snapshot-every <n>  override the snapshot cadence, 0 disables\n"
        "  --diagnostics <path>  override the diagnostics file, empty disables\n"
        "  --diag-every <n>      override the diagnostics cadence, 0 disables\n"
        "  --resume              continue from the newest snapshot in the snapshot directory\n"
        "  --device-info         print the device properties and exit\n"
        "  --print-config        print the resolved configuration and exit\n"
        "  --help                print this message\n");
}

std::string require_value(int argc, char** argv, int& index, const char* flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("option ") + flag + " needs a value");
    }
    ++index;
    return argv[index];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string config_path;
        bool resume = false;
        bool device_info_only = false;
        bool print_config_only = false;

        // Overrides are collected first and applied after the file is read, so
        // a command line switch always wins over the file regardless of order.
        std::string solver_override;
        std::string variant_override;
        std::string snapshot_dir_override;
        std::string diagnostics_override;
        bool diagnostics_override_set = false;
        long long steps_override = -1;
        long long particles_override = -1;
        long long snapshot_every_override = -1;
        long long diag_every_override = -1;
        double dt_override = -1.0;
        long long seed_override = -1;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            } else if (arg == "--config") {
                config_path = require_value(argc, argv, i, "--config");
            } else if (arg == "--solver") {
                solver_override = require_value(argc, argv, i, "--solver");
            } else if (arg == "--variant") {
                variant_override = require_value(argc, argv, i, "--variant");
            } else if (arg == "--steps") {
                steps_override = std::stoll(require_value(argc, argv, i, "--steps"));
            } else if (arg == "--particles") {
                particles_override = std::stoll(require_value(argc, argv, i, "--particles"));
            } else if (arg == "--dt") {
                dt_override = std::stod(require_value(argc, argv, i, "--dt"));
            } else if (arg == "--seed") {
                seed_override = std::stoll(require_value(argc, argv, i, "--seed"));
            } else if (arg == "--snapshot-dir") {
                snapshot_dir_override = require_value(argc, argv, i, "--snapshot-dir");
            } else if (arg == "--snapshot-every") {
                snapshot_every_override =
                    std::stoll(require_value(argc, argv, i, "--snapshot-every"));
            } else if (arg == "--diagnostics") {
                diagnostics_override = require_value(argc, argv, i, "--diagnostics");
                diagnostics_override_set = true;
            } else if (arg == "--diag-every") {
                diag_every_override = std::stoll(require_value(argc, argv, i, "--diag-every"));
            } else if (arg == "--resume") {
                resume = true;
            } else if (arg == "--device-info") {
                device_info_only = true;
            } else if (arg == "--print-config") {
                print_config_only = true;
            } else {
                std::fprintf(stderr, "unknown option %s\n\n", arg.c_str());
                print_usage();
                return 2;
            }
        }

        if (device_info_only) {
            const cps::DeviceInfo info = cps::query_device(0);
            cps::print_device_info(info, stdout);
            return 0;
        }

        if (config_path.empty()) {
            std::fprintf(stderr, "no configuration given\n\n");
            print_usage();
            return 2;
        }

        cps::Config config = cps::Config::from_json_file(config_path);
        if (!solver_override.empty()) {
            config.solver = cps::solver_from_string(solver_override);
        }
        if (!variant_override.empty()) {
            config.variant = cps::variant_from_string(variant_override);
        }
        if (steps_override >= 0) {
            config.steps = steps_override;
        }
        if (particles_override > 0) {
            config.num_particles = static_cast<int>(particles_override);
        }
        if (dt_override > 0.0) {
            config.dt = static_cast<float>(dt_override);
        }
        if (seed_override >= 0) {
            config.seed = static_cast<uint32_t>(seed_override);
        }
        if (!snapshot_dir_override.empty()) {
            config.snapshot_dir = snapshot_dir_override;
        }
        if (snapshot_every_override >= 0) {
            config.snapshot_interval = snapshot_every_override;
        }
        if (diagnostics_override_set) {
            config.diagnostics_path = diagnostics_override;
        }
        if (diag_every_override >= 0) {
            config.diagnostics_interval = diag_every_override;
        }
        config.validate();

        if (print_config_only) {
            std::printf("%s\n", config.to_json_string().c_str());
            return 0;
        }

        const cps::DeviceInfo info = cps::query_device(0);
        cps::print_device_info(info, stdout);

        std::printf("\nconfiguration: %s\n", config.name.c_str());
        std::printf("  solver        : %s", cps::to_string(config.solver).c_str());
        if (config.solver == cps::SolverType::NBody) {
            std::printf(" (%s)", cps::to_string(config.variant).c_str());
        }
        std::printf("\n");
        std::printf("  particles     : %s\n",
                    cps::format_count(config.num_particles).c_str());
        std::printf("  steps         : %s\n", cps::format_count(config.steps).c_str());
        std::printf("  time step     : %g\n", static_cast<double>(config.dt));
        std::printf("  seed          : %u\n", config.seed);
        std::printf("  fast math     : %s\n", CPS_FAST_MATH ? "on" : "off");
        if (config.solver == cps::SolverType::Sph) {
            const cps::SphDerivedParameters derived = cps::derive_sph_parameters(config);
            std::printf("  smoothing h   : %g\n", static_cast<double>(derived.smoothing_radius));
            std::printf("  particle mass : %g\n", static_cast<double>(derived.particle_mass));
            std::printf("  neighbours    : %.1f expected inside the support\n",
                        derived.expected_neighbour_count);
        }

        cps::SimulationEngine engine(config);
        if (resume) {
            engine.resume();
        } else {
            engine.initialize();
        }

        std::printf("  device state  : %.2f MiB\n",
                    static_cast<double>(engine.particles().device_bytes()) / (1024.0 * 1024.0));
        std::printf("\n");
        std::fflush(stdout);

        engine.run();

        const double seconds = engine.solver_milliseconds() / 1.0e3;
        const long long done = engine.current_step();
        std::printf("\ncompleted %s steps in %.3f s", cps::format_count(done).c_str(), seconds);
        if (seconds > 0.0) {
            std::printf(" (%.1f steps/s, %.3f ms/step)", static_cast<double>(done) / seconds,
                        engine.solver_milliseconds() / static_cast<double>(done));
        }
        std::printf("\n");

        if (config.solver == cps::SolverType::NBody) {
            const cps::Diagnostics final_row = engine.diagnostics(true);
            std::printf("final energy: kinetic %.9g potential %.9g total %.9g\n",
                        final_row.kinetic, final_row.potential, final_row.total);
            std::printf("final momentum magnitude: %.6g\n", final_row.momentum_magnitude);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
