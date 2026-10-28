#include <cstdio>
#include <exception>
#include <string>

#include "core/cli.hpp"
#include "core/initial_conditions.hpp"
#include "core/simulation_engine.hpp"
#include "utils/cuda_helpers.hpp"
#include "utils/progress.hpp"

int main(int argc, char** argv) {
    cps::CliOptions options;
    try {
        options = cps::parse_cli(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n\n%s", error.what(), cps::cli_usage().c_str());
        return 2;
    }
    if (options.show_help) {
        std::printf("%s", cps::cli_usage().c_str());
        return 0;
    }

    try {
        if (options.device_info) {
            const cps::DeviceInfo info = cps::query_device(0);
            cps::print_device_info(info, stdout);
            return 0;
        }

        if (options.config_path.empty()) {
            std::fprintf(stderr, "no configuration given\n\n%s", cps::cli_usage().c_str());
            return 2;
        }

        cps::Config config = cps::Config::from_json_file(options.config_path);
        cps::apply_cli_overrides(options, config);
        config.validate();

        if (options.print_config) {
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
        if (options.resume) {
            engine.resume(options.force_resume);
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
