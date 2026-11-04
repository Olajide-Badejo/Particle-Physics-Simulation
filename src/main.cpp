#include <cstdio>
#include <exception>
#include <string>

#include "cps/core/cli.hpp"
#include "cps/core/convert.hpp"
#include "cps/core/presets.hpp"
#include "cps/core/simulation_engine.hpp"
#include "cps/version.hpp"
#include "cps/utils/cuda_helpers.hpp"
#include "cps/utils/progress.hpp"

namespace {

// The file or embedded preset the command line names. The name of a preset that
// does not exist has been ruled out by the time this runs.
cps::Config load_config(const cps::CliOptions& options) {
    if (options.command != cps::CliCommand::Run) {
        return cps::Config::from_json_file(options.config_path);
    }
    if (cps::run_target_is_preset(options.run_target)) {
        return cps::Config::from_json_string(cps::find_preset(options.run_target)->json);
    }
    return cps::Config::from_json_file(options.run_target);
}

}  // namespace

int main(int argc, char** argv) {
    cps::CliOptions options;
    try {
        options = cps::parse_command_line(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n\n%s", error.what(), cps::cli_usage().c_str());
        return 2;
    }
    if (options.command == cps::CliCommand::Usage) {
        std::fprintf(stderr, "%s", cps::cli_usage().c_str());
        return 2;
    }
    if (options.show_help) {
        std::printf("%s", cps::cli_usage(options.command).c_str());
        return 0;
    }
    if (options.show_version) {
        std::printf("%s\n", CPS_VERSION_STRING);
        return 0;
    }

    if (options.command == cps::CliCommand::Legacy) {
        std::fprintf(stderr, "%s\n", cps::legacy_form_notice().c_str());
    }
    if (options.command == cps::CliCommand::Run && cps::run_target_is_preset(options.run_target) &&
        cps::find_preset(options.run_target) == nullptr) {
        std::fprintf(stderr,
                     "error: unknown preset '%s': cps presets lists the choices, and a file needs "
                     "a path separator or a .json ending\n",
                     options.run_target.c_str());
        return 2;
    }

    try {
        if (options.command == cps::CliCommand::Presets) {
            std::printf("%s", cps::preset_listing().c_str());
            return 0;
        }
        if (options.command == cps::CliCommand::Convert) {
            const std::string what = cps::convert_file(options.convert_input, options.convert_output,
                                                       options.config_path);
            std::printf("wrote %s: %s\n", options.convert_output.c_str(), what.c_str());
            return 0;
        }
        if (options.command == cps::CliCommand::Info) {
            std::printf("cps %s\ncommit %s\n\n", CPS_VERSION, CPS_GIT_STAMP);
            cps::print_device_info(cps::query_device(0), stdout);
            return 0;
        }

        if (options.device_info) {
            const cps::DeviceInfo info = cps::query_device(0);
            cps::print_device_info(info, stdout);
            return 0;
        }

        if (options.command != cps::CliCommand::Run && options.config_path.empty()) {
            std::fprintf(stderr, "no configuration given\n\n%s", cps::cli_usage().c_str());
            return 2;
        }

        cps::Config config = load_config(options);
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

        cps::SimulationEngine engine(config);
        std::printf("%s", engine.solver().startup_description().c_str());
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

        // Both figures divide by the steps this process ran, not by the step
        // index, which after a resume includes the steps before the snapshot.
        // Solver time covers only the step() calls; wall time also covers
        // diagnostics, snapshot writes, the progress bar and the final state
        // check.
        const long long done = engine.steps_run();
        const double solver_ms = engine.solver_milliseconds();
        const double wall_ms = engine.wall_milliseconds();
        std::printf("\ncompleted %s steps in this process, final step %s\n",
                    cps::format_count(done).c_str(),
                    cps::format_count(engine.current_step()).c_str());
        if (done > 0) {
            const double per_step = static_cast<double>(done);
            std::printf("  solver ms/step: %.4f (%.3f s in step(), %.1f steps/s)\n",
                        solver_ms / per_step, solver_ms / 1.0e3,
                        solver_ms > 0.0 ? per_step / (solver_ms / 1.0e3) : 0.0);
            std::printf("  wall ms/step  : %.4f (%.3f s in run(), %.1f steps/s)\n",
                        wall_ms / per_step, wall_ms / 1.0e3,
                        wall_ms > 0.0 ? per_step / (wall_ms / 1.0e3) : 0.0);
        }

        if (engine.solver().has_potential_energy()) {
            const cps::Diagnostics final_row = engine.diagnostics(true);
            std::printf("final energy: kinetic %.9g potential %.9g total %.9g\n",
                        final_row.kinetic, final_row.potential, final_row.total);
            std::printf("final momentum magnitude: %.6g\n", final_row.momentum_magnitude);
        }
        return 0;
    } catch (const cps::ConvertInputError& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
