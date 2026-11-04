#include "cps/core/cli.hpp"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace cps {
namespace {

std::string require_value(int argc, const char* const* argv, int& index, const char* flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("option ") + flag + " needs a value");
    }
    ++index;
    return argv[index];
}

// The whole text must be a base 10 integer in [low, high]. Unlike std::stoll
// this rejects trailing junk such as "10abc" and names the flag on failure.
long long parse_integer(const std::string& text, const char* flag, long long low, long long high) {
    const bool starts_cleanly = !text.empty() && std::isspace(static_cast<unsigned char>(text[0])) == 0;
    char* end = nullptr;
    errno = 0;
    const long long value = starts_cleanly ? std::strtoll(text.c_str(), &end, 10) : 0;
    if (!starts_cleanly || end != text.c_str() + text.size()) {
        throw std::runtime_error(std::string("option ") + flag + " expects an integer, got '" + text +
                                 "'");
    }
    if (errno == ERANGE || value < low || value > high) {
        throw std::runtime_error(std::string("option ") + flag + " must be between " +
                                 std::to_string(low) + " and " + std::to_string(high) + ", got '" +
                                 text + "'");
    }
    return value;
}

// The whole text must be a number that is positive and finite once narrowed
// to float, which is how the time step is stored.
float parse_positive_float(const std::string& text, const char* flag) {
    const bool starts_cleanly = !text.empty() && std::isspace(static_cast<unsigned char>(text[0])) == 0;
    char* end = nullptr;
    const double value = starts_cleanly ? std::strtod(text.c_str(), &end) : 0.0;
    if (!starts_cleanly || end != text.c_str() + text.size()) {
        throw std::runtime_error(std::string("option ") + flag + " expects a number, got '" + text +
                                 "'");
    }
    const float narrowed = static_cast<float>(value);
    if (!(std::isfinite(narrowed) && narrowed > 0.0f)) {
        throw std::runtime_error(std::string("option ") + flag +
                                 " must be a positive finite number, got '" + text + "'");
    }
    return narrowed;
}

bool is_help(const std::string& arg) { return arg == "--help" || arg == "-h"; }

const char* const RUN_OPTIONS =
    "options:\n"
    "  --solver <name>       override the solver: nbody or sph\n"
    "  --variant <name>      override the N-body variant: naive, tiled or fused\n"
    "  --steps <n>           override the step count\n"
    "  --particles <n>       override the particle count\n"
    "  --dt <value>          override the time step\n"
    "  --allow-unstable-dt   warn instead of refusing a fluid time step above its\n"
    "                        stability limits\n"
    "  --graphs <on|off>     launch the step as a captured CUDA graph, or not\n"
    "  --seed <n>            override the random seed\n"
    "  --snapshot-dir <path> override the snapshot directory\n"
    "  --snapshot-every <n>  override the snapshot cadence, 0 disables\n"
    "  --diagnostics <path>  override the diagnostics file, empty disables\n"
    "  --diag-every <n>      override the diagnostics cadence, 0 disables\n"
    "  --vtk-every <n>       write a VTK frame every n steps, 0 disables\n"
    "  --vtk-dir <path>      override the directory the VTK frames and series go to\n"
    "  --resume              continue from the newest snapshot in the snapshot directory\n"
    "  --force-resume        resume even if the snapshot's physics differ from the\n"
    "                        configuration (implies --resume)\n"
    "  --print-config        print the resolved configuration and exit\n";

template <typename Parse>
auto with_flag(const char* flag, Parse parse) {
    try {
        return parse();
    } catch (const std::runtime_error& error) {
        throw std::runtime_error(std::string("option ") + flag + ": " + error.what());
    }
}

}  // namespace

std::string cli_usage() {
    return std::string(
               "cps: CUDA particle physics engine\n"
               "\n"
               "usage: cps <command> [arguments]\n"
               "\n"
               "commands:\n"
               "  run <preset-or-path> [options]  run an embedded preset or a configuration file\n"
               "  presets                         list the embedded presets\n"
               "  info                            print the version, commit and device properties\n"
               "  convert <input> <output>        turn a v1 configuration into schema 2, or a CPS1\n"
               "                                  snapshot into CPS2 (needs --config <file>)\n"
               "  --version                       print the version and commit, then exit\n"
               "  --help                          print this message; after a command, its help\n"
               "\n"
               "cps run ") +
           RUN_OPTIONS +
           "\n"
           "old form, going away in the next major version, also under the name\n"
           "cuda_physics_sim:\n"
           "  --config <path>       the file that cps run takes as its argument\n"
           "  --device-info         print the device properties and exit; use cps info\n";
}

std::string cli_usage(CliCommand command) {
    switch (command) {
        case CliCommand::Run:
            return std::string(
                       "usage: cps run <preset-or-path> [options]\n"
                       "\n"
                       "A name with no path separator and no .json is an embedded preset, which\n"
                       "cps presets lists. Anything else is a configuration file.\n"
                       "\n") +
                   RUN_OPTIONS;
        case CliCommand::Presets:
            return "usage: cps presets\n"
                   "\n"
                   "Lists every embedded preset with its solver, particle count and steps.\n";
        case CliCommand::Info:
            return "usage: cps info\n"
                   "\n"
                   "Prints the version, the commit and the device properties.\n";
        case CliCommand::Convert:
            return "usage: cps convert <input> <output> [--config <file>]\n"
                   "\n"
                   "What happens depends on what the input is:\n"
                   "  a configuration file  is written out in schema 2\n"
                   "  a CPS1 snapshot       is written out as CPS2. CPS1 records no configuration,\n"
                   "                        so --config names the one to fingerprint it with\n";
        case CliCommand::Legacy:
        case CliCommand::Usage:
            break;
    }
    return cli_usage();
}

std::string legacy_form_notice() {
    return "notice: this form goes away in the next major version; use 'cps run <preset-or-path>' "
           "in place of --config, and 'cps info' in place of --device-info";
}

bool run_target_is_preset(const std::string& target) {
    return target.find_first_of("/\\") == std::string::npos &&
           !(target.size() >= 5 && target.compare(target.size() - 5, 5, ".json") == 0);
}

namespace {

// Parses the flags from argv[first] on.
CliOptions parse_flags(int first, int argc, const char* const* argv) {
    constexpr long long kLongMax = std::numeric_limits<long long>::max();
    CliOptions options;
    for (int i = first; i < argc; ++i) {
        const std::string arg = argv[i];
        if (is_help(arg)) {
            options.show_help = true;
            return options;
        } else if (arg == "--version") {
            options.show_version = true;
            return options;
        } else if (arg == "--config") {
            options.config_path = require_value(argc, argv, i, "--config");
        } else if (arg == "--solver") {
            const std::string text = require_value(argc, argv, i, "--solver");
            options.solver = with_flag("--solver", [&] { return solver_from_string(text); });
        } else if (arg == "--variant") {
            const std::string text = require_value(argc, argv, i, "--variant");
            options.variant = with_flag("--variant", [&] { return variant_from_string(text); });
        } else if (arg == "--steps") {
            options.steps = parse_integer(require_value(argc, argv, i, "--steps"), "--steps", 0, kLongMax);
        } else if (arg == "--particles") {
            options.particles = static_cast<int>(
                parse_integer(require_value(argc, argv, i, "--particles"), "--particles", 1, INT_MAX));
        } else if (arg == "--dt") {
            options.dt = parse_positive_float(require_value(argc, argv, i, "--dt"), "--dt");
        } else if (arg == "--allow-unstable-dt") {
            options.allow_unstable_dt = true;
        } else if (arg == "--graphs") {
            const std::string text = require_value(argc, argv, i, "--graphs");
            if (text != "on" && text != "off") {
                throw std::runtime_error("option --graphs expects on or off, got '" + text + "'");
            }
            options.use_graphs = text == "on";
        } else if (arg == "--seed") {
            options.seed = static_cast<uint32_t>(
                parse_integer(require_value(argc, argv, i, "--seed"), "--seed", 0, UINT32_MAX));
        } else if (arg == "--snapshot-dir") {
            options.snapshot_dir = require_value(argc, argv, i, "--snapshot-dir");
        } else if (arg == "--snapshot-every") {
            options.snapshot_every = parse_integer(require_value(argc, argv, i, "--snapshot-every"),
                                                   "--snapshot-every", 0, kLongMax);
        } else if (arg == "--diagnostics") {
            options.diagnostics_path = require_value(argc, argv, i, "--diagnostics");
        } else if (arg == "--diag-every") {
            options.diag_every = parse_integer(require_value(argc, argv, i, "--diag-every"),
                                               "--diag-every", 0, kLongMax);
        } else if (arg == "--vtk-every") {
            options.vtk_every = parse_integer(require_value(argc, argv, i, "--vtk-every"),
                                              "--vtk-every", 0, kLongMax);
        } else if (arg == "--vtk-dir") {
            options.vtk_dir = require_value(argc, argv, i, "--vtk-dir");
            if (options.vtk_dir->empty()) {
                throw std::runtime_error("--vtk-dir needs a directory, got an empty path");
            }
        } else if (arg == "--resume") {
            options.resume = true;
        } else if (arg == "--force-resume") {
            options.resume = true;
            options.force_resume = true;
        } else if (arg == "--device-info") {
            options.device_info = true;
        } else if (arg == "--print-config") {
            options.print_config = true;
        } else {
            throw std::runtime_error("unknown option " + arg);
        }
    }
    return options;
}

CliOptions parse_run(int argc, const char* const* argv) {
    if (argc > 2 && is_help(argv[2])) {
        CliOptions options;
        options.command = CliCommand::Run;
        options.show_help = true;
        return options;
    }
    if (argc <= 2 || argv[2][0] == '-') {
        throw std::runtime_error(
            "cps run needs a preset name or a configuration file before any option; "
            "cps presets lists the presets");
    }
    CliOptions options = parse_flags(3, argc, argv);
    if (!options.config_path.empty()) {
        throw std::runtime_error("option --config has no meaning after cps run, which reads " +
                                 std::string(argv[2]));
    }
    options.command = CliCommand::Run;
    options.run_target = argv[2];
    return options;
}

CliOptions parse_no_arguments(CliCommand command, const char* name, int argc,
                              const char* const* argv) {
    CliOptions options;
    options.command = command;
    if (argc > 2 && is_help(argv[2])) {
        options.show_help = true;
    } else if (argc > 2) {
        throw std::runtime_error(std::string("cps ") + name + " takes no arguments, got '" +
                                 argv[2] + "'");
    }
    return options;
}

CliOptions parse_convert(int argc, const char* const* argv) {
    CliOptions options;
    options.command = CliCommand::Convert;
    std::vector<std::string> files;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (is_help(arg)) {
            options.show_help = true;
            return options;
        } else if (arg == "--config") {
            options.config_path = require_value(argc, argv, i, "--config");
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown option " + arg + " for cps convert");
        } else {
            files.push_back(arg);
        }
    }
    if (files.size() != 2) {
        throw std::runtime_error("cps convert needs an input and an output, got " +
                                 std::to_string(files.size()) + " paths");
    }
    options.convert_input = files[0];
    options.convert_output = files[1];
    return options;
}

}  // namespace

CliOptions parse_cli(int argc, const char* const* argv) { return parse_flags(1, argc, argv); }

CliOptions parse_command_line(int argc, const char* const* argv) {
    if (argc < 2) {
        CliOptions options;
        options.command = CliCommand::Usage;
        return options;
    }
    const std::string first = argv[1];
    if (!first.empty() && first[0] == '-') {
        return parse_cli(argc, argv);
    }
    if (first == "run") {
        return parse_run(argc, argv);
    }
    if (first == "presets") {
        return parse_no_arguments(CliCommand::Presets, "presets", argc, argv);
    }
    if (first == "info") {
        return parse_no_arguments(CliCommand::Info, "info", argc, argv);
    }
    if (first == "convert") {
        return parse_convert(argc, argv);
    }
    throw std::runtime_error("unknown subcommand '" + first +
                             "'; the choices are run, presets, info and convert");
}

void apply_cli_overrides(const CliOptions& options, Config& config) {
    if (options.solver) {
        config.solver = *options.solver;
    }
    if (options.variant) {
        config.variant = *options.variant;
    }
    if (options.steps) {
        config.steps = *options.steps;
    }
    if (options.particles) {
        config.num_particles = *options.particles;
    }
    if (options.dt) {
        config.dt = *options.dt;
    }
    if (options.allow_unstable_dt) {
        config.allow_unstable_dt = true;
    }
    if (options.use_graphs) {
        config.use_graphs = *options.use_graphs;
    }
    if (options.seed) {
        config.seed = *options.seed;
    }
    // An empty snapshot directory keeps the file's value, as before.
    if (options.snapshot_dir && !options.snapshot_dir->empty()) {
        config.snapshot_dir = *options.snapshot_dir;
    }
    if (options.snapshot_every) {
        config.snapshot_interval = *options.snapshot_every;
    }
    // An empty diagnostics path is meaningful: it disables the file.
    if (options.diagnostics_path) {
        config.diagnostics_path = *options.diagnostics_path;
    }
    if (options.diag_every) {
        config.diagnostics_interval = *options.diag_every;
    }
    if (options.vtk_every) {
        config.vtk_interval = *options.vtk_every;
    }
    if (options.vtk_dir) {
        config.vtk_dir = *options.vtk_dir;
    }
}

}  // namespace cps
