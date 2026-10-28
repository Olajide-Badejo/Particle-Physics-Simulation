#include "core/cli.hpp"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

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
    return "cuda_physics_sim: CUDA particle physics engine\n"
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
           "  --help                print this message\n";
}

CliOptions parse_cli(int argc, const char* const* argv) {
    constexpr long long kLongMax = std::numeric_limits<long long>::max();
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            options.show_help = true;
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
        } else if (arg == "--resume") {
            options.resume = true;
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
}

}  // namespace cps
