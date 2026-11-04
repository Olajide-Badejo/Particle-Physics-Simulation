#include "cps/core/convert.hpp"

#include <cstdint>
#include <exception>
#include <fstream>
#include <vector>

#include "cps/core/config.hpp"
#include "cps/core/physics_fingerprint.hpp"
#include "cps/core/snapshot.hpp"

namespace cps {
namespace {

Config load_valid_config(const std::filesystem::path& path) {
    Config config = Config::from_json_file(path);
    config.validate();
    return config;
}

std::string convert_snapshot(const std::filesystem::path& input,
                             const std::filesystem::path& output, const std::string& config_path) {
    if (config_path.empty()) {
        throw ConvertInputError(input.string() +
                                " is a CPS1 snapshot, which records no configuration: "
                                "converting it needs --config <file> to compute a fingerprint");
    }
    Snapshot snapshot;
    Config config;
    try {
        snapshot = read_snapshot(input);
        config = load_valid_config(config_path);
    } catch (const std::exception& error) {
        throw ConvertInputError(error.what());
    }
    if (snapshot.state.size() != static_cast<std::size_t>(config.num_particles)) {
        throw ConvertInputError("snapshot " + input.string() + " has " +
                                std::to_string(snapshot.state.size()) +
                                " particles but --config " + config_path + " says " +
                                std::to_string(config.num_particles));
    }
    const std::vector<PhysicsField> fields = physics_fields(config);
    snapshot.fingerprint = physics_fingerprint(fields);
    snapshot.fingerprint_version = PHYSICS_FINGERPRINT_VERSION;
    snapshot.physics = physics_json(fields);
    write_snapshot(output, snapshot);
    return "CPS1 snapshot converted to CPS2";
}

std::string convert_config(const std::filesystem::path& input, const std::filesystem::path& output,
                           const std::string& config_path) {
    if (!config_path.empty()) {
        throw ConvertInputError("--config applies only to a CPS1 snapshot, and " + input.string() +
                                " is not one");
    }
    Config config;
    try {
        config = load_valid_config(input);
    } catch (const std::exception& error) {
        throw ConvertInputError(input.string() +
                                " is neither a valid configuration nor a CPS1 snapshot: " +
                                error.what());
    }
    std::ofstream file(output, std::ios::binary);
    file << config.to_json_string() << '\n';
    file.close();
    if (!file) {
        throw std::runtime_error("cannot write " + output.string());
    }
    return "configuration converted to schema 2";
}

}  // namespace

std::string convert_file(const std::filesystem::path& input, const std::filesystem::path& output,
                         const std::string& config_path) {
    std::ifstream file(input, std::ios::binary);
    if (!file) {
        throw ConvertInputError("cannot read " + input.string());
    }
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.close();
    if (magic == SNAPSHOT_MAGIC) {
        throw ConvertInputError(input.string() + " is already a CPS2 snapshot");
    }
    return magic == SNAPSHOT_MAGIC_V1 ? convert_snapshot(input, output, config_path)
                                      : convert_config(input, output, config_path);
}

}  // namespace cps
