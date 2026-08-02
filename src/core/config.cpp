#include "core/config.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cps {
namespace {

using json = nlohmann::json;

std::string lower(std::string_view text) {
    std::string result(text);
    for (char& c : result) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return result;
}

void read_vec3(const json& node, const char* key, Vec3& target) {
    if (!node.contains(key)) {
        return;
    }
    const json& value = node.at(key);
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(std::string("configuration field '") + key +
                                 "' must be an array of three numbers");
    }
    target.x = value.at(0).get<float>();
    target.y = value.at(1).get<float>();
    target.z = value.at(2).get<float>();
}

template <typename T>
void read_field(const json& node, const char* key, T& target) {
    if (node.contains(key)) {
        target = node.at(key).get<T>();
    }
}

json vec3_to_json(const Vec3& v) { return json::array({v.x, v.y, v.z}); }

}  // namespace

std::string to_string(SolverType value) {
    switch (value) {
        case SolverType::NBody:
            return "nbody";
        case SolverType::Sph:
            return "sph";
    }
    return "unknown";
}

std::string to_string(NBodyVariant value) {
    switch (value) {
        case NBodyVariant::Naive:
            return "naive";
        case NBodyVariant::Tiled:
            return "tiled";
        case NBodyVariant::Fused:
            return "fused";
    }
    return "unknown";
}

std::string to_string(InitialCondition value) {
    switch (value) {
        case InitialCondition::TwoBody:
            return "two_body";
        case InitialCondition::Plummer:
            return "plummer";
        case InitialCondition::DamBreak:
            return "dam_break";
    }
    return "unknown";
}

SolverType solver_from_string(std::string_view text) {
    const std::string key = lower(text);
    if (key == "nbody" || key == "n_body") {
        return SolverType::NBody;
    }
    if (key == "sph" || key == "fluid") {
        return SolverType::Sph;
    }
    throw std::runtime_error("unknown solver '" + std::string(text) + "', expected nbody or sph");
}

NBodyVariant variant_from_string(std::string_view text) {
    const std::string key = lower(text);
    if (key == "naive") {
        return NBodyVariant::Naive;
    }
    if (key == "tiled") {
        return NBodyVariant::Tiled;
    }
    if (key == "fused") {
        return NBodyVariant::Fused;
    }
    throw std::runtime_error("unknown variant '" + std::string(text) +
                             "', expected naive, tiled or fused");
}

InitialCondition initial_condition_from_string(std::string_view text) {
    const std::string key = lower(text);
    if (key == "two_body" || key == "twobody") {
        return InitialCondition::TwoBody;
    }
    if (key == "plummer") {
        return InitialCondition::Plummer;
    }
    if (key == "dam_break" || key == "dambreak") {
        return InitialCondition::DamBreak;
    }
    throw std::runtime_error("unknown initial condition '" + std::string(text) +
                             "', expected two_body, plummer or dam_break");
}

Config Config::from_json_string(std::string_view text) {
    json root;
    try {
        root = json::parse(text);
    } catch (const json::parse_error& error) {
        throw std::runtime_error(std::string("configuration is not valid JSON: ") + error.what());
    }
    if (!root.is_object()) {
        throw std::runtime_error("configuration must be a JSON object");
    }

    Config config;
    read_field(root, "name", config.name);

    if (root.contains("solver")) {
        config.solver = solver_from_string(root.at("solver").get<std::string>());
    }
    if (root.contains("variant")) {
        config.variant = variant_from_string(root.at("variant").get<std::string>());
    }
    if (root.contains("initial_condition")) {
        config.initial_condition =
            initial_condition_from_string(root.at("initial_condition").get<std::string>());
    }

    read_field(root, "num_particles", config.num_particles);
    read_field(root, "steps", config.steps);
    read_field(root, "dt", config.dt);
    read_field(root, "softening", config.softening);
    read_field(root, "gravitational_constant", config.gravitational_constant);
    read_field(root, "seed", config.seed);
    read_field(root, "block_size", config.block_size);
    read_field(root, "total_mass", config.total_mass);
    read_field(root, "plummer_radius", config.plummer_radius);
    read_field(root, "two_body_separation", config.two_body_separation);
    read_vec3(root, "external_gravity", config.external_gravity);
    read_field(root, "snapshot_interval", config.snapshot_interval);
    read_field(root, "snapshot_dir", config.snapshot_dir);
    read_field(root, "diagnostics_interval", config.diagnostics_interval);
    read_field(root, "diagnostics_path", config.diagnostics_path);

    if (root.contains("sph")) {
        const json& node = root.at("sph");
        if (!node.is_object()) {
            throw std::runtime_error("configuration field 'sph' must be an object");
        }
        read_field(node, "smoothing_radius", config.sph.smoothing_radius);
        read_field(node, "rest_density", config.sph.rest_density);
        read_field(node, "gas_constant", config.sph.gas_constant);
        read_field(node, "viscosity", config.sph.viscosity);
        read_field(node, "particle_mass", config.sph.particle_mass);
        read_field(node, "boundary_damping", config.sph.boundary_damping);
        read_field(node, "jitter", config.sph.jitter);
        read_vec3(node, "domain_min", config.sph.domain_min);
        read_vec3(node, "domain_max", config.sph.domain_max);
        read_vec3(node, "fluid_min", config.sph.fluid_min);
        read_vec3(node, "fluid_max", config.sph.fluid_max);
    }

    config.validate();
    return config;
}

Config Config::from_json_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open configuration file " + path.string());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    try {
        return from_json_string(buffer.str());
    } catch (const std::runtime_error& error) {
        throw std::runtime_error(path.string() + ": " + error.what());
    }
}

void Config::validate() const {
    if (num_particles < 2) {
        throw std::runtime_error("num_particles must be at least 2");
    }
    if (steps < 0) {
        throw std::runtime_error("steps must not be negative");
    }
    if (!(dt > 0.0f)) {
        throw std::runtime_error("dt must be positive");
    }
    if (softening < 0.0f) {
        throw std::runtime_error("softening must not be negative");
    }
    if (block_size <= 0 || block_size % 32 != 0 || block_size > 1024) {
        throw std::runtime_error("block_size must be a positive multiple of 32 up to 1024");
    }
    if (solver == SolverType::NBody && initial_condition == InitialCondition::DamBreak) {
        throw std::runtime_error("the dam_break initial condition belongs to the sph solver");
    }
    if (solver == SolverType::Sph && initial_condition != InitialCondition::DamBreak) {
        throw std::runtime_error("the sph solver expects the dam_break initial condition");
    }
    if (initial_condition == InitialCondition::TwoBody && num_particles != 2) {
        throw std::runtime_error("the two_body initial condition requires num_particles equal to 2");
    }
    if (solver == SolverType::Sph) {
        if (!(sph.rest_density > 0.0f)) {
            throw std::runtime_error("sph.rest_density must be positive");
        }
        if (!(sph.gas_constant > 0.0f)) {
            throw std::runtime_error("sph.gas_constant must be positive");
        }
        if (sph.viscosity < 0.0f) {
            throw std::runtime_error("sph.viscosity must not be negative");
        }
        const bool ordered = sph.fluid_min.x < sph.fluid_max.x && sph.fluid_min.y < sph.fluid_max.y &&
                             sph.fluid_min.z < sph.fluid_max.z;
        if (!ordered) {
            throw std::runtime_error("sph.fluid_min must be strictly below sph.fluid_max");
        }
        const bool inside = sph.fluid_min.x >= sph.domain_min.x && sph.fluid_min.y >= sph.domain_min.y &&
                            sph.fluid_min.z >= sph.domain_min.z && sph.fluid_max.x <= sph.domain_max.x &&
                            sph.fluid_max.y <= sph.domain_max.y && sph.fluid_max.z <= sph.domain_max.z;
        if (!inside) {
            throw std::runtime_error("the initial fluid block must lie inside the domain");
        }
    }
}

std::string Config::to_json_string(int indent) const {
    json root;
    root["name"] = name;
    root["solver"] = to_string(solver);
    root["variant"] = to_string(variant);
    root["initial_condition"] = to_string(initial_condition);
    root["num_particles"] = num_particles;
    root["steps"] = steps;
    root["dt"] = dt;
    root["softening"] = softening;
    root["gravitational_constant"] = gravitational_constant;
    root["seed"] = seed;
    root["block_size"] = block_size;
    root["total_mass"] = total_mass;
    root["plummer_radius"] = plummer_radius;
    root["two_body_separation"] = two_body_separation;
    root["external_gravity"] = vec3_to_json(external_gravity);
    root["snapshot_interval"] = snapshot_interval;
    root["snapshot_dir"] = snapshot_dir;
    root["diagnostics_interval"] = diagnostics_interval;
    root["diagnostics_path"] = diagnostics_path;

    json sph_node;
    sph_node["smoothing_radius"] = sph.smoothing_radius;
    sph_node["rest_density"] = sph.rest_density;
    sph_node["gas_constant"] = sph.gas_constant;
    sph_node["viscosity"] = sph.viscosity;
    sph_node["particle_mass"] = sph.particle_mass;
    sph_node["boundary_damping"] = sph.boundary_damping;
    sph_node["jitter"] = sph.jitter;
    sph_node["domain_min"] = vec3_to_json(sph.domain_min);
    sph_node["domain_max"] = vec3_to_json(sph.domain_max);
    sph_node["fluid_min"] = vec3_to_json(sph.fluid_min);
    sph_node["fluid_max"] = vec3_to_json(sph.fluid_max);
    root["sph"] = sph_node;

    return root.dump(indent);
}

}  // namespace cps
