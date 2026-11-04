#include "cps/core/config.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

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

// Reads the fields of one JSON object and remembers every key it was asked
// for, present or not. The allow-list for unknown key detection is exactly that
// record, so adding a read allows the key and the two cannot drift apart.
class ObjectReader {
public:
    ObjectReader(const json& node, std::string prefix) : node_(node), prefix_(std::move(prefix)) {}

    std::string path(const char* key) const { return prefix_ + key; }

    // Returns the value for key, or nullptr when the key is absent.
    const json* find(const char* key) {
        known_.emplace_back(key);
        const auto it = node_.find(key);
        return it == node_.end() ? nullptr : &*it;
    }

    void read(const char* key, std::string& target) {
        if (const json* value = find(key)) {
            if (!value->is_string()) {
                throw std::runtime_error("configuration field '" + path(key) +
                                         "' must be a string, got " + value->dump());
            }
            target = value->get<std::string>();
        }
    }

    void read(const char* key, bool& target) {
        if (const json* value = find(key)) {
            if (!value->is_boolean()) {
                throw std::runtime_error("configuration field '" + path(key) +
                                         "' must be true or false, got " + value->dump());
            }
            target = value->get<bool>();
        }
    }

    void read(const char* key, float& target) {
        if (const json* value = find(key)) {
            target = number(*value, path(key));
        }
    }

    // Integers must be JSON integers. A fractional value such as 10.5 is
    // rejected rather than truncated, and an out of range value is rejected
    // rather than wrapped.
    template <typename T>
        requires std::is_integral_v<T>
    void read(const char* key, T& target) {
        const json* value = find(key);
        if (value == nullptr) {
            return;
        }
        const std::string field = path(key);
        if (!value->is_number_integer()) {
            throw std::runtime_error("configuration field '" + field +
                                     "' must be an integer, got " + value->dump());
        }
        bool in_range = false;
        if (value->is_number_unsigned()) {
            in_range = std::cmp_less_equal(value->get<std::uint64_t>(), std::numeric_limits<T>::max());
        } else {
            const auto raw = value->get<std::int64_t>();
            in_range = std::cmp_greater_equal(raw, std::numeric_limits<T>::min()) &&
                       std::cmp_less_equal(raw, std::numeric_limits<T>::max());
        }
        if (!in_range) {
            throw std::runtime_error("configuration field '" + field + "' is out of range, got " +
                                     value->dump());
        }
        target = value->get<T>();
    }

    void read(const char* key, Vec3& target) {
        const json* value = find(key);
        if (value == nullptr) {
            return;
        }
        const std::string field = path(key);
        if (!value->is_array() || value->size() != 3) {
            throw std::runtime_error("configuration field '" + field +
                                     "' must be an array of three numbers, got " + value->dump());
        }
        target.x = number(value->at(0), field + "[0]");
        target.y = number(value->at(1), field + "[1]");
        target.z = number(value->at(2), field + "[2]");
    }

    // Every key asked for so far, with its full path.
    std::vector<std::string> known_paths() const {
        std::vector<std::string> paths;
        for (const std::string& key : known_) {
            paths.push_back(prefix_ + key);
        }
        return paths;
    }

    // Throws for the first key that no read asked for, suggesting the closest
    // known key when one is near enough to be a likely typo. Keys at this level
    // are suggested by name; elsewhere holds full paths from other levels, so
    // that a fluid key placed at the top level points at its real home.
    void reject_unknown_keys(const std::vector<std::string>& elsewhere = {}) const {
        for (const auto& item : node_.items()) {
            if (std::find(known_.begin(), known_.end(), item.key()) != known_.end()) {
                continue;
            }
            std::string message = "unknown configuration key '" + prefix_ + item.key() + "'";
            std::string suggestion = closest(item.key(), known_);
            if (suggestion.empty()) {
                suggestion = closest(item.key(), elsewhere, true);
            }
            if (!suggestion.empty()) {
                message += " (did you mean '" + suggestion + "'?)";
            }
            throw std::runtime_error(message);
        }
    }

private:
    static float number(const json& value, const std::string& field) {
        if (!value.is_number()) {
            throw std::runtime_error("configuration field '" + field + "' must be a number, got " +
                                     value.dump());
        }
        // Narrowing a double beyond the float range is undefined, so the range
        // is checked first rather than relying on the conversion giving Inf.
        const double wide = value.get<double>();
        if (!(std::fabs(wide) <= static_cast<double>(std::numeric_limits<float>::max()))) {
            throw std::runtime_error("configuration field '" + field +
                                     "' is out of range for a float, got " + value.dump());
        }
        return static_cast<float>(wide);
    }

    // With leaf_only, a candidate such as "sph.viscosity" is compared by its
    // last component but returned whole.
    static std::string closest(const std::string& key, const std::vector<std::string>& candidates,
                               bool leaf_only = false) {
        std::string best;
        std::size_t best_distance = std::numeric_limits<std::size_t>::max();
        for (const std::string& candidate : candidates) {
            const std::size_t dot = leaf_only ? candidate.rfind('.') : std::string::npos;
            const std::string leaf = dot == std::string::npos ? candidate : candidate.substr(dot + 1);
            const std::size_t distance = edit_distance(key, leaf);
            if (distance < best_distance) {
                best_distance = distance;
                best = candidate;
            }
        }
        // Allow roughly one edit per three characters, and never fewer than two.
        const std::size_t limit = std::max<std::size_t>(2, key.size() / 3);
        return best_distance <= limit ? best : std::string();
    }

    // Levenshtein distance with a single rolling row.
    static std::size_t edit_distance(const std::string& a, const std::string& b) {
        std::vector<std::size_t> row(b.size() + 1);
        for (std::size_t j = 0; j <= b.size(); ++j) {
            row[j] = j;
        }
        for (std::size_t i = 1; i <= a.size(); ++i) {
            std::size_t diagonal = row[0];
            row[0] = i;
            for (std::size_t j = 1; j <= b.size(); ++j) {
                const std::size_t above = row[j];
                const std::size_t substitution = diagonal + (a[i - 1] == b[j - 1] ? 0 : 1);
                row[j] = std::min({above + 1, row[j - 1] + 1, substitution});
                diagonal = above;
            }
        }
        return row[b.size()];
    }

    const json& node_;
    std::string prefix_;
    std::vector<std::string> known_;
};

std::string format_value(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.9g", value);
    return buffer;
}

std::string format_vec3(const Vec3& v) {
    return "[" + format_value(v.x) + ", " + format_value(v.y) + ", " + format_value(v.z) + "]";
}

// NaN fails every comparison, so each check is written with NaN landing on the
// throwing side.
void require_positive(const char* field, float value) {
    if (!(std::isfinite(value) && value > 0.0f)) {
        throw std::runtime_error(std::string(field) + " must be a positive finite number, got " +
                                 format_value(value));
    }
}

void require_non_negative(const char* field, float value) {
    if (!(std::isfinite(value) && value >= 0.0f)) {
        throw std::runtime_error(std::string(field) +
                                 " must be a finite number that is not negative, got " +
                                 format_value(value));
    }
}

void require_finite(const char* field, const Vec3& v) {
    if (!(std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z))) {
        throw std::runtime_error(std::string(field) + " must have finite components, got " +
                                 format_vec3(v));
    }
}

bool strictly_below(const Vec3& a, const Vec3& b) { return a.x < b.x && a.y < b.y && a.z < b.z; }

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
    ObjectReader top(root, "");
    top.read("name", config.name);

    std::string text_value;
    if (top.find("solver") != nullptr) {
        top.read("solver", text_value);
        config.solver = solver_from_string(text_value);
    }
    if (top.find("variant") != nullptr) {
        top.read("variant", text_value);
        config.variant = variant_from_string(text_value);
    }
    if (top.find("initial_condition") != nullptr) {
        top.read("initial_condition", text_value);
        config.initial_condition = initial_condition_from_string(text_value);
    }

    top.read("num_particles", config.num_particles);
    top.read("steps", config.steps);
    top.read("dt", config.dt);
    top.read("allow_unstable_dt", config.allow_unstable_dt);
    top.read("softening", config.softening);
    top.read("gravitational_constant", config.gravitational_constant);
    top.read("seed", config.seed);
    top.read("block_size", config.block_size);
    top.read("total_mass", config.total_mass);
    top.read("plummer_radius", config.plummer_radius);
    top.read("two_body_separation", config.two_body_separation);
    top.read("external_gravity", config.external_gravity);
    top.read("snapshot_interval", config.snapshot_interval);
    top.read("snapshot_dir", config.snapshot_dir);
    top.read("diagnostics_interval", config.diagnostics_interval);
    top.read("diagnostics_path", config.diagnostics_path);

    // The nested readers run even when their block is absent, reading nothing,
    // so their key lists are available for suggestions at the top level.
    const json* sph_node = top.find("sph");
    if (sph_node != nullptr && !sph_node->is_object()) {
        throw std::runtime_error("configuration field 'sph' must be an object, got " +
                                 sph_node->dump());
    }
    const json* nbody_node = top.find("nbody");
    if (nbody_node != nullptr && !nbody_node->is_object()) {
        throw std::runtime_error("configuration field 'nbody' must be an object, got " +
                                 nbody_node->dump());
    }
    const json empty_object = json::object();
    {
        ObjectReader gravity(nbody_node != nullptr ? *nbody_node : empty_object, "nbody.");
        gravity.read("tile_size", config.nbody.tile_size);
        gravity.reject_unknown_keys();

        ObjectReader fluid(sph_node != nullptr ? *sph_node : empty_object, "sph.");
        fluid.read("smoothing_radius", config.sph.smoothing_radius);
        fluid.read("rest_density", config.sph.rest_density);
        fluid.read("gas_constant", config.sph.gas_constant);
        fluid.read("viscosity", config.sph.viscosity);
        fluid.read("particle_mass", config.sph.particle_mass);
        fluid.read("boundary_damping", config.sph.boundary_damping);
        fluid.read("jitter", config.sph.jitter);
        fluid.read("domain_min", config.sph.domain_min);
        fluid.read("domain_max", config.sph.domain_max);
        fluid.read("fluid_min", config.sph.fluid_min);
        fluid.read("fluid_max", config.sph.fluid_max);
        fluid.reject_unknown_keys();

        std::vector<std::string> nested = gravity.known_paths();
        const std::vector<std::string> fluid_paths = fluid.known_paths();
        nested.insert(nested.end(), fluid_paths.begin(), fluid_paths.end());
        top.reject_unknown_keys(nested);
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
        throw std::runtime_error("num_particles must be at least 2, got " +
                                 std::to_string(num_particles));
    }
    if (steps < 0) {
        throw std::runtime_error("steps must not be negative, got " + std::to_string(steps));
    }
    require_positive("dt", dt);
    // The N-body kernels sum every pair including a particle with itself, and
    // only the softening keeps that zero over zero finite.
    if (solver == SolverType::NBody) {
        require_positive("softening", softening);
    } else {
        require_non_negative("softening", softening);
    }
    require_positive("gravitational_constant", gravitational_constant);
    if (block_size <= 0 || block_size % 32 != 0 || block_size > 1024) {
        throw std::runtime_error("block_size must be a positive multiple of 32 up to 1024, got " +
                                 std::to_string(block_size));
    }
    require_positive("total_mass", total_mass);
    require_positive("plummer_radius", plummer_radius);
    require_positive("two_body_separation", two_body_separation);
    require_finite("external_gravity", external_gravity);
    if (nbody.tile_size != 32 && nbody.tile_size != 64 && nbody.tile_size != 128 &&
        nbody.tile_size != 256) {
        throw std::runtime_error("nbody.tile_size must be 32, 64, 128 or 256, got " +
                                 std::to_string(nbody.tile_size));
    }
    if (snapshot_interval < 0) {
        throw std::runtime_error("snapshot_interval must not be negative, got " +
                                 std::to_string(snapshot_interval));
    }
    if (diagnostics_interval < 0) {
        throw std::runtime_error("diagnostics_interval must not be negative, got " +
                                 std::to_string(diagnostics_interval));
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

    // The fluid block is checked whatever the solver. Its defaults are valid,
    // so this costs nothing, and a bad value cannot hide until a solver switch.
    require_non_negative("sph.smoothing_radius", sph.smoothing_radius);
    require_positive("sph.rest_density", sph.rest_density);
    require_positive("sph.gas_constant", sph.gas_constant);
    require_non_negative("sph.viscosity", sph.viscosity);
    require_non_negative("sph.particle_mass", sph.particle_mass);
    // Damping above 1 would add energy at every wall contact.
    if (!(sph.boundary_damping >= 0.0f && sph.boundary_damping <= 1.0f)) {
        throw std::runtime_error("sph.boundary_damping must lie in [0, 1], got " +
                                 format_value(sph.boundary_damping));
    }
    // Half a spacing or more would let neighbouring lattice sites overlap.
    if (!(sph.jitter >= 0.0f && sph.jitter < 0.5f)) {
        throw std::runtime_error("sph.jitter must lie in [0, 0.5), got " + format_value(sph.jitter));
    }
    require_finite("sph.domain_min", sph.domain_min);
    require_finite("sph.domain_max", sph.domain_max);
    require_finite("sph.fluid_min", sph.fluid_min);
    require_finite("sph.fluid_max", sph.fluid_max);
    if (!strictly_below(sph.domain_min, sph.domain_max)) {
        throw std::runtime_error("sph.domain_min must be strictly below sph.domain_max, got " +
                                 format_vec3(sph.domain_min) + " and " + format_vec3(sph.domain_max));
    }
    if (!strictly_below(sph.fluid_min, sph.fluid_max)) {
        throw std::runtime_error("sph.fluid_min must be strictly below sph.fluid_max, got " +
                                 format_vec3(sph.fluid_min) + " and " + format_vec3(sph.fluid_max));
    }
    const bool inside = sph.fluid_min.x >= sph.domain_min.x && sph.fluid_min.y >= sph.domain_min.y &&
                        sph.fluid_min.z >= sph.domain_min.z && sph.fluid_max.x <= sph.domain_max.x &&
                        sph.fluid_max.y <= sph.domain_max.y && sph.fluid_max.z <= sph.domain_max.z;
    if (!inside) {
        throw std::runtime_error("the initial fluid block " + format_vec3(sph.fluid_min) + " to " +
                                 format_vec3(sph.fluid_max) + " must lie inside the domain " +
                                 format_vec3(sph.domain_min) + " to " + format_vec3(sph.domain_max));
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
    root["allow_unstable_dt"] = allow_unstable_dt;
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

    json nbody_node;
    nbody_node["tile_size"] = nbody.tile_size;
    root["nbody"] = nbody_node;

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
