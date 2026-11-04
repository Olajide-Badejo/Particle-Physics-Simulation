#include "cps/core/config.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
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

    // Throws for the first key that no read asked for, suggesting a likely home.
    // The key's exact name in another section comes first, so that a flat dt
    // points at run.dt. Then the closest key at this level, suggested by name,
    // and then the closest in elsewhere, which holds full paths from every level.
    void reject_unknown_keys(const std::vector<std::string>& elsewhere = {}) const {
        for (const auto& item : node_.items()) {
            if (std::find(known_.begin(), known_.end(), item.key()) != known_.end()) {
                continue;
            }
            std::string message = "unknown configuration key '" + prefix_ + item.key() + "'";
            std::string suggestion = exact_leaf(item.key(), elsewhere);
            if (suggestion.empty()) {
                suggestion = closest(item.key(), known_);
            }
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

    static std::string exact_leaf(const std::string& key, const std::vector<std::string>& paths) {
        for (const std::string& path : paths) {
            if (path.ends_with("." + key)) {
                return path;
            }
        }
        return {};
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

// The layout the reader below understands and to_json_string writes.
constexpr int CURRENT_SCHEMA = 2;

json parse_root(std::string_view text) {
    json root;
    try {
        root = json::parse(text);
    } catch (const json::parse_error& error) {
        throw std::runtime_error(std::string("configuration is not valid JSON: ") + error.what());
    }
    if (!root.is_object()) {
        throw std::runtime_error("configuration must be a JSON object");
    }
    return root;
}

// A file with no schema key is schema 1.
int schema_of(const json& root) {
    const auto it = root.find("schema");
    if (it == root.end()) {
        return 1;
    }
    if (!it->is_number_integer()) {
        throw std::runtime_error(
            "configuration field 'schema' must be an integer, 1 or 2 are supported, got " +
            it->dump());
    }
    if (*it != 1 && *it != CURRENT_SCHEMA) {
        throw std::runtime_error("unsupported configuration schema " + it->dump() +
                                 ", 1 or 2 are supported");
    }
    return it->get<int>();
}

// Where each flat schema 1 key lives in schema 2.
struct MovedKey {
    const char* flat;
    const char* section;
    const char* leaf;
};

constexpr MovedKey MOVED_KEYS[] = {
    {"steps", "run", "steps"},
    {"dt", "run", "dt"},
    {"block_size", "run", "block_size"},
    {"allow_unstable_dt", "run", "allow_unstable_dt"},
    {"initial_condition", "initial_condition", "type"},
    {"num_particles", "initial_condition", "num_particles"},
    {"seed", "initial_condition", "seed"},
    {"total_mass", "initial_condition", "total_mass"},
    {"plummer_radius", "initial_condition", "plummer_radius"},
    {"two_body_separation", "initial_condition", "two_body_separation"},
    {"variant", "nbody", "variant"},
    {"softening", "nbody", "softening"},
    {"gravitational_constant", "nbody", "gravitational_constant"},
    // Only the fluid solver reads it. N-body has no external field.
    {"external_gravity", "sph", "external_gravity"},
    {"snapshot_interval", "output", "snapshot_interval"},
    {"snapshot_dir", "output", "snapshot_dir"},
    {"diagnostics_interval", "output", "diagnostics_interval"},
    {"diagnostics_path", "output", "diagnostics_path"},
};

[[noreturn]] void reject_schema_2_key_in_schema_1(const std::string& key) {
    throw std::runtime_error("unknown configuration key '" + key +
                             "' (a file without \"schema\": 2 is read as schema 1)");
}

// Rewrites the tree and nothing else, so that there is one reader and the two
// forms cannot drift apart. Unknown keys are copied across for that reader to
// refuse. A key that only schema 2 has is refused here, as it always was, since
// otherwise it would start to work in a file that does not ask for schema 2.
json migrate_schema_1(const json& v1) {
    for (const char* section : {"run", "output"}) {
        if (v1.contains(section)) {
            reject_schema_2_key_in_schema_1(section);
        }
    }
    json v2 = json::object();
    for (const auto& item : v1.items()) {
        const auto moved = std::find_if(std::begin(MOVED_KEYS), std::end(MOVED_KEYS),
                                        [&](const MovedKey& m) { return item.key() == m.flat; });
        if (moved == std::end(MOVED_KEYS) && item.key() != "schema") {
            v2[item.key()] = item.value();
        }
    }
    for (const MovedKey& moved : MOVED_KEYS) {
        const auto block = v2.find(moved.section);
        if (block != v2.end() && block->is_object() && block->contains(moved.leaf)) {
            reject_schema_2_key_in_schema_1(std::string(moved.section) + "." + moved.leaf);
        }
    }
    for (const MovedKey& moved : MOVED_KEYS) {
        const auto value = v1.find(moved.flat);
        if (value == v1.end()) {
            continue;
        }
        // A block that is not an object is left for the reader to refuse by name,
        // which it does before it looks at any stray key.
        json& block = v2[moved.section];
        if (block.is_null() || block.is_object()) {
            block[moved.leaf] = *value;
        } else {
            v2[moved.flat] = *value;
        }
    }
    v2["schema"] = CURRENT_SCHEMA;
    return v2;
}

// Reads schema 2, and is the only code that does.
Config read_schema_2(const json& root) {
    Config config;
    ObjectReader top(root, "");
    top.find("schema");
    top.read("name", config.name);

    std::string text_value;
    if (top.find("solver") != nullptr) {
        top.read("solver", text_value);
        config.solver = solver_from_string(text_value);
    }

    // Every section reader runs even when its section is absent, reading
    // nothing, so its key list is available for suggestions everywhere else.
    const json empty_object = json::object();
    const auto section = [&](const char* key) -> const json& {
        const json* node = top.find(key);
        if (node == nullptr) {
            return empty_object;
        }
        if (!node->is_object()) {
            throw std::runtime_error("configuration field '" + std::string(key) +
                                     "' must be an object, got " + node->dump());
        }
        return *node;
    };

    ObjectReader run(section("run"), "run.");
    run.read("steps", config.steps);
    run.read("dt", config.dt);
    run.read("block_size", config.block_size);
    run.read("allow_unstable_dt", config.allow_unstable_dt);
    run.read("use_graphs", config.use_graphs);

    ObjectReader initial(section("initial_condition"), "initial_condition.");
    if (initial.find("type") != nullptr) {
        initial.read("type", text_value);
        config.initial_condition = initial_condition_from_string(text_value);
    }
    initial.read("num_particles", config.num_particles);
    initial.read("seed", config.seed);
    initial.read("total_mass", config.total_mass);
    initial.read("plummer_radius", config.plummer_radius);
    initial.read("two_body_separation", config.two_body_separation);

    ObjectReader gravity(section("nbody"), "nbody.");
    if (gravity.find("variant") != nullptr) {
        gravity.read("variant", text_value);
        config.variant = variant_from_string(text_value);
    }
    gravity.read("softening", config.softening);
    gravity.read("gravitational_constant", config.gravitational_constant);
    gravity.read("tile_size", config.nbody.tile_size);

    ObjectReader fluid(section("sph"), "sph.");
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
    fluid.read("external_gravity", config.external_gravity);

    ObjectReader output(section("output"), "output.");
    output.read("snapshot_interval", config.snapshot_interval);
    output.read("snapshot_dir", config.snapshot_dir);
    output.read("diagnostics_interval", config.diagnostics_interval);
    output.read("diagnostics_path", config.diagnostics_path);
    output.read("vtk_interval", config.vtk_interval);
    output.read("vtk_dir", config.vtk_dir);

    std::vector<std::string> paths;
    for (const ObjectReader* reader : {&run, &initial, &gravity, &fluid, &output}) {
        const std::vector<std::string> known = reader->known_paths();
        paths.insert(paths.end(), known.begin(), known.end());
    }
    for (const ObjectReader* reader : {&run, &initial, &gravity, &fluid, &output, &top}) {
        reader->reject_unknown_keys(paths);
    }

    config.validate();
    return config;
}

// Whether text survives what the reader does to a number. The JSON library parses
// to a double, the reader refuses a double beyond the float range, and narrowing
// the rest is a second rounding.
bool reads_back_as(const char* first, const char* last, float value) {
    double wide = 0.0;
    const auto [end, error] = std::from_chars(first, last, wide);
    return error == std::errc() && end == last &&
           std::fabs(wide) <= static_cast<double>(std::numeric_limits<float>::max()) &&
           std::bit_cast<std::uint32_t>(static_cast<float>(wide)) ==
               std::bit_cast<std::uint32_t>(value);
}

// The shortest decimal that reads back to the same float bit for bit, in plain
// notation at ordinary magnitudes. A point or an exponent is always present: a
// bare -0 would parse as the integer 0 and lose its sign. JSON has no NaN or Inf,
// so those print as null, which the reader refuses.
std::string float_text(float value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    char buffer[64];
    char* const last = buffer + sizeof(buffer);
    const float magnitude = std::fabs(value);
    auto result = magnitude >= 1.0e-4f && magnitude < 1.0e7f
                      ? std::to_chars(buffer, last, value, std::chars_format::fixed)
                      : std::to_chars(buffer, last, value);
    // An exhaustive scan finds two magnitudes that need this. The text of the
    // largest float is just above it as a double, so the reader would refuse it,
    // and the double of 7.038531e-26 lands on the midpoint between two floats and
    // narrows to the neighbour. The shortest text of the float's own double always
    // reads back.
    if (result.ec != std::errc() || !reads_back_as(buffer, result.ptr, value)) {
        result = std::to_chars(buffer, last, static_cast<double>(value));
    }
    if (result.ec != std::errc()) {
        throw std::logic_error("cannot format a float");
    }
    std::string text(buffer, result.ptr);
    if (text.find_first_of(".e") == std::string::npos) {
        text += ".0";
    }
    return text;
}

std::string vec3_text(const Vec3& v) {
    return "[" + float_text(v.x) + ", " + float_text(v.y) + ", " + float_text(v.z) + "]";
}

std::string string_text(const std::string& value) { return json(value).dump(); }

std::string bool_text(bool value) { return value ? "true" : "false"; }

// One key and its value, already written as JSON.
using TextField = std::pair<const char*, std::string>;

struct TextSection {
    const char* name;
    std::vector<TextField> fields;
};

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
    const json root = parse_root(text);
    return read_schema_2(schema_of(root) == 1 ? migrate_schema_1(root) : root);
}

Config Config::from_json_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open configuration file " + path.string());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    try {
        const json root = parse_root(buffer.str());
        const bool migrated = schema_of(root) == 1;
        const Config config = read_schema_2(migrated ? migrate_schema_1(root) : root);
        if (migrated) {
            std::fprintf(stderr,
                         "warning: %s uses configuration schema 1 and was migrated; "
                         "--print-config prints its schema 2 form\n",
                         path.string().c_str());
        }
        return config;
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
    if (vtk_interval < 0) {
        throw std::runtime_error("vtk_interval must not be negative, got " +
                                 std::to_string(vtk_interval));
    }
    if (vtk_interval > 0 && vtk_dir.empty()) {
        throw std::runtime_error("vtk_dir must not be empty when vtk_interval is set");
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
    const std::vector<TextField> head{
        {"schema", std::to_string(CURRENT_SCHEMA)},
        {"name", string_text(name)},
        {"solver", string_text(to_string(solver))},
    };
    const std::vector<TextSection> sections{
        {"run",
         {
             {"steps", std::to_string(steps)},
             {"dt", float_text(dt)},
             {"block_size", std::to_string(block_size)},
             {"allow_unstable_dt", bool_text(allow_unstable_dt)},
             {"use_graphs", bool_text(use_graphs)},
         }},
        {"initial_condition",
         {
             {"type", string_text(to_string(initial_condition))},
             {"num_particles", std::to_string(num_particles)},
             {"seed", std::to_string(seed)},
             {"total_mass", float_text(total_mass)},
             {"plummer_radius", float_text(plummer_radius)},
             {"two_body_separation", float_text(two_body_separation)},
         }},
        {"nbody",
         {
             {"variant", string_text(to_string(variant))},
             {"softening", float_text(softening)},
             {"gravitational_constant", float_text(gravitational_constant)},
             {"tile_size", std::to_string(nbody.tile_size)},
         }},
        {"sph",
         {
             {"smoothing_radius", float_text(sph.smoothing_radius)},
             {"rest_density", float_text(sph.rest_density)},
             {"gas_constant", float_text(sph.gas_constant)},
             {"viscosity", float_text(sph.viscosity)},
             {"particle_mass", float_text(sph.particle_mass)},
             {"boundary_damping", float_text(sph.boundary_damping)},
             {"jitter", float_text(sph.jitter)},
             {"domain_min", vec3_text(sph.domain_min)},
             {"domain_max", vec3_text(sph.domain_max)},
             {"fluid_min", vec3_text(sph.fluid_min)},
             {"fluid_max", vec3_text(sph.fluid_max)},
             {"external_gravity", vec3_text(external_gravity)},
         }},
        {"output",
         {
             {"snapshot_interval", std::to_string(snapshot_interval)},
             {"snapshot_dir", string_text(snapshot_dir)},
             {"diagnostics_interval", std::to_string(diagnostics_interval)},
             {"diagnostics_path", string_text(diagnostics_path)},
             {"vtk_interval", std::to_string(vtk_interval)},
             {"vtk_dir", string_text(vtk_dir)},
         }},
    };

    // Written by hand rather than through the JSON library, which holds a float
    // as a double and would print 0.0010000000474974513 for a dt of 0.001. A
    // negative indent is the compact form, as it is for the library.
    const bool compact = indent < 0;
    const std::string line_end = compact ? "" : "\n";
    const std::string colon = compact ? ":" : ": ";
    const auto pad = [&](int depth) {
        return std::string(compact ? 0 : static_cast<std::size_t>(indent * depth), ' ');
    };
    const auto key_text = [](const char* key) { return "\"" + std::string(key) + "\""; };

    std::string out = "{" + line_end;
    for (const TextField& field : head) {
        out += pad(1) + key_text(field.first) + colon + field.second + "," + line_end;
    }
    for (std::size_t s = 0; s < sections.size(); ++s) {
        out += pad(1) + key_text(sections[s].name) + colon + "{" + line_end;
        const std::vector<TextField>& fields = sections[s].fields;
        for (std::size_t f = 0; f < fields.size(); ++f) {
            out += pad(2) + key_text(fields[f].first) + colon + fields[f].second +
                   (f + 1 < fields.size() ? "," : "") + line_end;
        }
        out += pad(1) + "}" + (s + 1 < sections.size() ? "," : "") + line_end;
    }
    out += "}";
    return out;
}

}  // namespace cps
