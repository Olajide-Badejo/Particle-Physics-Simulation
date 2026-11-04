#include "cps/core/physics_fingerprint.hpp"

#include <nlohmann/json.hpp>

#include <bit>
#include <charconv>
#include <cstring>
#include <stdexcept>
#include <system_error>

#include "cps/core/initial_conditions.hpp"
#include "cps/core/snapshot.hpp"

namespace cps {
namespace {

using ordered_json = nlohmann::ordered_json;

constexpr const char* ABSENT = "(absent)";

// Shortest text that reads back to the same float, so two values print the
// same exactly when they have the same bits.
std::string float_text(float value) {
    char buffer[64];
    const auto [end, error] = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (error != std::errc()) {
        throw std::logic_error("cannot format a float");
    }
    return std::string(buffer, end);
}

// Codes are spelled out rather than taken from the enum's underlying value,
// so reordering the enum in config.hpp cannot change a fingerprint.
uint32_t solver_code(SolverType value) {
    switch (value) {
        case SolverType::NBody:
            return 0;
        case SolverType::Sph:
            return 1;
    }
    throw std::logic_error("unknown solver");
}

uint32_t variant_code(NBodyVariant value) {
    switch (value) {
        case NBodyVariant::Naive:
            return 0;
        case NBodyVariant::Tiled:
            return 1;
        case NBodyVariant::Fused:
            return 2;
    }
    throw std::logic_error("unknown variant");
}

void add_enum(std::vector<PhysicsField>& fields, const char* name, uint32_t code,
              std::string text) {
    fields.push_back({name, PhysicsFieldKind::Enum, code, std::move(text)});
}

void add_float(std::vector<PhysicsField>& fields, const std::string& name, float value) {
    fields.push_back(
        {name, PhysicsFieldKind::Float32, std::bit_cast<uint32_t>(value), float_text(value)});
}

void add_vec3(std::vector<PhysicsField>& fields, const std::string& name, const Vec3& value) {
    add_float(fields, name + ".x", value.x);
    add_float(fields, name + ".y", value.y);
    add_float(fields, name + ".z", value.z);
}

void add_integer(std::vector<PhysicsField>& fields, const char* name, long long value) {
    fields.push_back({name, PhysicsFieldKind::Integer, static_cast<uint64_t>(value),
                      std::to_string(value)});
}

void hash_u32(uint64_t& hash, uint32_t value) { hash = fnv1a_64(&value, sizeof(value), hash); }
void hash_u64(uint64_t& hash, uint64_t value) { hash = fnv1a_64(&value, sizeof(value), hash); }

}  // namespace

// The one list of physics fields, in fingerprint order.
//
// Inclusion rule: a field is listed when changing it changes the evolution
// operator, that is, when the same state stepped under the old and the new
// value would diverge. That covers the solver and variant, the particle count,
// the time step, the force law constants and external gravity and, for the
// fluid solver, its physical parameters and walls. The smoothing radius and
// particle mass are listed as the solver resolves them, so a derived value
// that moves with the particle count or the mass calibration is caught.
//
// Excluded on purpose: run control (steps, snapshot_*, diagnostics_*, name,
// allow_unstable_dt, and block_size, nbody.tile_size and nbody.blocking_factor,
// which only change the launch shape), and fields that only shape the initial
// state (seed, initial_condition, sph.jitter, sph.fluid_min and sph.fluid_max,
// plummer_radius, total_mass, two_body_separation). A resumed run never regenerates
// its initial state, so those cannot affect it.
//
// The general group is listed for every solver, even where one solver ignores
// a field (the fluid solver never reads softening or variant), because a
// spurious refusal costs a --force-resume while a missed one costs a silently
// wrong run. The fluid group is listed only when the solver is sph. The
// N-body solver never reads it, and its resolved smoothing radius depends on
// the fluid block, so listing it would let an initial state field refuse an
// N-body resume.
std::vector<PhysicsField> physics_fields(const Config& config) {
    std::vector<PhysicsField> fields;
    add_enum(fields, "solver", solver_code(config.solver), to_string(config.solver));
    add_enum(fields, "variant", variant_code(config.variant), to_string(config.variant));
    add_integer(fields, "num_particles", config.num_particles);
    add_float(fields, "dt", config.dt);
    add_float(fields, "softening", config.softening);
    add_float(fields, "gravitational_constant", config.gravitational_constant);
    add_vec3(fields, "external_gravity", config.external_gravity);

    if (config.solver == SolverType::Sph) {
        const SphDerivedParameters derived = derive_sph_parameters(config);
        add_float(fields, "sph.smoothing_radius", derived.smoothing_radius);
        add_float(fields, "sph.particle_mass", derived.particle_mass);
        add_float(fields, "sph.rest_density", config.sph.rest_density);
        add_float(fields, "sph.gas_constant", config.sph.gas_constant);
        add_float(fields, "sph.viscosity", config.sph.viscosity);
        add_float(fields, "sph.boundary_damping", config.sph.boundary_damping);
        add_vec3(fields, "sph.domain_min", config.sph.domain_min);
        add_vec3(fields, "sph.domain_max", config.sph.domain_max);
    }
    return fields;
}

uint64_t physics_fingerprint(const std::vector<PhysicsField>& fields) {
    uint64_t hash = FNV1A_64_BASIS;
    hash_u32(hash, PHYSICS_FINGERPRINT_VERSION);
    hash_u32(hash, static_cast<uint32_t>(fields.size()));
    for (const PhysicsField& field : fields) {
        hash_u32(hash, static_cast<uint32_t>(field.name.size()));
        hash = fnv1a_64(field.name.data(), field.name.size(), hash);
        hash_u32(hash, static_cast<uint32_t>(field.kind));
        if (field.kind == PhysicsFieldKind::Integer) {
            hash_u64(hash, field.bits);
        } else {
            hash_u32(hash, static_cast<uint32_t>(field.bits));
        }
    }
    return hash;
}

uint64_t physics_fingerprint(const Config& config) {
    return physics_fingerprint(physics_fields(config));
}

std::string physics_json(const std::vector<PhysicsField>& fields) {
    ordered_json values = ordered_json::object();
    for (const PhysicsField& field : fields) {
        values[field.name] = field.text;
    }
    ordered_json root = ordered_json::object();
    root["fingerprint_version"] = PHYSICS_FINGERPRINT_VERSION;
    root["fields"] = std::move(values);
    return root.dump();
}

std::vector<PhysicsDifference> compare_physics(std::string_view recorded_json,
                                               const std::vector<PhysicsField>& fields) {
    const ordered_json root = ordered_json::parse(recorded_json, nullptr, false);
    if (root.is_discarded() || !root.is_object() || !root.contains("fields") ||
        !root["fields"].is_object()) {
        throw std::runtime_error("its physics record is not readable");
    }
    const ordered_json& recorded = root["fields"];

    std::vector<PhysicsDifference> differences;
    for (const PhysicsField& field : fields) {
        const auto it = recorded.find(field.name);
        if (it == recorded.end()) {
            differences.push_back({field.name, ABSENT, field.text});
        } else if (!it->is_string() || it->get<std::string>() != field.text) {
            differences.push_back(
                {field.name, it->is_string() ? it->get<std::string>() : it->dump(), field.text});
        }
    }
    for (auto it = recorded.begin(); it != recorded.end(); ++it) {
        bool known = false;
        for (const PhysicsField& field : fields) {
            known = known || field.name == it.key();
        }
        if (!known) {
            differences.push_back(
                {it.key(), it->is_string() ? it->get<std::string>() : it->dump(), ABSENT});
        }
    }
    return differences;
}

std::string describe_differences(const std::vector<PhysicsDifference>& differences) {
    std::string text;
    for (const PhysicsDifference& difference : differences) {
        if (!text.empty()) {
            text += "; ";
        }
        text += difference.name + " is " + difference.snapshot_value + " in the snapshot and " +
                difference.config_value + " in the configuration";
    }
    return text;
}

}  // namespace cps
