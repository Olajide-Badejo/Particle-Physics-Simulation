#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "cps/core/config.hpp"

namespace cps {

// What a snapshot remembers about the physics that produced it.
//
// A resume is only meaningful under the same evolution operator, so the
// snapshot carries a 64 bit fingerprint of the fields that define it and a
// readable copy of their values. The fingerprint is computed from an explicit
// list of fields rather than from the configuration's JSON, so reformatting or
// restructuring the configuration does not invalidate existing snapshots. The
// list lives in physics_fields() and nowhere else. Changing what it covers or
// how a field is encoded means bumping PHYSICS_FINGERPRINT_VERSION.
inline constexpr uint32_t PHYSICS_FINGERPRINT_VERSION = 1;

// Explicit values: they are hashed, so they are never renumbered.
enum class PhysicsFieldKind : uint32_t {
    Enum = 1,     // hashed as uint32 with the explicit codes in physics_fields()
    Float32 = 2,  // hashed by bit pattern
    Integer = 3,  // hashed as a two's complement int64
};

struct PhysicsField {
    std::string name;
    PhysicsFieldKind kind = PhysicsFieldKind::Float32;
    uint64_t bits = 0;  // the enum code, the float's bit pattern, or the integer
    std::string text;   // the enum name, the shortest round trip float, or the decimal
};

// The ordered field list for this configuration.
std::vector<PhysicsField> physics_fields(const Config& config);

// FNV-1a 64 of PHYSICS_FINGERPRINT_VERSION and each field's name, kind and
// value, all little endian, with lengths written ahead of the strings, so the
// value does not depend on any text formatting.
uint64_t physics_fingerprint(const std::vector<PhysicsField>& fields);
uint64_t physics_fingerprint(const Config& config);

// {"fingerprint_version":1,"fields":{"solver":"nbody",...}} in list order,
// compact. Values are strings so comparing two records is exact.
std::string physics_json(const std::vector<PhysicsField>& fields);

struct PhysicsDifference {
    std::string name;
    std::string snapshot_value;  // "(absent)" when the snapshot lacks the field
    std::string config_value;    // "(absent)" when this build does not know it
};

// Every field whose recorded value differs from the configuration's, in list
// order, followed by any recorded field this build does not know. Throws
// std::runtime_error if the record is not a physics record.
std::vector<PhysicsDifference> compare_physics(std::string_view recorded_json,
                                               const std::vector<PhysicsField>& fields);

// "dt is 0.001 in the snapshot and 0.002 in the configuration; ..."
std::string describe_differences(const std::vector<PhysicsDifference>& differences);

}  // namespace cps
