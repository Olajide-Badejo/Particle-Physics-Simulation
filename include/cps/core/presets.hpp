#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cps {

// A file from presets/, compiled in at configure time, so that an installed
// binary needs nothing beside it.
struct EmbeddedPreset {
    std::string_view name;  // the file name without .json
    std::string_view json;
};

// Every preset, sorted by name.
const std::vector<EmbeddedPreset>& embedded_presets();

// Null when there is no preset of that name.
const EmbeddedPreset* find_preset(std::string_view name);

// One line per preset: its name, solver, particle count and steps.
std::string preset_listing();

}  // namespace cps
