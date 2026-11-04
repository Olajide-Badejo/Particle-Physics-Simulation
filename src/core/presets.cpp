#include "cps/core/presets.hpp"

#include <cstdio>

#include "cps/core/config.hpp"
#include "cps/utils/progress.hpp"

namespace cps {

const std::vector<EmbeddedPreset>& embedded_presets() {
    static const std::vector<EmbeddedPreset> presets = {
#include "embedded_presets.inc"
    };
    return presets;
}

const EmbeddedPreset* find_preset(std::string_view name) {
    for (const EmbeddedPreset& preset : embedded_presets()) {
        if (preset.name == name) {
            return &preset;
        }
    }
    return nullptr;
}

std::string preset_listing() {
    std::string listing;
    for (const EmbeddedPreset& preset : embedded_presets()) {
        const Config config = Config::from_json_string(preset.json);
        char line[192];
        std::snprintf(line, sizeof(line), "%-20s %-6s %12s particles %12s steps\n",
                      std::string(preset.name).c_str(), to_string(config.solver).c_str(),
                      format_count(config.num_particles).c_str(),
                      format_count(config.steps).c_str());
        listing += line;
    }
    return listing;
}

}  // namespace cps
