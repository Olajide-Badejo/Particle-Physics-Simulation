#include "cps/io/vtk.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <regex>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace cps {

// The appended block is the memory image of these arrays, and the file says
// LittleEndian and Float32.
static_assert(std::endian::native == std::endian::little,
              "VTK frames are declared little endian and written as raw memory");
static_assert(std::numeric_limits<float>::is_iec559, "VTK frames store IEEE 754 floats");

namespace {

// %.17g gives back the same double, so a time read from an index and written again
// does not change.
std::string double_text(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return buffer;
}

// Runs body on a sibling temporary file, then renames it over path. A rename within
// one filesystem is atomic, and replaces an existing file, so a reader sees the old
// file or the new one. Removing the target first, the way a rename failure is
// handled for snapshots, would open a gap in which a series has no index, so a
// failure here is simply a failure.
template <typename Body>
void write_atomically(const fs::path& path, const char* what, Body body) {
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }
    const fs::path temporary = path.string() + ".tmp";
    std::FILE* file = std::fopen(temporary.string().c_str(), "wb");
    if (file == nullptr) {
        throw std::runtime_error(std::string("cannot open ") + what + " " + temporary.string() +
                                 " for writing");
    }
    const auto write = [&](const void* data, std::size_t bytes) {
        if (bytes != 0 && std::fwrite(data, 1, bytes, file) != bytes) {
            throw std::runtime_error(std::string("short write while saving ") + what + " " +
                                     path.string());
        }
    };
    std::error_code ignored;
    try {
        body(write);
        if (std::fflush(file) != 0) {
            throw std::runtime_error("cannot flush " + temporary.string());
        }
    } catch (...) {
        std::fclose(file);
        fs::remove(temporary, ignored);
        throw;
    }
    if (std::fclose(file) != 0) {
        fs::remove(temporary, ignored);
        throw std::runtime_error("cannot close " + temporary.string() + " after writing it");
    }
    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        fs::remove(temporary, ignored);
        throw std::runtime_error(std::string("cannot move ") + what + " into place at " +
                                 path.string() + ": " + error.message());
    }
}

// One array as it lands in the appended block.
struct AppendedArray {
    const void* data = nullptr;
    uint64_t bytes = 0;
};

// A VTK string array in ASCII is its bytes as decimal numbers, NUL terminated.
std::string ascii_string_values(const std::string& text) {
    std::string values;
    for (const char c : text) {
        values += std::to_string(static_cast<unsigned int>(static_cast<unsigned char>(c)));
        values += ' ';
    }
    values += '0';
    return values;
}

// The step in a frame's file name, if it is one and fits.
std::optional<uint64_t> step_from_frame_name(const std::string& name) {
    static const std::regex pattern(R"(frame_(\d+)\.vtp)");
    std::smatch match;
    if (!std::regex_match(name, match, pattern)) {
        return std::nullopt;
    }
    const std::string digits = match[1].str();
    uint64_t step = 0;
    const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), step);
    if (error != std::errc() || end != digits.data() + digits.size()) {
        return std::nullopt;
    }
    return step;
}

// What a killed run can leave behind. The rename never happened, so nothing refers
// to these.
void remove_temporaries(const fs::path& directory) {
    if (!fs::exists(directory)) {
        return;
    }
    static const std::regex pattern(R"((series\.pvd|frame_\d+\.vtp)\.tmp)");
    std::vector<fs::path> doomed;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (std::regex_match(entry.path().filename().string(), pattern)) {
            doomed.push_back(entry.path());
        }
    }
    for (const fs::path& path : doomed) {
        fs::remove(path);
    }
}

// Every frame file in the directory that keep turns down, by name rather than by
// the index, which may be behind the disk if a run died between the two renames.
template <typename Keep>
void remove_frames_unless(const fs::path& directory, Keep keep) {
    if (!fs::exists(directory)) {
        return;
    }
    std::vector<fs::path> doomed;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        const std::optional<uint64_t> step = step_from_frame_name(entry.path().filename().string());
        if (step && !keep(*step)) {
            doomed.push_back(entry.path());
        }
    }
    for (const fs::path& path : doomed) {
        fs::remove(path);
    }
}

}  // namespace

fs::path vtk_frame_path(const fs::path& directory, uint64_t step) {
    char name[64];
    std::snprintf(name, sizeof(name), "frame_%09" PRIu64 ".vtp", step);
    return directory / name;
}

void write_vtk_frame(const fs::path& path, const HostParticleState& state,
                     const VtkFrameInfo& info) {
    const std::size_t count = state.size();
    for (const std::vector<float>* array :
         {&state.y, &state.z, &state.vx, &state.vy, &state.vz, &state.mass}) {
        if (array->size() != count) {
            throw std::runtime_error("cannot write VTK frame " + path.string() +
                                     ": its float arrays have inconsistent lengths");
        }
    }
    if (state.id.size() != count) {
        throw std::runtime_error("cannot write VTK frame " + path.string() + ": " +
                                 std::to_string(count) + " particles but " +
                                 std::to_string(state.id.size()) + " ids");
    }
    if (info.step > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("cannot write VTK frame " + path.string() + ": step " +
                                 std::to_string(info.step) + " does not fit an Int64");
    }

    std::vector<float> points(3 * count);
    std::vector<float> velocity(3 * count);
    std::vector<int64_t> connectivity(count);
    std::vector<int64_t> offsets(count);
    for (std::size_t i = 0; i < count; ++i) {
        points[3 * i] = state.x[i];
        points[3 * i + 1] = state.y[i];
        points[3 * i + 2] = state.z[i];
        velocity[3 * i] = state.vx[i];
        velocity[3 * i + 1] = state.vy[i];
        velocity[3 * i + 2] = state.vz[i];
    }
    std::iota(connectivity.begin(), connectivity.end(), int64_t{0});
    std::iota(offsets.begin(), offsets.end(), int64_t{1});

    // In the order the XML below names them. An offset counts from just after the
    // underscore and includes the eight byte size in front of each array.
    const AppendedArray arrays[] = {
        {points.data(), points.size() * sizeof(float)},
        {velocity.data(), velocity.size() * sizeof(float)},
        {state.mass.data(), count * sizeof(float)},
        {state.id.data(), count * sizeof(uint32_t)},
        {connectivity.data(), count * sizeof(int64_t)},
        {offsets.data(), count * sizeof(int64_t)},
    };
    uint64_t next_offset = 0;
    const auto offset_text = [&](const AppendedArray& array) {
        const uint64_t offset = next_offset;
        next_offset += sizeof(uint64_t) + array.bytes;
        return std::to_string(offset);
    };

    const char* convention =
        info.half_step_velocity ? VTK_VELOCITY_HALF_STEP : VTK_VELOCITY_SYNCHRONISED;
    const std::string n = std::to_string(count);
    std::string xml;
    xml += "<?xml version=\"1.0\"?>\n";
    xml += "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" ";
    xml += "header_type=\"UInt64\">\n";
    xml += "  <PolyData>\n";
    xml += "    <FieldData>\n";
    xml += "      <DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" ";
    xml += "format=\"ascii\">" + double_text(info.time) + "</DataArray>\n";
    xml += "      <DataArray type=\"Int64\" Name=\"Step\" NumberOfTuples=\"1\" ";
    xml += "format=\"ascii\">" + std::to_string(info.step) + "</DataArray>\n";
    xml += "      <Array type=\"String\" Name=\"VelocityConvention\" NumberOfTuples=\"1\" ";
    xml += "format=\"ascii\">" + ascii_string_values(convention) + "</Array>\n";
    xml += "    </FieldData>\n";
    xml += "    <Piece NumberOfPoints=\"" + n + "\" NumberOfVerts=\"" + n + "\" ";
    xml += "NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
    xml += "      <Points>\n";
    xml += "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" ";
    xml += "offset=\"" + offset_text(arrays[0]) + "\"/>\n";
    xml += "      </Points>\n";
    xml += "      <PointData>\n";
    xml += "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" ";
    xml += "format=\"appended\" offset=\"" + offset_text(arrays[1]) + "\"/>\n";
    xml += "        <DataArray type=\"Float32\" Name=\"mass\" format=\"appended\" ";
    xml += "offset=\"" + offset_text(arrays[2]) + "\"/>\n";
    xml += "        <DataArray type=\"UInt32\" Name=\"id\" format=\"appended\" ";
    xml += "offset=\"" + offset_text(arrays[3]) + "\"/>\n";
    xml += "      </PointData>\n";
    xml += "      <Verts>\n";
    xml += "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" ";
    xml += "offset=\"" + offset_text(arrays[4]) + "\"/>\n";
    xml += "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" ";
    xml += "offset=\"" + offset_text(arrays[5]) + "\"/>\n";
    xml += "      </Verts>\n";
    xml += "    </Piece>\n";
    xml += "  </PolyData>\n";
    xml += "  <AppendedData encoding=\"raw\">\n    _";

    write_atomically(path, "VTK frame", [&](const auto& write) {
        write(xml.data(), xml.size());
        for (const AppendedArray& array : arrays) {
            write(&array.bytes, sizeof(array.bytes));
            write(array.data, static_cast<std::size_t>(array.bytes));
        }
        static constexpr char TAIL[] = "\n  </AppendedData>\n</VTKFile>\n";
        write(TAIL, sizeof(TAIL) - 1);
    });
}

void write_vtk_series_index(const fs::path& path, const std::vector<VtkSeriesEntry>& entries) {
    std::string xml;
    xml += "<?xml version=\"1.0\"?>\n";
    xml += "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\" ";
    xml += "header_type=\"UInt64\">\n";
    xml += "  <Collection>\n";
    for (const VtkSeriesEntry& entry : entries) {
        xml += "    <DataSet timestep=\"" + double_text(entry.time) + "\" part=\"0\" file=\"" +
               entry.file + "\"/>\n";
    }
    xml += "  </Collection>\n";
    xml += "</VTKFile>\n";
    write_atomically(path, "VTK series", [&](const auto& write) { write(xml.data(), xml.size()); });
}

std::vector<VtkSeriesEntry> read_vtk_series_index(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open VTK series " + path.string());
    }
    // The index is rewritten whole and only ever by this code, so each line is one
    // of a known few. A hand edited or foreign file is refused, not half understood.
    static const std::regex dataset(
        R"re(\s*<DataSet timestep="([^"]+)" part="0" file="([^"]+)"/>\s*)re");
    static const std::regex opening(R"re(\s*<VTKFile type="Collection"[^>]*>\s*)re");
    const auto refuse = [&path](std::size_t number, const std::string& why) {
        return std::runtime_error("VTK series " + path.string() + " line " +
                                  std::to_string(number) + ": " + why +
                                  ". Move it aside or choose another --vtk-dir");
    };

    std::vector<VtkSeriesEntry> entries;
    std::string line;
    std::size_t number = 0;
    bool is_collection = false;
    bool closed = false;
    while (std::getline(in, line)) {
        ++number;
        if (std::regex_match(line, opening)) {
            is_collection = true;
            continue;
        }
        std::smatch match;
        if (std::regex_match(line, match, dataset)) {
            const std::optional<uint64_t> step = step_from_frame_name(match[2].str());
            if (!step) {
                throw refuse(number, "'" + match[2].str() + "' is not a frame name");
            }
            const std::string time_text = match[1].str();
            char* end = nullptr;
            const double time = std::strtod(time_text.c_str(), &end);
            if (end == time_text.c_str() || *end != '\0') {
                throw refuse(number, "'" + time_text + "' is not a time");
            }
            entries.push_back({*step, time, match[2].str()});
            continue;
        }
        if (line.find("<?xml") != std::string::npos ||
            line.find("Collection>") != std::string::npos) {
            continue;
        }
        if (line.find("</VTKFile>") != std::string::npos) {
            closed = true;
            continue;
        }
        throw refuse(number, "'" + line.substr(0, 80) + "' is not part of a series");
    }
    if (!is_collection || !closed) {
        throw refuse(number, "this is not a complete VTK collection");
    }
    return entries;
}

VtkSeries VtkSeries::fresh(const fs::path& directory) {
    VtkSeries series(directory);
    write_vtk_series_index(series.index_path(), series.entries_);
    remove_frames_unless(directory, [](uint64_t) { return false; });
    remove_temporaries(directory);
    return series;
}

VtkSeries VtkSeries::resumed(const fs::path& directory, uint64_t resume_step) {
    VtkSeries series(directory);
    const fs::path index = series.index_path();
    std::vector<VtkSeriesEntry> previous;
    if (fs::exists(index)) {
        previous = read_vtk_series_index(index);
    }
    for (const VtkSeriesEntry& entry : previous) {
        if (entry.step > resume_step) {
            continue;
        }
        if (!fs::exists(directory / entry.file)) {
            std::fprintf(stderr, "warning: VTK series %s names %s, which is missing; dropping it\n",
                         index.string().c_str(), entry.file.c_str());
            continue;
        }
        series.entries_.push_back(entry);
    }
    std::sort(series.entries_.begin(), series.entries_.end(),
              [](const VtkSeriesEntry& a, const VtkSeriesEntry& b) { return a.step < b.step; });

    // The index goes first, for the same reason as in fresh().
    write_vtk_series_index(index, series.entries_);
    remove_frames_unless(directory, [resume_step](uint64_t step) { return step <= resume_step; });
    remove_temporaries(directory);
    return series;
}

void VtkSeries::add_frame(const HostParticleState& state, const VtkFrameInfo& info) {
    const fs::path path = vtk_frame_path(directory_, info.step);
    write_vtk_frame(path, state, info);

    const VtkSeriesEntry added{info.step, info.time, path.filename().string()};
    const auto place = std::lower_bound(
        entries_.begin(), entries_.end(), info.step,
        [](const VtkSeriesEntry& entry, uint64_t step) { return entry.step < step; });
    if (place != entries_.end() && place->step == info.step) {
        *place = added;
    } else {
        entries_.insert(place, added);
    }
    write_vtk_series_index(index_path(), entries_);
}

bool VtkSeries::has_frame(uint64_t step) const {
    return std::any_of(entries_.begin(), entries_.end(),
                       [step](const VtkSeriesEntry& entry) { return entry.step == step; });
}

}  // namespace cps
