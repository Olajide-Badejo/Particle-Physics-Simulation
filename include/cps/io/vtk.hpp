#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "cps/core/snapshot.hpp"

namespace cps {

// VTK output for looking at a run in ParaView. Host code with no GPU in it.
//
// A frame is XML PolyData (.vtp): little endian, 64 bit block headers, every
// array in one appended raw binary block.
//
//   Points      Float32, 3 components, interleaved from x, y and z
//   PointData   velocity (Float32, 3 components), mass (Float32), id (UInt32)
//   Verts       Int64 connectivity and offsets, one vertex per point, so the
//               default view shows the points with no filter
//   FieldData   TimeValue (Float64), Step (Int64), VelocityConvention (text)
//
// Points appear in the state's slot order. The fluid solver's sort changes that
// order every step, so point number k is not the same particle from one frame to
// the next. id is what follows a particle across frames.
//
// The velocity array is what the state holds. The fused and fluid solvers carry a
// velocity half a step behind the positions, and a frame writes that as it is,
// with VelocityConvention saying so, rather than correcting it.
//
// Density and pressure are not written: they do not exist until the first step
// has run, so a frame at step 0 would have nothing meaningful to show.

inline constexpr const char* VTK_SERIES_FILE_NAME = "series.pvd";
inline constexpr const char* VTK_VELOCITY_SYNCHRONISED = "synchronised";
inline constexpr const char* VTK_VELOCITY_HALF_STEP = "half_step";

struct VtkFrameInfo {
    uint64_t step = 0;
    double time = 0.0;
    bool half_step_velocity = false;
};

// frame_ followed by the step, zero padded to at least nine digits, like a
// snapshot's name.
std::filesystem::path vtk_frame_path(const std::filesystem::path& directory, uint64_t step);

// Writes one frame to a temporary file in the same directory and renames it into
// place, so a reader never sees a partial frame. Creates the directory if needed.
// Throws std::runtime_error, naming the path, on inconsistent array lengths or a
// failed write.
void write_vtk_frame(const std::filesystem::path& path, const HostParticleState& state,
                     const VtkFrameInfo& info);

struct VtkSeriesEntry {
    uint64_t step = 0;
    double time = 0.0;
    std::string file;  // relative to the directory holding the .pvd
};

// Writes the .pvd index the same way, temporary file then rename, so a run killed
// at any moment leaves a valid series.
void write_vtk_series_index(const std::filesystem::path& path,
                            const std::vector<VtkSeriesEntry>& entries);

// Reads an index this code wrote. Throws std::runtime_error, naming the file and
// the line, on anything else.
std::vector<VtkSeriesEntry> read_vtk_series_index(const std::filesystem::path& path);

// The frames of one run and their index, in one directory. Only series.pvd and
// frame_<step>.vtp, with their temporaries, are ever touched there.
class VtkSeries {
   public:
    // A fresh run replaces whatever series the directory holds. The empty index
    // goes in first, so a run killed part way through the clean up still leaves a
    // valid series rather than one naming frames that are gone.
    static VtkSeries fresh(const std::filesystem::path& directory);

    // A resumed run: keeps the frames up to and including resume_step, deletes any
    // past it, and carries on, which is what a resume does to the diagnostics
    // file. A frame the index names but the disk has lost is dropped with a warning
    // on stderr. With no index the series starts empty. Throws std::runtime_error on
    // an index that cannot be read, before anything has been changed.
    static VtkSeries resumed(const std::filesystem::path& directory, uint64_t resume_step);

    // Writes the frame, then the index that names it, so the index only ever names
    // complete frames. Writing a step the series already has replaces that frame.
    void add_frame(const HostParticleState& state, const VtkFrameInfo& info);

    bool has_frame(uint64_t step) const;
    const std::vector<VtkSeriesEntry>& entries() const { return entries_; }
    std::filesystem::path index_path() const { return directory_ / VTK_SERIES_FILE_NAME; }

   private:
    explicit VtkSeries(std::filesystem::path directory) : directory_(std::move(directory)) {}

    std::filesystem::path directory_;
    std::vector<VtkSeriesEntry> entries_;  // ascending by step
};

}  // namespace cps
