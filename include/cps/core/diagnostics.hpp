#pragma once

#include <cstdio>
#include <string>

#include "cps/core/particle_system.hpp"
#include "cps/utils/cuda_helpers.hpp"

namespace cps {

struct Diagnostics {
    long long step = 0;
    double time = 0.0;
    double kinetic = 0.0;
    double potential = 0.0;
    double total = 0.0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double momentum_magnitude = 0.0;
};

// Conserved quantity reduction.
//
// The particle state stays in single precision because that is the whole point
// of the exercise, but the reduction that measures conservation does not. A
// float32 tree reduction over ten thousand terms loses three to four decimal
// digits, and the energy gate resolves one part in a thousand over ten
// thousand steps. Reducing a float32 state in float32 would therefore have
// measured the reduction rather than the integrator. The accumulators are
// double; the traffic cost is one extra small array and nothing per particle.
//
// The reduction is also deterministic: block partials are written to a buffer
// and summed in a fixed order by a single block, rather than being accumulated
// with atomics whose ordering varies between launches.
//
// Each reduction launches on the stream it is given, so it sees everything
// already enqueued there, copies its partials back on the same stream, and
// drains the stream before summing them on the host. The stream has no default:
// see nbody_solver.hpp. That is also why half_dt, which comes before it, has none.
class DiagnosticsReducer {
   public:
    explicit DiagnosticsReducer(int count, int block_size = 256);

    // Kinetic energy and total momentum, order N.
    //
    // The fused variant carries the leapfrog half step velocity, which is not
    // the velocity that belongs with the current positions. Passing half_dt
    // equal to dt/2 makes the reduction evaluate v + a half_dt instead, which
    // is the synchronised velocity. Correcting it inside the kernel rather
    // than by shifting the velocity array and shifting it back is deliberate:
    // adding and then subtracting the same quantity in single precision does
    // not return the original bits, and that would break the bit identity
    // guarantee of the resume path for any run that took a diagnostic. The
    // acceleration array has to hold a_{n+1} for that to be the synchronised
    // velocity, so the fused solver refreshes it first, in prepare_diagnostics.
    void reduce_kinetic_and_momentum(const ParticleSystem& particles, double& kinetic, double& px,
                                     double& py, double& pz, float half_dt, cudaStream_t stream);

    // Softened gravitational potential energy, order N squared. Each unordered
    // pair is visited twice by the all pairs sum, so the result is halved.
    double reduce_potential(const ParticleSystem& particles, float softening,
                            float gravitational_constant, cudaStream_t stream);

    Diagnostics compute(const ParticleSystem& particles, long long step, double time,
                        float softening, float gravitational_constant, bool include_potential,
                        float half_dt, cudaStream_t stream);

   private:
    int count_ = 0;
    int block_size_ = 256;
    int partial_blocks_ = 0;
    DeviceBuffer<double> partials_;   // four doubles per block
    DeviceBuffer<double> partials1_;  // one double per block
    HostBuffer<double> partials_host_;
    HostBuffer<double> partials1_host_;
};

// The first line of every diagnostics file, without its newline.
inline constexpr const char* DIAGNOSTICS_HEADER =
    "step,time,kinetic,potential,total,px,py,pz,momentum_magnitude";

// Streams diagnostics rows to a comma separated file. The header names the
// units so a reader never has to guess what the numbers are.
class DiagnosticsWriter {
   public:
    DiagnosticsWriter() = default;

    // A fresh run: truncates the file and writes the header.
    explicit DiagnosticsWriter(const std::string& path);

    // A resumed run: continues the file the original run wrote, so the file
    // ends up holding every step exactly once. A missing or empty file gets
    // the header. Otherwise the first line must be the header, and every row
    // at or past resume_step is dropped before appending, because the resumed
    // run writes those steps again; that also covers a run that died after
    // its diagnostics got ahead of its last snapshot. An unterminated last
    // line, which is what a crash in the middle of a write leaves, is dropped
    // too. The drop rewrites the file through a temporary file and a rename,
    // so an interruption leaves either the old file or the new one. Throws
    // std::runtime_error on a header that does not match or a row whose step
    // cannot be read, rather than mixing two formats in one file.
    DiagnosticsWriter(const std::string& path, long long resume_step);

    ~DiagnosticsWriter();

    DiagnosticsWriter(const DiagnosticsWriter&) = delete;
    DiagnosticsWriter& operator=(const DiagnosticsWriter&) = delete;

    bool is_open() const { return file_ != nullptr; }
    void write(const Diagnostics& row);

    // Pushes buffered rows to the file. The engine calls it before every
    // snapshot, so a run that dies later has diagnostics on disk up to the
    // step a resume will start from.
    void flush();
    void close();

   private:
    std::FILE* file_ = nullptr;
};

}  // namespace cps
