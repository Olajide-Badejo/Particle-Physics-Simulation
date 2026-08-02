#pragma once

#include <cstdio>
#include <string>

#include "core/particle_system.hpp"
#include "utils/cuda_helpers.hpp"

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
    // guarantee of the resume path for any run that took a diagnostic.
    void reduce_kinetic_and_momentum(const ParticleSystem& particles, double& kinetic, double& px,
                                     double& py, double& pz, float half_dt = 0.0f);

    // Softened gravitational potential energy, order N squared. Each unordered
    // pair is visited twice by the all pairs sum, so the result is halved.
    double reduce_potential(const ParticleSystem& particles, float softening,
                            float gravitational_constant);

    Diagnostics compute(const ParticleSystem& particles, long long step, double time,
                        float softening, float gravitational_constant, bool include_potential,
                        float half_dt = 0.0f);

   private:
    int count_ = 0;
    int block_size_ = 256;
    int partial_blocks_ = 0;
    DeviceBuffer<double> partials_;   // four doubles per block
    DeviceBuffer<double> partials1_;  // one double per block
    DeviceBuffer<double> result_;
};

// Streams diagnostics rows to a comma separated file. The header names the
// units so a reader never has to guess what the numbers are.
class DiagnosticsWriter {
   public:
    DiagnosticsWriter() = default;
    explicit DiagnosticsWriter(const std::string& path);
    ~DiagnosticsWriter();

    DiagnosticsWriter(const DiagnosticsWriter&) = delete;
    DiagnosticsWriter& operator=(const DiagnosticsWriter&) = delete;

    bool is_open() const { return file_ != nullptr; }
    void write(const Diagnostics& row);
    void close();

   private:
    std::FILE* file_ = nullptr;
};

}  // namespace cps
