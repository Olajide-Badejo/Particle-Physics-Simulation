#pragma once

#include <cstddef>

#include "core/snapshot.hpp"
#include "utils/cuda_helpers.hpp"

namespace cps {

// Device resident particle state in structure of arrays form.
//
// The layout is the first and largest performance decision in the project.
// With an array of structures, the seven floats belonging to one particle sit
// together, so a warp reading the x component of thirty two consecutive
// particles touches thirty two separate cache lines and wastes most of every
// transaction it issues. With separate arrays, the same read is thirty two
// consecutive floats, which the memory system serves as a small number of
// fully used 128 byte transactions. The naive against tiled comparison in the
// benchmark ladder is only meaningful because both variants already start from
// a coalesced layout.
//
// Positions are double buffered. The fused kernel computes forces from the
// position array and writes updated positions in the same launch, and there is
// no grid wide barrier between the two. Writing in place would let a block
// that has already advanced its particles corrupt the positions another block
// is still reading. Writing to a second buffer and swapping afterwards costs
// 12N bytes of memory and no bandwidth at all, because the write has to happen
// either way.
class ParticleSystem {
   public:
    explicit ParticleSystem(int count);
    ~ParticleSystem() = default;

    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;
    ParticleSystem(ParticleSystem&&) noexcept = default;
    ParticleSystem& operator=(ParticleSystem&&) noexcept = default;

    int size() const { return count_; }

    float* x() { return x_.get(); }
    float* y() { return y_.get(); }
    float* z() { return z_.get(); }
    const float* x() const { return x_.get(); }
    const float* y() const { return y_.get(); }
    const float* z() const { return z_.get(); }

    float* x_next() { return x_next_.get(); }
    float* y_next() { return y_next_.get(); }
    float* z_next() { return z_next_.get(); }

    float* vx() { return vx_.get(); }
    float* vy() { return vy_.get(); }
    float* vz() { return vz_.get(); }
    const float* vx() const { return vx_.get(); }
    const float* vy() const { return vy_.get(); }
    const float* vz() const { return vz_.get(); }

    float* ax() { return ax_.get(); }
    float* ay() { return ay_.get(); }
    float* az() { return az_.get(); }
    const float* ax() const { return ax_.get(); }
    const float* ay() const { return ay_.get(); }
    const float* az() const { return az_.get(); }

    float* mass() { return mass_.get(); }
    const float* mass() const { return mass_.get(); }

    // Exchanges the two position buffers after a fused launch.
    void swap_position_buffers();

    void upload(const HostParticleState& state);
    void download(HostParticleState& state) const;

    // Total device bytes held by the particle state, reported at start up so
    // the memory budget in the specification can be checked against reality.
    std::size_t device_bytes() const;

   private:
    int count_ = 0;
    DeviceBuffer<float> x_, y_, z_;
    DeviceBuffer<float> x_next_, y_next_, z_next_;
    DeviceBuffer<float> vx_, vy_, vz_;
    DeviceBuffer<float> ax_, ay_, az_;
    DeviceBuffer<float> mass_;
};

}  // namespace cps
