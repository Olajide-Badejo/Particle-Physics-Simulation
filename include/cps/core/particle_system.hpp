#pragma once

#include <cstddef>
#include <cstdint>

#include "cps/core/snapshot.hpp"
#include "cps/utils/cuda_helpers.hpp"

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

    // A label for each particle, 0 to N - 1 in the initial state's order. The
    // fluid solver permutes it with the rest of the particle, so it says which
    // particle is in which slot. No force, sort key or integrator reads it.
    uint32_t* id() { return id_.get(); }
    const uint32_t* id() const { return id_.get(); }

    // Exchanges the two position buffers after a fused launch.
    void swap_position_buffers();

    // Host transfers go through one pinned staging buffer and are ordered on the
    // stream, so they see the work already enqueued on it, which a transfer on
    // any other stream would not. Both return with the stream drained: upload
    // because the staging buffer is reused, download because the host cannot
    // read the result sooner. Neither is for the solver path.
    void upload(const HostParticleState& state, cudaStream_t stream);
    void download(HostParticleState& state, cudaStream_t stream) const;

    // Total device bytes held by the particle state, reported at start up so a
    // run's memory use is on the record.
    std::size_t device_bytes() const;

   private:
    // Where one array sits in the staging buffer, which is allocated at the
    // first transfer and kept, so a run writing snapshots does not pin and unpin
    // memory for each one. Every slot holds count four byte elements.
    template <typename T>
    T* staging(std::size_t array) const;

    int count_ = 0;
    DeviceBuffer<float> x_, y_, z_;
    DeviceBuffer<float> x_next_, y_next_, z_next_;
    DeviceBuffer<float> vx_, vy_, vz_;
    DeviceBuffer<float> ax_, ay_, az_;
    DeviceBuffer<float> mass_;
    DeviceBuffer<uint32_t> id_;
    mutable HostBuffer<unsigned char> staging_;
};

}  // namespace cps
