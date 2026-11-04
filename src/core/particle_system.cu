#include "cps/core/particle_system.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace cps {

ParticleSystem::ParticleSystem(int count) : count_(count) {
    if (count <= 0) {
        throw std::runtime_error("a particle system needs at least one particle");
    }
    const std::size_t n = static_cast<std::size_t>(count);
    x_.allocate(n);
    y_.allocate(n);
    z_.allocate(n);
    x_next_.allocate(n);
    y_next_.allocate(n);
    z_next_.allocate(n);
    vx_.allocate(n);
    vy_.allocate(n);
    vz_.allocate(n);
    ax_.allocate(n);
    ay_.allocate(n);
    az_.allocate(n);
    mass_.allocate(n);
    id_.allocate(n);

    ax_.zero();
    ay_.zero();
    az_.zero();
}

void ParticleSystem::swap_position_buffers() {
    std::swap(x_, x_next_);
    std::swap(y_, y_next_);
    std::swap(z_, z_next_);
}

template <typename T>
T* ParticleSystem::staging(std::size_t array) const {
    // x, y, z, vx, vy, vz, mass and id, one after the other. The host only ever
    // copies these bytes, so one buffer can hold arrays of two types.
    constexpr std::size_t STAGED_ARRAYS = 8;
    constexpr std::size_t SLOT_ELEMENT_BYTES = 4;
    static_assert(sizeof(T) == SLOT_ELEMENT_BYTES);
    const std::size_t n = static_cast<std::size_t>(count_);
    if (staging_.get() == nullptr) {
        staging_.allocate(STAGED_ARRAYS * n * SLOT_ELEMENT_BYTES);
    }
    return reinterpret_cast<T*>(staging_.get() + array * n * SLOT_ELEMENT_BYTES);
}

void ParticleSystem::upload(const HostParticleState& state, cudaStream_t stream) {
    if (state.size() != static_cast<std::size_t>(count_)) {
        throw std::runtime_error("host state size does not match the device particle count");
    }
    if (state.id.size() != static_cast<std::size_t>(count_)) {
        throw std::runtime_error("host state has " + std::to_string(state.id.size()) +
                                 " ids for " + std::to_string(count_) + " particles");
    }
    const std::size_t n = static_cast<std::size_t>(count_);
    const std::size_t bytes = n * sizeof(float);
    std::memcpy(staging<float>(0), state.x.data(), bytes);
    std::memcpy(staging<float>(1), state.y.data(), bytes);
    std::memcpy(staging<float>(2), state.z.data(), bytes);
    std::memcpy(staging<float>(3), state.vx.data(), bytes);
    std::memcpy(staging<float>(4), state.vy.data(), bytes);
    std::memcpy(staging<float>(5), state.vz.data(), bytes);
    std::memcpy(staging<float>(6), state.mass.data(), bytes);
    std::memcpy(staging<uint32_t>(7), state.id.data(), n * sizeof(uint32_t));
    copy_to_device_async(x_.get(), staging<float>(0), n, stream);
    copy_to_device_async(y_.get(), staging<float>(1), n, stream);
    copy_to_device_async(z_.get(), staging<float>(2), n, stream);
    copy_to_device_async(vx_.get(), staging<float>(3), n, stream);
    copy_to_device_async(vy_.get(), staging<float>(4), n, stream);
    copy_to_device_async(vz_.get(), staging<float>(5), n, stream);
    copy_to_device_async(mass_.get(), staging<float>(6), n, stream);
    copy_to_device_async(id_.get(), staging<uint32_t>(7), n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void ParticleSystem::download(HostParticleState& state, cudaStream_t stream) const {
    const std::size_t n = static_cast<std::size_t>(count_);
    const std::size_t bytes = n * sizeof(float);
    state.resize(n);
    copy_to_host_async(staging<float>(0), x_.get(), n, stream);
    copy_to_host_async(staging<float>(1), y_.get(), n, stream);
    copy_to_host_async(staging<float>(2), z_.get(), n, stream);
    copy_to_host_async(staging<float>(3), vx_.get(), n, stream);
    copy_to_host_async(staging<float>(4), vy_.get(), n, stream);
    copy_to_host_async(staging<float>(5), vz_.get(), n, stream);
    copy_to_host_async(staging<float>(6), mass_.get(), n, stream);
    copy_to_host_async(staging<uint32_t>(7), id_.get(), n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::memcpy(state.x.data(), staging<float>(0), bytes);
    std::memcpy(state.y.data(), staging<float>(1), bytes);
    std::memcpy(state.z.data(), staging<float>(2), bytes);
    std::memcpy(state.vx.data(), staging<float>(3), bytes);
    std::memcpy(state.vy.data(), staging<float>(4), bytes);
    std::memcpy(state.vz.data(), staging<float>(5), bytes);
    std::memcpy(state.mass.data(), staging<float>(6), bytes);
    std::memcpy(state.id.data(), staging<uint32_t>(7), n * sizeof(uint32_t));
}

std::size_t ParticleSystem::device_bytes() const {
    return x_.bytes() + y_.bytes() + z_.bytes() + x_next_.bytes() + y_next_.bytes() +
           z_next_.bytes() + vx_.bytes() + vy_.bytes() + vz_.bytes() + ax_.bytes() + ay_.bytes() +
           az_.bytes() + mass_.bytes() + id_.bytes();
}

}  // namespace cps
