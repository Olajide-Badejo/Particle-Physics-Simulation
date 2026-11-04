#include "cps/core/particle_system.hpp"

#include <cstring>
#include <stdexcept>
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

    ax_.zero();
    ay_.zero();
    az_.zero();
}

void ParticleSystem::swap_position_buffers() {
    std::swap(x_, x_next_);
    std::swap(y_, y_next_);
    std::swap(z_, z_next_);
}

float* ParticleSystem::staging(std::size_t array) const {
    // x, y, z, vx, vy, vz and mass, one after the other.
    constexpr std::size_t STAGED_ARRAYS = 7;
    const std::size_t n = static_cast<std::size_t>(count_);
    if (staging_.get() == nullptr) {
        staging_.allocate(STAGED_ARRAYS * n);
    }
    return staging_.get() + array * n;
}

void ParticleSystem::upload(const HostParticleState& state, cudaStream_t stream) {
    if (state.size() != static_cast<std::size_t>(count_)) {
        throw std::runtime_error("host state size does not match the device particle count");
    }
    const std::size_t n = static_cast<std::size_t>(count_);
    const std::size_t bytes = n * sizeof(float);
    std::memcpy(staging(0), state.x.data(), bytes);
    std::memcpy(staging(1), state.y.data(), bytes);
    std::memcpy(staging(2), state.z.data(), bytes);
    std::memcpy(staging(3), state.vx.data(), bytes);
    std::memcpy(staging(4), state.vy.data(), bytes);
    std::memcpy(staging(5), state.vz.data(), bytes);
    std::memcpy(staging(6), state.mass.data(), bytes);
    copy_to_device_async(x_.get(), staging(0), n, stream);
    copy_to_device_async(y_.get(), staging(1), n, stream);
    copy_to_device_async(z_.get(), staging(2), n, stream);
    copy_to_device_async(vx_.get(), staging(3), n, stream);
    copy_to_device_async(vy_.get(), staging(4), n, stream);
    copy_to_device_async(vz_.get(), staging(5), n, stream);
    copy_to_device_async(mass_.get(), staging(6), n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void ParticleSystem::download(HostParticleState& state, cudaStream_t stream) const {
    const std::size_t n = static_cast<std::size_t>(count_);
    const std::size_t bytes = n * sizeof(float);
    state.resize(n);
    copy_to_host_async(staging(0), x_.get(), n, stream);
    copy_to_host_async(staging(1), y_.get(), n, stream);
    copy_to_host_async(staging(2), z_.get(), n, stream);
    copy_to_host_async(staging(3), vx_.get(), n, stream);
    copy_to_host_async(staging(4), vy_.get(), n, stream);
    copy_to_host_async(staging(5), vz_.get(), n, stream);
    copy_to_host_async(staging(6), mass_.get(), n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::memcpy(state.x.data(), staging(0), bytes);
    std::memcpy(state.y.data(), staging(1), bytes);
    std::memcpy(state.z.data(), staging(2), bytes);
    std::memcpy(state.vx.data(), staging(3), bytes);
    std::memcpy(state.vy.data(), staging(4), bytes);
    std::memcpy(state.vz.data(), staging(5), bytes);
    std::memcpy(state.mass.data(), staging(6), bytes);
}

std::size_t ParticleSystem::device_bytes() const {
    return x_.bytes() + y_.bytes() + z_.bytes() + x_next_.bytes() + y_next_.bytes() +
           z_next_.bytes() + vx_.bytes() + vy_.bytes() + vz_.bytes() + ax_.bytes() + ay_.bytes() +
           az_.bytes() + mass_.bytes();
}

}  // namespace cps
