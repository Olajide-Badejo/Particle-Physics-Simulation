#include "cps/core/particle_system.hpp"

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

void ParticleSystem::upload(const HostParticleState& state) {
    if (state.size() != static_cast<std::size_t>(count_)) {
        throw std::runtime_error("host state size does not match the device particle count");
    }
    const std::size_t bytes = static_cast<std::size_t>(count_) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(x_.get(), state.x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_.get(), state.y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(z_.get(), state.z.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vx_.get(), state.vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vy_.get(), state.vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vz_.get(), state.vz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mass_.get(), state.mass.data(), bytes, cudaMemcpyHostToDevice));
}

void ParticleSystem::download(HostParticleState& state) const {
    state.resize(static_cast<std::size_t>(count_));
    const std::size_t bytes = static_cast<std::size_t>(count_) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(state.x.data(), x_.get(), bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.y.data(), y_.get(), bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.z.data(), z_.get(), bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.vx.data(), vx_.get(), bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.vy.data(), vy_.get(), bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.vz.data(), vz_.get(), bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.mass.data(), mass_.get(), bytes, cudaMemcpyDeviceToHost));
}

std::size_t ParticleSystem::device_bytes() const {
    return x_.bytes() + y_.bytes() + z_.bytes() + x_next_.bytes() + y_next_.bytes() +
           z_next_.bytes() + vx_.bytes() + vy_.bytes() + vz_.bytes() + ax_.bytes() + ay_.bytes() +
           az_.bytes() + mass_.bytes();
}

}  // namespace cps
