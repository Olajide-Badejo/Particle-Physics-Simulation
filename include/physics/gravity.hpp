#pragma once

#include <cmath>

#include "physics/hd_macros.hpp"

namespace cps {

// Plummer softened gravity.
//
// The pairwise acceleration on body i due to body j is
//
//     a_i = G m_j (r_j - r_i) / (|r_ij|^2 + eps^2)^{3/2}
//
// The softening length eps removes the singularity as the separation goes to
// zero. Without it a close pass produces an unbounded acceleration, one step
// of the integrator throws the pair to infinity, and the run is destroyed.
// Softening is standard practice for collisionless systems, where the smooth
// mean field rather than individual close encounters carries the dynamics. The
// shipped runs are not fully in that regime: eps = 0.01 in a 10,000 particle
// Plummer sphere with a = 1 is about 7.5 times below the mean separation
// (4 pi / 3 / N)^(1/3), so close encounters still matter and the run is mildly
// collisional. Here softening keeps the integrator stable; it does not make the
// system collisionless.
//
// Reciprocal square root policy. Benchmark builds use rsqrtf, which the fast
// math switch lowers to the multifunction unit approximation. Validation
// builds use an IEEE division by sqrtf instead, so that no compiler flag can
// relax a physics gate. The two paths are selected by the CPS_FAST_MATH
// definition that the build system attaches to each library target.

#if defined(CPS_FAST_MATH) && CPS_FAST_MATH
#define CPS_INV_SQRT(x) rsqrtf(x)
#else
#define CPS_INV_SQRT(x) (1.0f / sqrtf(x))
#endif

// Accumulate the acceleration on a body at (xi, yi, zi) caused by a body of
// mass mj at (xj, yj, zj). Self interaction is harmless: the separation is
// zero, so the numerator vanishes while the softened denominator stays finite.
// That is what lets the tiled kernel loop over a whole tile without a branch.
CPS_HD CPS_INLINE void accumulate_pair_acceleration(float xi, float yi, float zi, float xj,
                                                    float yj, float zj, float mj,
                                                    float softening_sq, float& ax, float& ay,
                                                    float& az) {
    const float dx = xj - xi;
    const float dy = yj - yi;
    const float dz = zj - zi;
    const float dist_sq = dx * dx + dy * dy + dz * dz + softening_sq;
#if defined(__CUDA_ARCH__)
    const float inv_dist = CPS_INV_SQRT(dist_sq);
#else
    const float inv_dist = 1.0f / std::sqrt(dist_sq);
#endif
    const float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
    const float scale = mj * inv_dist_cubed;
    ax += dx * scale;
    ay += dy * scale;
    az += dz * scale;
}

// Softened pair potential energy, consistent with the force law above:
//
//     U_ij = -G m_i m_j / sqrt(|r_ij|^2 + eps^2)
//
// Using the unsoftened potential together with the softened force would break
// the energy conservation test for a reason that has nothing to do with the
// integrator, so both are derived from the same expression.
CPS_HD CPS_INLINE double pair_potential(float xi, float yi, float zi, float xj, float yj, float zj,
                                        float mi, float mj, float softening_sq) {
    const double dx = static_cast<double>(xj) - static_cast<double>(xi);
    const double dy = static_cast<double>(yj) - static_cast<double>(yi);
    const double dz = static_cast<double>(zj) - static_cast<double>(zi);
    const double dist_sq = dx * dx + dy * dy + dz * dz + static_cast<double>(softening_sq);
    return -static_cast<double>(mi) * static_cast<double>(mj) / std::sqrt(dist_sq);
}

}  // namespace cps
