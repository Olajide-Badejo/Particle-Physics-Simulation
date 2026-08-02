#pragma once

#include <cmath>

#include "physics/hd_macros.hpp"

namespace cps {

// Smoothing kernels for smoothed particle hydrodynamics, in the combination
// introduced by Muller, Charypar and Gross for interactive fluids (SCA 2003).
//
// Three different kernels are used on purpose, each for the one job it does
// well:
//
//   poly6      density and general interpolation. Smooth and cheap because it
//              is a polynomial in r^2, so no square root is needed.
//   spiky      the pressure gradient. Its gradient does not vanish as the
//              separation goes to zero, which is what stops particles from
//              clumping into each other at high compression. The poly6
//              gradient does vanish there and produces exactly that failure.
//   viscosity  the velocity Laplacian. Its second derivative is positive
//              everywhere inside the support, so the viscous term always
//              damps and never injects energy.
//
// Each kernel integrates to one over the sphere of radius h, which is the
// property the normalization test checks by quadrature.

// W_poly6(r, h) = 315 / (64 pi h^9) * (h^2 - r^2)^3 for 0 <= r <= h.
CPS_HD CPS_INLINE float poly6_kernel(float r_sq, float h) {
    const float h_sq = h * h;
    if (r_sq >= h_sq) {
        return 0.0f;
    }
    const float h9 = h_sq * h_sq * h_sq * h_sq * h;
    const float coeff = 315.0f / (64.0f * PI_F * h9);
    const float d = h_sq - r_sq;
    return coeff * d * d * d;
}

// Magnitude of grad W_spiky, which is 45 / (pi h^6) * (h - r)^2.
// The gradient itself points along the unit separation vector; the caller
// multiplies by that direction and by the sign convention of the pressure
// term. Returns zero outside the support and at exactly zero separation,
// where the direction is undefined.
CPS_HD CPS_INLINE float spiky_gradient_magnitude(float r, float h) {
    if (r >= h || r <= 0.0f) {
        return 0.0f;
    }
    const float h_sq = h * h;
    const float h6 = h_sq * h_sq * h_sq;
    const float coeff = 45.0f / (PI_F * h6);
    const float d = h - r;
    return coeff * d * d;
}

// W_spiky(r, h) = 15 / (pi h^6) * (h - r)^3. Provided so the normalization of
// the pressure kernel can be tested in its own right.
CPS_HD CPS_INLINE float spiky_kernel(float r, float h) {
    if (r >= h) {
        return 0.0f;
    }
    const float h_sq = h * h;
    const float h6 = h_sq * h_sq * h_sq;
    const float coeff = 15.0f / (PI_F * h6);
    const float d = h - r;
    return coeff * d * d * d;
}

// Laplacian of the viscosity kernel: 45 / (pi h^6) * (h - r).
CPS_HD CPS_INLINE float viscosity_laplacian(float r, float h) {
    if (r >= h) {
        return 0.0f;
    }
    const float h_sq = h * h;
    const float h6 = h_sq * h_sq * h_sq;
    return 45.0f / (PI_F * h6) * (h - r);
}

// W_visc(r, h) = 15 / (2 pi h^3) * (-r^3 / (2 h^3) + r^2 / h^2 + h / (2 r) - 1).
// Singular at r = 0 in value but integrable; the quadrature test starts from a
// small positive radius.
CPS_HD CPS_INLINE float viscosity_kernel(float r, float h) {
    if (r >= h || r <= 0.0f) {
        return 0.0f;
    }
    const float h3 = h * h * h;
    const float coeff = 15.0f / (2.0f * PI_F * h3);
    return coeff * (-(r * r * r) / (2.0f * h3) + (r * r) / (h * h) + h / (2.0f * r) - 1.0f);
}

// Equation of state. A stiff linear relation between density excess and
// pressure keeps the fluid weakly compressible without the cost of solving a
// pressure Poisson equation every step. Clamping at zero suppresses the
// spurious attraction that would otherwise pull the free surface apart when a
// particle finds itself below rest density simply because it has too few
// neighbours on the outside.
CPS_HD CPS_INLINE float equation_of_state(float density, float rest_density, float gas_constant) {
    const float excess = density - rest_density;
    return excess > 0.0f ? gas_constant * excess : 0.0f;
}

}  // namespace cps
