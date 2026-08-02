#pragma once

// Physics formulas live in headers that compile both as device code and as
// ordinary host C++. That gives two things: the kernels and the host side
// reference implementations can never drift apart, and the formula level unit
// tests run in continuous integration on a machine with no GPU.

#if defined(__CUDACC__)
#define CPS_HD __host__ __device__
#define CPS_INLINE __forceinline__
#else
#define CPS_HD
#define CPS_INLINE inline
#endif

namespace cps {

// Pi to float precision. Written out rather than pulled from <numbers> because
// this header is included from device translation units.
inline constexpr float PI_F = 3.14159265358979323846f;
inline constexpr double PI_D = 3.14159265358979323846;

}  // namespace cps
