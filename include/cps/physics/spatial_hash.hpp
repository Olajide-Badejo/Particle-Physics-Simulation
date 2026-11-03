#pragma once

#include <cstdint>
#include <cmath>

#include "cps/physics/hd_macros.hpp"

namespace cps {

// Spatial hashing for neighbour search, after Teschner et al., "Optimized
// Spatial Hashing for Collision Detection of Deformable Objects", VMV 2003.
//
// A dense uniform grid would need storage proportional to the bounding volume
// rather than to the particle count. A dam break spends most of its life with
// the fluid occupying a small and moving part of the domain, so most of that
// grid would be empty. Hashing the integer cell coordinate into a table sized
// by particle count keeps the storage proportional to the work.
//
// The three multipliers are large primes; the exclusive or of the products
// decorrelates the three axes.

inline constexpr uint32_t HASH_PRIME_X = 73856093u;
inline constexpr uint32_t HASH_PRIME_Y = 19349663u;
inline constexpr uint32_t HASH_PRIME_Z = 83492791u;

// Integer cell coordinate of a point. Uses floor rather than truncation so
// that negative coordinates map correctly; truncation would fold the cells on
// either side of the origin onto each other.
CPS_HD CPS_INLINE int cell_coordinate(float position, float cell_size) {
    return static_cast<int>(std::floor(position / cell_size));
}

CPS_HD CPS_INLINE uint32_t spatial_hash(int ix, int iy, int iz, uint32_t table_size) {
    const uint32_t hx = static_cast<uint32_t>(ix) * HASH_PRIME_X;
    const uint32_t hy = static_cast<uint32_t>(iy) * HASH_PRIME_Y;
    const uint32_t hz = static_cast<uint32_t>(iz) * HASH_PRIME_Z;
    return (hx ^ hy ^ hz) % table_size;
}

// Exact cell identity, packed into 64 bits as three 21 bit fields.
//
// This exists because hashing is lossy. Two different cells can land in the
// same bucket, and when the neighbour search visits the twenty seven cells
// around a particle, two of those cells can hash to the same bucket. Scanning
// a bucket without checking which cell a candidate actually belongs to would
// then count the same neighbour twice and corrupt the density sum. Comparing
// this key rejects both the foreign particles pulled in by a collision and the
// double visit, at the cost of one 64 bit compare per candidate.
inline constexpr int CELL_KEY_BIAS = 1 << 20;

CPS_HD CPS_INLINE int64_t cell_key(int ix, int iy, int iz) {
    const int64_t bx = static_cast<int64_t>(ix + CELL_KEY_BIAS);
    const int64_t by = static_cast<int64_t>(iy + CELL_KEY_BIAS);
    const int64_t bz = static_cast<int64_t>(iz + CELL_KEY_BIAS);
    return (bx << 42) | (by << 21) | bz;
}

// Reads the exponent bits rather than calling isfinite, so the answer cannot
// depend on what a fast math build is allowed to assume about NaN.
CPS_HD CPS_INLINE bool is_finite_component(float value) {
    union {
        float as_float;
        uint32_t as_bits;
    } pun;
    pun.as_float = value;
    return (pun.as_bits & 0x7F800000u) != 0x7F800000u;
}

CPS_HD CPS_INLINE bool position_is_finite(float x, float y, float z) {
    return is_finite_component(x) && is_finite_component(y) && is_finite_component(z);
}

// The limit on |domain| / h per axis, enforced on the host when the fluid
// solver is built. A position inside the domain then has a coordinate in
// [-(2^20 - 2), 2^20 - 3], and the twenty seven cells around it stay inside
// [-(2^20 - 1), 2^20 - 2], so every key field fits its 21 bits and no two
// cells alias.
inline constexpr int SPH_DOMAIN_CELL_LIMIT = CELL_KEY_BIAS - 2;

// The cell a NaN or Inf position is filed under, since it has no real one. It
// is the one coordinate the limit above leaves unreachable, so no finite
// particle's neighbour search ever matches its key and a non finite particle
// cannot contaminate the particles around it. Its bucket still comes from
// spatial_hash, which keeps it inside the table and inside the digits the
// radix sort walks.
inline constexpr int NON_FINITE_CELL = CELL_KEY_BIAS - 1;

// Smallest prime not less than n, used to size the hash table. A prime
// modulus spreads the exclusive or of the three products far better than a
// power of two mask, which would simply discard the high bits where most of
// the multiplication has done its mixing.
inline uint32_t next_prime(uint32_t n) {
    if (n <= 2u) {
        return 2u;
    }
    uint32_t candidate = (n % 2u == 0u) ? n + 1u : n;
    while (true) {
        bool is_prime = true;
        for (uint64_t d = 3; d * d <= static_cast<uint64_t>(candidate); d += 2) {
            if (candidate % static_cast<uint32_t>(d) == 0u) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) {
            return candidate;
        }
        candidate += 2u;
    }
}

}  // namespace cps
