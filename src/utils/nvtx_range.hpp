#pragma once

#include <nvtx3/nvToolsExt.h>

namespace cps {

// A named range on the timeline Nsight Systems records. With no tool attached
// the push and the pop each test one function pointer and return.
//
// The engine's stream is non-blocking, so a range around a launch marks the time
// the host spent launching, not the time the kernel ran. Where the device time
// went comes from the kernel records; see scripts/profile_sph_split.sh.
//
// A range is host code, so it cannot be captured. The ranges around a step's passes
// fire once when a graph is captured, inside the "graph capture" range, and never
// when it is launched; only "step" still fires for each step. The kernel records are
// unaffected, and the profiling scripts run with graphs off so they keep one record
// per launch.
//
// This header is private to the device libraries, which link CUDA::nvtx3
// privately. The host library and an installed consumer never see it.
class NvtxRange {
public:
    explicit NvtxRange(const char* name) noexcept { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace cps
