# The same image continuous integration builds in, so the toolkit is pinned. The
# build needs no GPU and runs nothing that does. At run time, pass --gpus all for
# the device tests and the simulator; without it the device tests read Skipped.
#
# The tree has no .git in it, so name the commit for the version stamp:
#
#   docker build -t cps:dev --build-arg CPS_GIT_COMMIT="$(git rev-parse --short=12 HEAD)" .
#
# Add a -dirty suffix for an uncommitted tree. Without the argument the image
# reports the commit as "unknown".
FROM nvidia/cuda:13.3.1-devel-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake git make ca-certificates python3 python3-venv build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Named one by one, on top of .dockerignore, so that a file nobody thought about
# stays out by default.
COPY CMakeLists.txt CMakePresets.json Makefile LICENSE requirements.txt ./
COPY benchmarks/ benchmarks/
COPY cmake/ cmake/
COPY configs/ configs/
COPY docs/ docs/
COPY experiments/ experiments/
COPY include/ include/
COPY report/ report/
COPY report_debug/ report_debug/
COPY scripts/ scripts/
COPY src/ src/
COPY tests/ tests/

# The plotting and report scripts, at the pinned versions.
RUN python3 -m venv /opt/cps-python \
    && /opt/cps-python/bin/pip install --no-cache-dir -r requirements.txt

# release-fat, so the simulator runs on any supported GPU. The tests and the
# strict library target sm_120, as they do in every build. The object files are
# not needed once the binaries are linked, and they are most of the build tree.
ARG CPS_GIT_COMMIT=""
RUN cmake --preset release-fat \
    && cmake --build --preset release-fat -j "$(nproc)" \
    && find build-fat -name '*.o' -delete

ENV PATH="/src/build-fat/bin:/opt/cps-python/bin:${PATH}"
