cmake_minimum_required(VERSION 3.28)

# This has to happen before project(), and it is not optional on this toolchain.
#
# nvcc 13.3 defaults to C++17 for host code. The libstdc++ 15 headers it is
# paired with here use "if consteval" inside c++config.h, which the C++17 front
# end rejects, so any nvcc invocation without an explicit standard fails to
# compile even an empty translation unit. CMake identifies the CUDA compiler by
# compiling a probe file with CMAKE_CUDA_FLAGS and nothing else, long before
# CMAKE_CUDA_STANDARD has any effect, so without this line configuration dies
# during project().
#
# C++20 is deliberately the seed rather than C++23: it is enough to get the
# probe through on every libstdc++ tested, and it does not assume the running
# CMake knows that nvcc can do C++23. The real standard for targets is chosen
# by capability below.
#
# The seed is only for project(). It is a normal variable shadowing the cache,
# and it is withdrawn once the standard is chosen, so that a compile carries one
# -std; see there. Flags that already name a standard, from the cache or from
# the CUDAFLAGS environment variable, belong to the user and are left alone, and
# any other user flags stay in the cache and keep applying. The test is on the
# flags rather than on whether the variable is defined, because after the first
# configure the cache always defines it, and the second configure has to come
# out the same as the first.
set(CPS_CUDA_SEED "-std=c++20")
set(CPS_CUDA_FLAGS_SEEDED OFF)
if(NOT "${CMAKE_CUDA_FLAGS} $ENV{CUDAFLAGS}" MATCHES "-std=")
  string(STRIP "${CMAKE_CUDA_FLAGS} ${CPS_CUDA_SEED}" CMAKE_CUDA_FLAGS)
  set(CPS_CUDA_FLAGS_SEEDED ON)
endif()

# Also before project(), and for a related reason: project() assigns a default
# architecture as soon as it enables the CUDA language, so a guard placed after
# it never fires and the build silently targets the wrong card. The reference
# GPU is an RTX 5070, compute capability 12.0. A value passed on the command
# line is already in the cache by this point and wins.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures to build for")
endif()

# Host only mode builds the CUDA free half of the project: the configuration
# layer, the snapshot format, the initial condition generators and the physics
# formulas, together with the unit tests that cover them. It exists so that a
# machine with no CUDA toolkit at all can still build and run a meaningful part
# of the test suite, which is what the continuous integration host job does.
# The CUDA language is not even enabled in this mode.
option(CPS_HOST_ONLY "Build only the host library and its tests, with no CUDA" OFF)

if(CPS_HOST_ONLY)
  set(CPS_LANGUAGES CXX)
else()
  set(CPS_LANGUAGES CXX CUDA)
endif()

project(cuda_physics_sim
        VERSION 1.1.0
        DESCRIPTION "CUDA particle physics engine: tiled and fused N-body gravity plus SPH fluids"
        LANGUAGES ${CPS_LANGUAGES})

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# Fast math is a benchmark-only switch. The energy drift gate is sensitive to
# the reduced precision of rsqrtf, so the validation library below is compiled
# with fast math forced off regardless of what the user asks for here.
option(CPS_USE_FAST_MATH "Compile the benchmark library with --use_fast_math" OFF)
option(CPS_BUILD_TESTS "Build the unit, integration and validation tests" ON)
option(CPS_BUILD_BENCHMARKS "Build the benchmark sweep driver" ON)
option(CPS_WERROR "Treat compiler warnings as errors" ON)
# Drains the engine's stream after every launch, so an ordering bug can be
# bisected. Results must not change with it on; timings are meaningless.
option(CPS_DEBUG_SYNC "Synchronise the stream after every launch and copy" OFF)
# Everyday builds compile the device code for one architecture, which is fast.
# A fat build adds every architecture a release should run on, but only to
# physics_core: physics_core_strict and the test binaries run on the development
# card, and leaving them out halves the extra device compiles.
option(CPS_FAT_BINARY "Build physics_core for every release architecture" OFF)
set(CPS_FAT_ARCHITECTURES 75 80 86 89 90 120)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The device standard is chosen by asking CMake what it can actually offer,
# rather than by asserting C++23 and hoping.
#
# Knowing that nvcc supports the C++23 dialect is a property of the running
# CMake, not of the compiler: CMake 4.4 lists cuda_std_23, while the CMake 3.28
# shipped by Ubuntu 24.04 does not, and asking it for CUDA23 fails at generate
# time with "the current compiler NVIDIA does not support this". The host
# standard stays at C++23 either way; nothing in the device code needs a C++23
# feature, so the fallback costs nothing but is worth reporting.
if(NOT CPS_HOST_ONLY)
  if(NOT DEFINED CPS_CUDA_STANDARD)
    if("cuda_std_23" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
      set(CPS_CUDA_STANDARD 23)
    else()
      set(CPS_CUDA_STANDARD 20)
    endif()
  endif()
  set(CMAKE_CUDA_STANDARD ${CPS_CUDA_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
# The probe is done, so the seed comes out of the flags, which leaves the cached
# value: empty, or whatever the user supplied. CMAKE_CUDA_STANDARD now provides
# the one -std. It does so for C++20 as well, although the probe measured that as
# the default under the seed: CMake 3.28 and 4.4 were both checked.
if(CPS_CUDA_FLAGS_SEEDED)
  unset(CMAKE_CUDA_FLAGS)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CPS_HOST_ONLY)
  find_package(CUDAToolkit REQUIRED)
endif()

# Every benchmark row records the commit it was produced by, so a number in the
# report can always be traced back to the exact tree that generated it. The
# hash is read at build time, by the cps_version target below, because one read
# at configure time goes stale with the next commit.
find_package(Git QUIET)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Only the custom targets and the test of the documentation tooling need Python,
# so a machine without it can still build and test everything else.
find_package(Python3 COMPONENTS Interpreter QUIET)

# ---------------------------------------------------------------------------
# Third party dependencies
# ---------------------------------------------------------------------------
# An installed copy is used when there is one at the pinned version or newer, so
# a machine with the packages needs no network. Otherwise the pinned tag is
# fetched. The floor is the pin rather than nothing because Ubuntu 24.04 packages
# older copies of both, and an older JSON library or test framework would change
# what a build was tested against without saying so.
include(FetchContent)

FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
  GIT_SHALLOW TRUE
  SYSTEM
  FIND_PACKAGE_ARGS 3.12.0 CONFIG QUIET)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

if(CPS_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
    SYSTEM
    FIND_PACKAGE_ARGS 1.17.0 NAMES GTest QUIET)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(CTest)
  enable_testing()
endif()

# ---------------------------------------------------------------------------
# Shared warning configuration
# ---------------------------------------------------------------------------
# MSVC spells these differently. This only keeps the flags valid there: full MSVC
# support is deferred.
add_library(cps_warnings INTERFACE)
if(MSVC)
  target_compile_options(cps_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
  if(CPS_WERROR)
    target_compile_options(cps_warnings INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/WX>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>)
  endif()
else()
  target_compile_options(cps_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
  if(CPS_WERROR)
    target_compile_options(cps_warnings INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
  endif()
endif()

# ---------------------------------------------------------------------------
# physics_host: pure host code with no CUDA runtime dependency.
# Kept separate so continuous integration can build and run a meaningful slice
# of the unit tests on a machine that has no GPU.
# ---------------------------------------------------------------------------
add_library(physics_host STATIC
  src/core/cli.cpp
  src/core/config.cpp
  src/core/physics_fingerprint.cpp
  src/core/snapshot.cpp
  src/core/initial_conditions.cpp
  src/utils/progress.cpp)
target_include_directories(physics_host PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# No installed header includes the JSON library, and it is header only, so it
# ends at the build tree: an installed consumer never needs it, and the export
# never names a fetched target. The same goes for the warning flags.
target_link_libraries(physics_host
  PUBLIC $<BUILD_LOCAL_INTERFACE:nlohmann_json::nlohmann_json>
  PRIVATE $<BUILD_LOCAL_INTERFACE:cps_warnings>)
target_compile_features(physics_host PUBLIC cxx_std_20)
add_library(cps::physics_host ALIAS physics_host)

# ---------------------------------------------------------------------------
# Build-time version stamp
# ---------------------------------------------------------------------------
# Always runs, and rewrites the header only when its contents change, so a build
# of an unchanged tree recompiles nothing. It is generated under cps/, so the
# include reads the same in the build tree and the installed tree.
set(CPS_GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
add_custom_target(cps_version ALL
  COMMAND ${CMAKE_COMMAND}
          -DPROJECT_VERSION=${PROJECT_VERSION}
          -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
          -DOUTPUT=${CPS_GENERATED_DIR}/cps/version.hpp
          -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_version_header.cmake
  BYPRODUCTS ${CPS_GENERATED_DIR}/cps/version.hpp
  VERBATIM)

# ---------------------------------------------------------------------------
# The device library is built twice from identical sources. The only
# difference is the fast math setting, which lets the benchmark ladder use
# rsqrtf while the physics gates run on IEEE compliant square roots.
# ---------------------------------------------------------------------------
if(NOT CPS_HOST_ONLY)

set(CPS_DEVICE_SOURCES
  src/core/engine.cpp
  src/core/particle_system.cu
  src/core/diagnostics.cu
  src/kernels/nbody_naive.cu
  src/kernels/nbody_tiled.cu
  src/kernels/nbody_fused.cu
  src/kernels/integrators.cu
  src/kernels/spatial_hash.cu
  src/kernels/sph_forces.cu
  src/solvers/sph_solver.cpp
  src/utils/cuda_helpers.cpp)

function(cps_add_device_library target fast_math architectures)
  add_library(${target} STATIC ${CPS_DEVICE_SOURCES})
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(${target}
    PUBLIC physics_host CUDA::cudart
    PRIVATE $<BUILD_LOCAL_INTERFACE:cps_warnings>)
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${architectures}"
    CUDA_SEPARABLE_COMPILATION OFF)
  if(fast_math)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    target_compile_definitions(${target} PUBLIC CPS_FAST_MATH=1)
  else()
    target_compile_definitions(${target} PUBLIC CPS_FAST_MATH=0)
  endif()
  # PUBLIC, because the helper is a macro in a header every consumer includes.
  target_compile_definitions(${target} PUBLIC CPS_DEBUG_SYNC=$<BOOL:${CPS_DEBUG_SYNC}>)
endfunction()

if(CPS_FAT_BINARY)
  set(CPS_CORE_ARCHITECTURES ${CPS_FAT_ARCHITECTURES})
else()
  set(CPS_CORE_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endif()

cps_add_device_library(physics_core ${CPS_USE_FAST_MATH} "${CPS_CORE_ARCHITECTURES}")
cps_add_device_library(physics_core_strict OFF "${CMAKE_CUDA_ARCHITECTURES}")
add_library(cps::physics_core ALIAS physics_core)

# The stamp is for whoever links physics_core, the same as in the installed
# tree. INTERFACE, because nothing physics_core compiles includes it.
target_include_directories(physics_core INTERFACE $<BUILD_INTERFACE:${CPS_GENERATED_DIR}>)
add_dependencies(physics_core cps_version)

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(cuda_physics_sim src/main.cpp)
target_link_libraries(cuda_physics_sim PRIVATE physics_core cps_warnings)
add_executable(cps::cuda_physics_sim ALIAS cuda_physics_sim)

add_executable(gpu_probe src/tools/gpu_probe.cu)
target_link_libraries(gpu_probe PRIVATE physics_core cps_warnings)
set_target_properties(gpu_probe PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

if(CPS_BUILD_BENCHMARKS)
  add_executable(bench_all benchmarks/bench_all.cpp)
  target_link_libraries(bench_all PRIVATE physics_core cps_warnings)
  target_compile_definitions(bench_all PRIVATE
    CPS_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
endif()

endif()  # NOT CPS_HOST_ONLY

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CPS_BUILD_TESTS)
  # Tests that read a shipped configuration need to know where the repository
  # is, whatever directory ctest runs them from. Recorded results go to the
  # build tree by default, so the tests also need to know where that is; only
  # CPS_RECORD_GATES=1 at run time sends them to experiments/results instead.
  add_library(cps_test_support INTERFACE)
  target_compile_definitions(cps_test_support INTERFACE
    CPS_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    CPS_BINARY_DIR="${CMAKE_BINARY_DIR}")
  target_include_directories(cps_test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

  # Host only: no CUDA runtime, no GPU, safe for continuous integration.
  add_executable(unit_tests_host
    tests/unit/test_bench_rows.cpp
    tests/unit/test_cli.cpp
    tests/unit/test_config.cpp
    tests/unit/test_gate_recording.cpp
    tests/unit/test_snapshot_format.cpp
    tests/unit/test_sph_kernels.cpp
    tests/unit/test_spatial_hash.cpp
    tests/unit/test_initial_conditions.cpp)
  target_link_libraries(unit_tests_host
    PRIVATE physics_host GTest::gtest_main cps_warnings cps_test_support)
  add_test(NAME unit_host COMMAND unit_tests_host)
  set_tests_properties(unit_host PROPERTIES LABELS "host")

  # The validation cases that make no CUDA call, directly or through a library:
  # the shared pair force expression, the Plummer sampler and the SPH kernel
  # normalisation. Built in both modes, so continuous integration runs them.
  # Both validation entries merge rows into the same gate file, so they never
  # run at the same time.
  add_executable(validation_tests_host
    tests/validation/host/test_convergence_host.cpp
    tests/validation/host/test_force_antisymmetry_host.cpp
    tests/validation/host/test_plummer_density_host.cpp
    tests/validation/host/test_sph_validation_host.cpp)
  target_link_libraries(validation_tests_host
    PRIVATE physics_host GTest::gtest_main cps_warnings cps_test_support)
  add_test(NAME validation_host COMMAND validation_tests_host)
  set_tests_properties(validation_host PROPERTIES
    LABELS "host;validation" RESOURCE_LOCK validation_gates)

  # The tool that keeps the numbers in the Markdown files from drifting.
  if(Python3_Interpreter_FOUND)
    add_test(NAME doc_numbers_tool
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/test_check_doc_numbers.py)
    set_tests_properties(doc_numbers_tool PROPERTIES LABELS "host")
  endif()

  # Everything below needs the CUDA runtime, so it is skipped entirely in host
  # only mode rather than built and skipped at run time.
  if(NOT CPS_HOST_ONLY)

  # The GPU binaries use their own main in place of gtest_main. With no usable
  # device it exits with code 77, which SKIP_RETURN_CODE turns into "Not Run"
  # instead of a pass made entirely of skipped cases.
  add_library(cps_gpu_test_main STATIC tests/gpu_main.cpp)
  target_link_libraries(cps_gpu_test_main
    PUBLIC GTest::gtest CUDA::cudart cps_test_support physics_host PRIVATE cps_warnings)

  # Device unit tests.
  add_executable(unit_tests_gpu
    tests/unit/test_device_reductions.cu
    tests/unit/test_kernel_parity.cu)
  target_link_libraries(unit_tests_gpu PRIVATE physics_core_strict cps_gpu_test_main cps_warnings cps_test_support)
  set_target_properties(unit_tests_gpu PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  add_test(NAME unit_gpu COMMAND unit_tests_gpu)
  set_tests_properties(unit_gpu PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 77)

  add_executable(integration_tests
    tests/integration/test_solver_smoke.cpp
    tests/integration/test_snapshot_resume.cpp
    tests/integration/test_diagnostics.cpp
    tests/integration/test_input_safety.cpp)
  target_link_libraries(integration_tests
    PRIVATE physics_core_strict cps_gpu_test_main cps_warnings cps_test_support)
  add_test(NAME integration COMMAND integration_tests)
  set_tests_properties(integration PROPERTIES LABELS "gpu" TIMEOUT 900 SKIP_RETURN_CODE 77)

  # Validation always links the strict library so the physics gates can never
  # be silently relaxed by a fast math build.
  add_executable(validation_tests
    tests/validation/test_energy_drift.cpp
    tests/validation/test_momentum.cpp
    tests/validation/test_force_antisymmetry.cpp
    tests/validation/test_sph_validation.cpp)
  target_link_libraries(validation_tests
    PRIVATE physics_core_strict cps_gpu_test_main cps_warnings cps_test_support)
  add_test(NAME validation COMMAND validation_tests)
  set_tests_properties(validation PROPERTIES
    LABELS "gpu;validation" TIMEOUT 3600 SKIP_RETURN_CODE 77 RESOURCE_LOCK validation_gates)

  endif()  # NOT CPS_HOST_ONLY
endif()

# ---------------------------------------------------------------------------
# Style gate
# ---------------------------------------------------------------------------
if(Python3_Interpreter_FOUND)
  add_custom_target(check_style
    COMMAND ${CMAKE_COMMAND} -E echo "Running the dash gate over ${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_no_dashes.py
            ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM)
endif()

# ---------------------------------------------------------------------------
# Install and export
# ---------------------------------------------------------------------------
# physics_core_strict and the test binaries exist to test this repository, so
# they stay out. Only the headers, the libraries and the simulator are installed.
set(CPS_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/cps)

set(CPS_INSTALL_TARGETS physics_host)
if(NOT CPS_HOST_ONLY)
  list(APPEND CPS_INSTALL_TARGETS physics_core cuda_physics_sim)
  # The CUDA runtime is a shared library outside the prefix, and the build tree
  # path to it is dropped on install. Keep it, so the installed simulator runs
  # without LD_LIBRARY_PATH on a toolkit that is not on the loader's path.
  set_target_properties(cuda_physics_sim PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
endif()

install(TARGETS ${CPS_INSTALL_TARGETS}
  EXPORT cpsTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/cps DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CPS_GENERATED_DIR}/cps/version.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cps)
install(EXPORT cpsTargets
  NAMESPACE cps::
  DESTINATION ${CPS_INSTALL_CMAKEDIR})

# For the consumer notes and the find_dependency in cpsConfig.cmake.
if(CPS_HOST_ONLY)
  set(CPS_CUDA_TOOLKIT_FLOOR "(none, this is a host only build)")
  set(CPS_CORE_ARCH_LIST "(none, this is a host only build)")
else()
  set(CPS_CUDA_TOOLKIT_FLOOR ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})
  string(REPLACE ";" ", " CPS_CORE_ARCH_LIST "${CPS_CORE_ARCHITECTURES}")
endif()

configure_package_config_file(cmake/cpsConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cpsConfig.cmake
  INSTALL_DESTINATION ${CPS_INSTALL_CMAKEDIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/cpsConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cpsConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cpsConfigVersion.cmake
  DESTINATION ${CPS_INSTALL_CMAKEDIR})

message(STATUS "cuda_physics_sim ${PROJECT_VERSION}")
if(NOT CPS_HOST_ONLY)
  message(STATUS "  CUDA toolkit      : ${CUDAToolkit_VERSION}")
  if(CPS_FAT_BINARY)
    message(STATUS "  CUDA architectures: physics_core ${CPS_CORE_ARCHITECTURES}, "
                   "everything else ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES} (for the release list use "
                   "-DCPS_FAT_BINARY=ON, for this machine's GPU "
                   "-DCMAKE_CUDA_ARCHITECTURES=native)")
  endif()
endif()
message(STATUS "  Host compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Host standard     : C++${CMAKE_CXX_STANDARD}")
if(NOT CPS_HOST_ONLY)
  message(STATUS "  Device standard   : C++${CMAKE_CUDA_STANDARD}")
endif()
message(STATUS "  Fast math         : ${CPS_USE_FAST_MATH} (validation library always off)")
if(nlohmann_json_SOURCE_DIR)
  message(STATUS "  nlohmann_json     : fetched at the pinned tag")
else()
  message(STATUS "  nlohmann_json     : found ${nlohmann_json_VERSION} at ${nlohmann_json_DIR}")
endif()
if(CPS_BUILD_TESTS)
  if(googletest_SOURCE_DIR)
    message(STATUS "  GoogleTest        : fetched at the pinned tag")
  else()
    message(STATUS "  GoogleTest        : found ${googletest_VERSION} at ${googletest_DIR}")
  endif()
endif()
