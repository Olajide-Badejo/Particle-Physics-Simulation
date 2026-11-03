# A small project that uses cps the way an outside project does, through
# find_package and nothing else. It is not part of the main build, and it must
# keep working from a copy of this directory, with the repository out of reach:
#
#   cmake --install build --prefix /tmp/cps-inst
#   cmake -S tests/install_consumer -B /tmp/cps-cons -DCMAKE_PREFIX_PATH=/tmp/cps-inst
#   cmake --build /tmp/cps-cons
#   /tmp/cps-cons/cps_consumer
#
# It has only C++ sources, so it does not enable the CUDA language; see the notes
# in cpsConfig.cmake.
cmake_minimum_required(VERSION 3.28)
project(cps_install_consumer LANGUAGES CXX)

find_package(cps 1.1 REQUIRED)

add_executable(cps_consumer main.cpp)
target_link_libraries(cps_consumer PRIVATE cps::physics_core)
