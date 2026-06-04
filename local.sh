# local.sh - machine-local CADO-NFS build configuration.
# Sourced by scripts/call_cmake.sh before cmake runs. Re-run `make cmake`
# after editing this file. build_tree defaults to build/`hostname`.

# CADO-NFS 2.3.0 declares cmake_minimum_required(VERSION 2.8.11), but cmake 4.x
# removed compatibility with policies < 3.5 and errors out during configure.
# This flag tells cmake to apply 3.5-era policy defaults so the old project
# configures. Drop it if you ever build with cmake < 4.0.
CMAKE_EXTRA_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# GCC 10+ defaults to -fno-common, which turns the project's tentative
# global definitions (e.g. `struct bw_common bw;` declared without `extern`
# in a shared header) into "multiple definition" link errors. -fcommon
# restores the pre-GCC-10 semantics this 2017 codebase was written against.
# C++ has no tentative definitions, so CXXFLAGS is left at its -O2 default.
CFLAGS="-O2 -fcommon"
