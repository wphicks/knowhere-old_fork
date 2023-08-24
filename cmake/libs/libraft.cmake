# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

set(RAPIDS_VERSION 23.08)
set(RAFT_VERSION "${RAPIDS_VERSION}")
set(RAFT_FORK "rapidsai")
set(RAFT_PINNED_TAG "branch-${RAPIDS_VERSION}")

add_definitions(-DKNOWHERE_WITH_RAFT)

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
  file(
    DOWNLOAD
    https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION}/RAPIDS.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/RAPIDS.cmake)

include(rapids-cpm)  # Dependency tracking
include(rapids-cuda) # Common CMake CUDA logic

rapids_cpm_init()

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

function(find_and_configure_raft)
  set(oneValueArgs VERSION FORK PINNED_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  # -----------------------------------------------------
  # Invoke CPM find_package()
  # -----------------------------------------------------
  rapids_cpm_find(
    raft
    ${PKG_VERSION}
    GLOBAL_TARGETS
    raft::raft
    COMPONENTS
    ${RAFT_COMPONENTS}
    CPM_ARGS
    GIT_REPOSITORY
    https://github.com/${PKG_FORK}/raft.git
    GIT_TAG
    ${PKG_PINNED_TAG}
    SOURCE_SUBDIR
    cpp
    OPTIONS
    "BUILD_TESTS OFF"
    "BUILD_BENCH OFF"
    "RAFT_COMPILE_LIBRARY ON"
    "BUILD_SHARED_LIBS OFF"
    "RAFT_USE_FAISS_STATIC OFF")

    if(raft_ADDED)
        message(VERBOSE "KNOWHERE: Using RAFT located in ${raft_SOURCE_DIR}")
    else()
        message(VERBOSE "KNOWHERE: Using RAFT located in ${raft_DIR}")
    endif()
endfunction()

# Change pinned tag here to test a commit in CI To use a different RAFT locally,
# set the CMake variable CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION ${RAFT_VERSION}.00 FORK ${RAFT_FORK} PINNED_TAG
                        ${RAFT_PINNED_TAG} COMPILE_LIBRARY OFF)
