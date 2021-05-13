cmake_minimum_required(VERSION 3.13)

project(SymbolicANM)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -Wall -Wextra -Wnon-virtual-dtor -Wno-unused-parameter -Werror \
    -march=native")

# the core SANM library
add_subdirectory(libsanm)

# SANM test cases
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/Catch2)
add_subdirectory(tests)

# fea is a 3D deformation application
add_subdirectory(fea)
