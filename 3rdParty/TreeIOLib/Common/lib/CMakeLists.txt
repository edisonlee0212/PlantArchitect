#################################################################################
# @project Common
# @brief Common libraries used throughout the project.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (CMAKE_CXX_STANDARD 17)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC --no-gnu-unique")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
endif ()
set (default_build_type "Release")

add_subdirectory (base64)
add_subdirectory (boost)
add_subdirectory (catch2)
add_subdirectory (cimg)
add_subdirectory (csvpp)
add_subdirectory (hdf5)
add_subdirectory (intersect)
add_subdirectory (json)
add_subdirectory (kd_tree)
add_subdirectory (model_loader)
add_subdirectory (nanoflann)
add_subdirectory (openfbx)
add_subdirectory (tiny_obj_loader)
add_subdirectory (tinyxml2)
