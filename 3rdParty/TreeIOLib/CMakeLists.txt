#################################################################################
# @project TreeIOLib
# @brief Library containing common code and data structures for TreeIO format.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME TreeIOLib)
project (${PROJECT_NAME})
add_subdirectory (Common)
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

#set_binary_dir (TreeIOLib)
add_subdirectory (src)

