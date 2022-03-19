#################################################################################
# @project Common
# @brief Common libraries used throughout the project.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (COMMON_LIB Common)
set (COMMON_LIB ${COMMON_LIB} PARENT_SCOPE)

if (BUILD_ARCHITECTURE STREQUAL "x86")
    set (PROJECT_COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/../Common/ PARENT_SCOPE)
    set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../Common/include PARENT_SCOPE)
    set (PROJECT_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../Common/lib/x86 PARENT_SCOPE)
elseif (BUILD_ARCHITECTURE STREQUAL "x64")
    set (PROJECT_COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/../Common/ PARENT_SCOPE)
    set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../Common/include PARENT_SCOPE)
    set (PROJECT_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../Common/lib/x64 PARENT_SCOPE)
else ()
    message(WARNING "Unknown architecture ${BUILD_ARCHITECTURE}, using x64 defalt")
    set (PROJECT_COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/../Common/ PARENT_SCOPE)
    set (PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../Common/include PARENT_SCOPE)
    set (PROJECT_LIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../Common/lib/x64 PARENT_SCOPE)
endif ()

set (COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set (COMMON_INCLUDE_DIR ${COMMON_INCLUDE_DIR} PARENT_SCOPE)
set (COMMON_LIB_DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libdist)
set (COMMON_LIB_DIST_DIR ${COMMON_LIB_DIST_DIR} PARENT_SCOPE)

add_subdirectory (lib)
