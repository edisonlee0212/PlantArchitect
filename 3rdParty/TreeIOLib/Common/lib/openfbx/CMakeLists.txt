#################################################################################
# @project Common
# @brief Common libraries used throughout the project.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME OpenFBX)
project (${PROJECT_NAME})

set (PROJECT_HEADERS
    openfbx/miniz.h
    openfbx/ofbx.h
)

set (PROJECT_SOURCES
    openfbx/miniz.cpp
    openfbx/ofbx.cpp
)

add_library (
    ${PROJECT_NAME}
    STATIC
    ${PROJECT_HEADERS}
    ${PROJECT_SOURCES}
)

target_include_directories (
    ${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library (
    ${COMMON_LIB}::${PROJECT_NAME}
    ALIAS
    ${PROJECT_NAME}
)
