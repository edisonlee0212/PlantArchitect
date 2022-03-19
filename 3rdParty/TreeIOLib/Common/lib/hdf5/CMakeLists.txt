#################################################################################
# @project Common
# @brief Common libraries used throughout the project.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME HDF5)
project (${PROJECT_NAME})

add_library (
    ${PROJECT_NAME}
    INTERFACE
)

target_include_directories (
    ${PROJECT_NAME}
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package (HDF5 REQUIRED)
target_include_directories (
    ${PROJECT_NAME} SYSTEM
    INTERFACE
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries (
    ${PROJECT_NAME}
    INTERFACE
    ${HDF5_C_LIBRARIES}
)
target_compile_definitions (
    ${PROJECT_NAME}
    INTERFACE
    ${HDF5_DEFINITIONS}
)

add_library (
    ${COMMON_LIB}::${PROJECT_NAME}
    ALIAS
    ${PROJECT_NAME}
)
