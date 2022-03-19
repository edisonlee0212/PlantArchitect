#################################################################################
# @project Common
# @brief Common libraries used throughout the project.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME cimg)
project (${PROJECT_NAME})

add_library (
    ${PROJECT_NAME}
    INTERFACE
)

target_include_directories (
    ${PROJECT_NAME}
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library (
    ${COMMON_LIB}::${PROJECT_NAME}
    ALIAS
    ${PROJECT_NAME}
)
