#################################################################################
# @project Common
# @brief Common libraries used throughout the project.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME catch2)
project (${PROJECT_NAME})

add_library (
    ${PROJECT_NAME}
    INTERFACE
)

target_include_directories (
    ${PROJECT_NAME}
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)

# Avoid repeatedly including the targets
if (NOT TARGET ${COMMON_LIB}::${PROJECT_NAME})
    # Provide path for scripts
    list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Catch2")

    include (${CMAKE_CURRENT_SOURCE_DIR}/cmake/Catch2/Catch2Targets.cmake)
    include (${CMAKE_CURRENT_SOURCE_DIR}/cmake/Catch2/Catch.cmake)
    include (${CMAKE_CURRENT_SOURCE_DIR}/cmake/Catch2/ParseAndAddCatchTests.cmake)
endif ()

add_library (
    ${COMMON_LIB}::${PROJECT_NAME}
    ALIAS
    ${PROJECT_NAME}
)
