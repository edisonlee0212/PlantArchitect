#################################################################################
# @project Common
# @brief Common libraries used throughout the project.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME TinyXML2)
project (${PROJECT_NAME})

set (PROJECT_HEADERS
    tinyxml2/tinyxml2.h
)

set (PROJECT_SOURCES
    tinyxml2/tinyxml2.cpp
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
