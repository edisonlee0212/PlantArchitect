cmake_minimum_required(VERSION 3.17)
project(
        PlantArchitect
        VERSION 0.1
        DESCRIPTION "An interactive application developed for my research purposes based on UniEngine"
)

option(BUILD_RAY_TRACER_FACILITY "Build Ray Tracer Facility" ON)
set(BUILD_RAY_TRACER_FACILITY OFF)
include(GenerateExportHeader)

# Set a default build type if none was specified
set(default_build_type "Release")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release")
endif ()

# ------------------------------------------------------------------
# 3rd Party libraries
# ------------------------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdParty)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)
if (WIN32)
    # Compiler settings for Windows platform
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
elseif (UNIX AND NOT APPLE)
    if (${CMAKE_BUILD_TYPE} STREQUAL Release)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    elseif (${CMAKE_BUILD_TYPE} STREQUAL Debug)
        set(CMAKE_CXX_FLAGS_DEBUG "-g")
    endif ()
elseif (APPLE)
endif ()
# ------------------------------------------------------------------
# PlantArchitect
# ------------------------------------------------------------------
file(GLOB PLANT_ARCHITECT_SOURCES_LOCAL "src/PlantArchitect/*.cpp")
add_library(PlantArchitect
        SHARED
        ${PLANT_ARCHITECT_SOURCES_LOCAL}
        )
set(PLANT_ARCHITECT_PCH_LOCAL
        ${UNIENGINE_PCH}
        ${RAY_TRACER_FACILITY_PCH}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/PlantArchitect-pch.hpp
        )
generate_export_header(PlantArchitect
        BASE_NAME PLANT_ARCHITECT
        EXPORT_MACRO_NAME PLANT_ARCHITECT_API
        )

target_precompile_headers(PlantArchitect
        PRIVATE
        ${PLANT_ARCHITECT_PCH_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_compile_definitions(PlantArchitect
            PRIVATE
            GLAD_GLAPI_EXPORT
            NOMINMAX
            RAYTRACERFACILITY
            )
    set(PLANT_ARCHITECT_INCLUDES_LOCAL
            ${RAY_TRACER_FACILITY_INCLUDES}
            ${TREEIO_INCLUDES}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/TreeIO
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/PlantModels
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeResources
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodePhyllotaxis
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/MeshFormation
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Volumes
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Utilities
            include/PlantArchitect/Utilities/quickhull
            )
    target_link_libraries(PlantArchitect
            RayTracerFacility
            csvpp
            JSON
            Base64
            )
else ()
    target_compile_definitions(PlantArchitect
            PRIVATE
            GLAD_GLAPI_EXPORT
            NOMINMAX
            )
    set(PLANT_ARCHITECT_INCLUDES_LOCAL
            ${UNIENGINE_INCLUDES}
            ${TREEIO_INCLUDES}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/TreeIO
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/PlantModels
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeResources
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodePhyllotaxis
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/MeshFormation
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Volumes
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Utilities
            include/PlantArchitect/Utilities/quickhull
            )
    target_link_libraries(PlantArchitect
            uniengine
            csvpp
            JSON
            Base64
            )
endif ()
target_include_directories(PlantArchitect
        PUBLIC
        ${PLANT_ARCHITECT_INCLUDES_LOCAL}
        )
# ------------------------------------------------------------------
# Scripts
# ------------------------------------------------------------------
file(GLOB SCRIPTS_SOURCES_LOCAL "src/Scripts/*.cpp")
add_library(Scripts
        ${SCRIPTS_SOURCES_LOCAL}
        )
target_precompile_headers(Scripts
        PRIVATE
        ${UNIENGINE_PCH_LOCAL}
        ${PLANT_ARCHITECT_PCH_LOCAL}
        )
set(SCRIPTS_INCLUDES_LOCAL
        ${PLANT_ARCHITECT_INCLUDES_LOCAL}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Scripts
        )
target_include_directories(Scripts
        PUBLIC
        ${SCRIPTS_INCLUDES_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_link_libraries(Scripts
            RayTracerFacility
            PlantArchitect
            )
else ()
    target_link_libraries(Scripts
            PlantArchitect
            )
endif ()
target_compile_definitions(Scripts
        PRIVATE
        NOMINMAX
        )
# ------------------------------------------------------------------
# Example program
# ------------------------------------------------------------------
add_executable(Application
        "src/app/Application.cpp")
target_precompile_headers(Application
        PRIVATE
        ${PLANT_ARCHITECT_PCH_LOCAL}
        )
target_include_directories(Application
        PUBLIC
        ${SCRIPTS_INCLUDES_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_link_libraries(Application
            RayTracerFacility
            PlantArchitect
            Scripts
            )
    target_compile_definitions(Application
            PRIVATE
            RAYTRACERFACILITY
            NOMINMAX
            )
else ()
    target_link_libraries(Application
            PlantArchitect
            Scripts
            )
    target_compile_definitions(Application
            PRIVATE
            NOMINMAX
            )
endif ()

# ------------------------------------------------------------------
# Copy Internal resources
# ------------------------------------------------------------------
add_custom_command(TARGET PlantArchitect POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Internals
        ${CMAKE_BINARY_DIR})

file(COPY src/app/imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})