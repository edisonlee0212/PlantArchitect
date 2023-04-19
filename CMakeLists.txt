cmake_minimum_required(VERSION 3.17)
project(
        PlantArchitect
        VERSION 0.1
        DESCRIPTION "An interactive application developed for my research purposes based on UniEngine"
)

option(BUILD_RAY_TRACER_FACILITY "Build Ray Tracer Facility" ON)
set(BUILD_RAY_TRACER_FACILITY ON)
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
# 3rd Party libraries
# ------------------------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdParty)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(UNIENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../UniEngine)
if(BUILD_RAY_TRACER_FACILITY) 
    set(RAYTRACERFACILITY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../RayTracerFacility)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT TARGET uniengine)
    add_subdirectory(${UNIENGINE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/UniEngine)
endif ()
if(BUILD_RAY_TRACER_FACILITY) 
    if (NOT TARGET RayTracerFacility)
        add_subdirectory(${RAYTRACERFACILITY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/RayTracerFacility)
    endif ()
endif()

include_directories(${UNIENGINE_INCLUDES})
include_directories(${CMAKE_CURRENT_BINARY_DIR})


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
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Orchards
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/TreeIO
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/MeshFormation
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Volumes
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Utilities
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Utilities/quickhull
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeModel/InternodeBehaviours
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeModel/InternodeResources
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeModel/InternodePhyllotaxis
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
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Orchards
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/TreeIO
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/MeshFormation
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Volumes
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Utilities
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/Utilities/quickhull
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeModel/InternodeBehaviours
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeModel/InternodeResources
            ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeModel/InternodePhyllotaxis
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
    target_compile_definitions(Scripts
            PRIVATE
            NOMINMAX
            RAYTRACERFACILITY
            )
else ()
    target_link_libraries(Scripts
            PlantArchitect
            )
    target_compile_definitions(Scripts
            PRIVATE
            NOMINMAX
            )
endif ()

# ------------------------------------------------------------------
# Example program
# ------------------------------------------------------------------
add_executable(TreeFramework
        "src/app/Application.cpp")
target_precompile_headers(TreeFramework
        PRIVATE
        ${PLANT_ARCHITECT_PCH_LOCAL}
        )
target_include_directories(TreeFramework
        PUBLIC
        ${SCRIPTS_INCLUDES_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_link_libraries(TreeFramework
            RayTracerFacility
            PlantArchitect
            Scripts
            )
    target_compile_definitions(TreeFramework
            PRIVATE
            RAYTRACERFACILITY
            NOMINMAX
            )
else ()
    target_link_libraries(TreeFramework
            PlantArchitect
            Scripts
            )
    target_compile_definitions(TreeFramework
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