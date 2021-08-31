cmake_minimum_required(VERSION 3.17)
project(
        PlantArchitect
        VERSION 0.1
        DESCRIPTION "An interactive application developed for my research purposes based on UniEngine"
)

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
if (NOT WIN32)
    set(OptiX_INSTALL_DIR "~/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64")
endif ()
# We need a CMAKE_DIR with some code to find external dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(3RD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${3RD_PARTY_DIR})


include(${CMAKE_MODULE_PATH}/configure_build_type.cmake)
include(${CMAKE_MODULE_PATH}/configure_optix.cmake)
mark_as_advanced(CUDA_SDK_ROOT_DIR)

include_directories(${OptiX_INCLUDE})

# ------------------------------------------------------------------
# RayTracer
# ------------------------------------------------------------------

set(PLANT_ARCHITECT_RESOURCE_DIR_LOCAL ${CMAKE_CURRENT_SOURCE_DIR}/Resources/)
set(PLANT_ARCHITECT_RESOURCE_DEF PLANT_ARCHITECT_RESOURCE_FOLDER=\"${PLANT_ARCHITECT_RESOURCE_DIR_LOCAL}\")

set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/RayTracerFacility)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${UNIENGINE_INCLUDES})

file(GLOB RAY_TRACER_FACILITY_SOURCES_LOCAL
        "src/RayTracerFacility/*.cpp"
        )
file(GLOB RAY_TRACER_FACILITY_CUDA_SOURCES_LOCAL
        "src/RayTracerFacility/*.cu"
        )

cuda_compile_and_embed(DEFAULT_RENDERING_PTX ${CMAKE_CURRENT_SOURCE_DIR}/src/RayTracerFacility/ptx/DefaultRendering.cu)
cuda_compile_and_embed(ILLUMINATION_ESTIMATION_PTX ${CMAKE_CURRENT_SOURCE_DIR}/src/RayTracerFacility/ptx/IlluminationEstimation.cu)

add_library(RayTracerFacility
        SHARED
        #Default
        ${DEFAULT_RENDERING_PTX}
        ${ILLUMINATION_ESTIMATION_PTX}

        #C++
        ${RAY_TRACER_FACILITY_SOURCES_LOCAL}
        ${RAY_TRACER_FACILITY_CUDA_SOURCES_LOCAL}
        ${CMAKE_MODULE_PATH}/configure_optix.cmake
        ${CMAKE_MODULE_PATH}/FindOptiX.cmake
        )

target_compile_definitions(RayTracerFacility
        PRIVATE
        ${PLANT_ARCHITECT_RESOURCE_DEF}
        )

set_property(TARGET RayTracerFacility PROPERTY CUDA_ARCHITECTURES 75-real 75-virtual)

generate_export_header(RayTracerFacility
        BASE_NAME RAY_TRACER_FACILITY
        EXPORT_MACRO_NAME RAY_TRACER_FACILITY_API
        )

if (NOT WIN32)
    message(STATUS CUDALIB=${CUDA_TOOLKIT_ROOT_DIR})
    set(CUDA_CUDA_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
endif ()
message(STATUS OptiX_INCLUDE:${OptiX_INCLUDE})
message(STATUS CUDA_LIBRARIES:${CUDA_LIBRARIES})
message(STATUS CUDA_CUDA_LIBRARY:${CUDA_CUDA_LIBRARY})
# Linker settings for all platforms
target_link_libraries(RayTracerFacility
        uniengine
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        )

set(RAY_TRACER_FACILITY_INCLUDES_LOCAL
        ${OptiX_INCLUDE}
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/RayTracerFacility
        ${UNIENGINE_INCLUDES}
        ${CMAKE_CURRENT_BINARY_DIR}
        )

target_include_directories(RayTracerFacility
        PRIVATE
        ${RAY_TRACER_FACILITY_INCLUDES_LOCAL}
        )

if (WIN32)
    # Compiler settings for Windows platform
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    target_compile_definitions(RayTracerFacility
            PRIVATE
            RAY_TRACER_FACILITY_EXPORTS
            )
    set(CUDA_PROPAGATE_HOST_FLAGS ON)
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
        ${PLANT_ARCHITECT_SOURCES_LOCAL}
        )
set(PLANT_ARCHITECT_PCH_LOCAL
        ${UNIENGINE_PCH}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/PlantArchitect-pch.hpp
        )
generate_export_header(PlantArchitect
        BASE_NAME PLANT_ARCHITECT
        EXPORT_MACRO_NAME PLANT_ARCHITECT_API
        )
target_compile_definitions(PlantArchitect
        PRIVATE
        GLAD_GLAPI_EXPORT
        ${PLANT_ARCHITECT_RESOURCE_DEF}
        )
target_precompile_headers(PlantArchitect
        PRIVATE
        ${PLANT_ARCHITECT_PCH_LOCAL}
        )
target_include_directories(PlantArchitect
        PUBLIC
        ${RAY_TRACER_FACILITY_INCLUDES_LOCAL}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect
        ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeBehaviours
        ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/InternodeResources
        ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/MeshFormation
        ${CMAKE_CURRENT_SOURCE_DIR}/include/PlantArchitect/quickhull
        )
target_link_libraries(PlantArchitect
        RayTracerFacility
        )

add_executable(Application
        "src/app/PlantArchitect.cpp"
        )

target_include_directories(Application
        PRIVATE
        ${RAY_TRACER_FACILITY_INCLUDES_LOCAL}
        )

target_precompile_headers(Application
        PRIVATE
        ${PLANT_ARCHITECT_PCH_LOCAL}
        )
target_compile_definitions(Application
        PRIVATE
        ${PLANT_ARCHITECT_RESOURCE_DEF}
        )
target_link_libraries(Application
        RayTracerFacility
        PlantArchitect
        )

add_executable(SpaceColonizationExample
        "src/app/SpaceColonizationExample.cpp"
        )

target_include_directories(SpaceColonizationExample
        PRIVATE
        ${RAY_TRACER_FACILITY_INCLUDES_LOCAL}
        )

target_precompile_headers(SpaceColonizationExample
        PRIVATE
        ${PLANT_ARCHITECT_PCH_LOCAL}
        )
target_compile_definitions(SpaceColonizationExample
        PRIVATE
        ${PLANT_ARCHITECT_RESOURCE_DEF}
        )
target_link_libraries(SpaceColonizationExample
        RayTracerFacility
        PlantArchitect
        )

file(COPY src/app/imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (NOT ${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    if (WIN32)
        add_custom_command(
                TARGET RayTracerFacility
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${PROJECT_BINARY_DIR}/RayTracerFacility.dll
                ${CMAKE_BINARY_DIR}
        )
    endif ()
    set(RAY_TRACER_FACILITY_INCLUDES ${RAY_TRACER_FACILITY_INCLUDES_LOCAL} PARENT_SCOPE)
    set(RAY_TRACER_FACILITY_PCH ${PLANT_ARCHITECT_PCH_LOCAL} PARENT_SCOPE)
endif ()
