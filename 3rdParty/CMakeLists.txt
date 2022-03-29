# ------------------------------------------------------------------
# uniengine
# ------------------------------------------------------------------
add_subdirectory(UniEngine)
add_subdirectory(json)
add_subdirectory(csvpp)
add_subdirectory(base64)
set(TREEIO_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/json
        ${CMAKE_CURRENT_SOURCE_DIR}/csvpp
        ${CMAKE_CURRENT_SOURCE_DIR}/base64
        PARENT_SCOPE)

set(UNIENGINE_INCLUDES ${UNIENGINE_INCLUDES} PARENT_SCOPE)
set(UNIENGINE_PCH ${UNIENGINE_PCH} PARENT_SCOPE)
if (BUILD_RAY_TRACER_FACILITY)
    add_subdirectory(RayTracerFacility)
    set(RAY_TRACER_FACILITY_INCLUDES ${RAY_TRACER_FACILITY_INCLUDES} PARENT_SCOPE)
    set(RAY_TRACER_FACILITY_PCH ${RAY_TRACER_FACILITY_PCH} PARENT_SCOPE)
endif ()