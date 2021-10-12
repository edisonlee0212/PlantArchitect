# ------------------------------------------------------------------
# uniengine
# ------------------------------------------------------------------
add_subdirectory(UniEngine)
set(UNIENGINE_INCLUDES ${UNIENGINE_INCLUDES} PARENT_SCOPE)
set(UNIENGINE_PCH ${UNIENGINE_PCH} PARENT_SCOPE)
if (BUILD_RAY_TRACER_FACILITY)
    add_subdirectory(RayTracerFacility)
    set(RAY_TRACER_FACILITY_INCLUDES ${RAY_TRACER_FACILITY_INCLUDES} PARENT_SCOPE)
    set(RAY_TRACER_FACILITY_PCH ${RAY_TRACER_FACILITY_PCH} PARENT_SCOPE)
endif ()
