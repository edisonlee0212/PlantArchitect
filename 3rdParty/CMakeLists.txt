# ------------------------------------------------------------------
# uniengine
# ------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/UniEngine/cmake/")
add_subdirectory(UniEngine/ EXCLUDE_FROM_ALL)
set(UNIENGINE_INCLUDES ${UNIENGINE_INCLUDES} PARENT_SCOPE)
set(UNIENGINE_PCH ${UNIENGINE_PCH} PARENT_SCOPE)