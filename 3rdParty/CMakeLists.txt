# ------------------------------------------------------------------
# uniengine
# ------------------------------------------------------------------
add_subdirectory(json)
add_subdirectory(csvpp)
add_subdirectory(base64)
set(TREEIO_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/json
        ${CMAKE_CURRENT_SOURCE_DIR}/csvpp
        ${CMAKE_CURRENT_SOURCE_DIR}/base64
        PARENT_SCOPE)