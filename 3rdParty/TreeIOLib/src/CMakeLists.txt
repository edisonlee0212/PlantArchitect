#################################################################################
# @project TreeIOLib
# @brief Library containing common code and data structures for TreeIO format.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME TreeIOLib)
project (${PROJECT_NAME})

set (STATIC_LIB_NAME TreeIOLib_static)
set (SHARED_LIB_NAME TreeIOLib)
set (SHARED_LIB_NAME TreeIOLib)
set (PROJECT_HEADERS
    TreeIO/impl/TreeIOVectorT.h
    TreeIO/impl/TreeIOUtils.h

    TreeIO/impl/TreeIOImageImporter.h
    TreeIO/impl/TreeIOModelImporter.h
    TreeIO/impl/TreeIOObjImporter.h
    TreeIO/impl/TreeIOImageExporter.h
    TreeIO/impl/TreeIOFbxImporter.h

    TreeIO/impl/TreeIOTree.h
    TreeIO/impl/TreeIOChains.h
)

set (PROJECT_SOURCES
    TreeIO/impl/TreeIOUtils.cpp

    TreeIO/impl/TreeIOImageImporter.cpp
    TreeIO/impl/TreeIOModelImporter.cpp
    TreeIO/impl/TreeIOObjImporter.cpp
    TreeIO/impl/TreeIOImageExporter.cpp
    TreeIO/impl/TreeIOFbxImporter.cpp

    TreeIO/impl/TreeIOTree.cpp
    TreeIO/impl/TreeIOChains.cpp
)

find_package (glm CONFIG REQUIRED)
if (WIN32)
    find_package (libpng CONFIG REQUIRED)
    find_package (JPEG REQUIRED)
    set (COMMON_LIBRARIES
        ${glm_LIBRARIES}
        png
        ${JPEG_LIBRARIES}
    )
else ()
    set (COMMON_LIBRARIES
        glm::glm
        png
        jpeg
    )
endif ()

set (PROJECT_LIBRARIES
    ${COMMON_LIBRARIES}
    ${COMMON_LIB}::Base64
    ${COMMON_LIB}::boost
    ${COMMON_LIB}::cimg
    ${COMMON_LIB}::csvpp
    ${COMMON_LIB}::JSON
    ${COMMON_LIB}::HDF5
    ${COMMON_LIB}::ModelLoader
    ${COMMON_LIB}::TinyObjLoader
    ${COMMON_LIB}::OpenFBX
)

add_library (
    ${STATIC_LIB_NAME} STATIC
    ${PROJECT_HEADERS}
    ${PROJECT_SOURCES}
)

target_include_directories (
    ${STATIC_LIB_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/
)

target_link_libraries (
    ${STATIC_LIB_NAME}
    ${PROJECT_LIBRARIES}
)

add_library (
    TreeIO::${STATIC_LIB_NAME}
    ALIAS
    ${STATIC_LIB_NAME}
)

add_library (
    ${SHARED_LIB_NAME} SHARED
    ${PROJECT_HEADERS}
    ${PROJECT_SOURCES}
)

target_include_directories (
    ${SHARED_LIB_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/
)

target_link_libraries (
    ${SHARED_LIB_NAME}
    ${PROJECT_LIBRARIES}
)

add_library (
    TreeIO::${SHARED_LIB_NAME}
    ALIAS
    ${SHARED_LIB_NAME}
)

# Prepare variables for the test generation.
set (PROJECT_LIB_NAME ${STATIC_LIB_NAME})
set (PROJECT_TEST_NAME ${PROJECT_NAME}_test)
# Prepare project tests.
include (../test/CMakeLists.txt)
