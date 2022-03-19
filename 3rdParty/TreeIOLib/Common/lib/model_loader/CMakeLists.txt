###############################################################################
# @project Common
# @brief Common libraries used throughout the project.
# @author Tomas Polasek
###############################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME ModelLoader)
project (${PROJECT_NAME})

add_library (
    ${PROJECT_NAME} STATIC 
    model_loader/ModelLoaderObj.h
    model_loader/ModelLoaderObj.cpp
)

set_target_properties (
    ${PROJECT_NAME} PROPERTIES
    FOLDER
    "3rd_party"
)
target_include_directories (
    ${PROJECT_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/model_loader
)

add_library (
    ${COMMON_LIB}::${PROJECT_NAME}
    ALIAS
    ${PROJECT_NAME}
)
