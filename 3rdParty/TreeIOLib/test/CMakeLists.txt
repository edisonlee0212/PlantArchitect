#################################################################################
# @project Tests
# @brief This folder contains tests for the application.
# @author Tomas Polasek
#################################################################################

# PROJECT_TEST_NAME and PROJECT_LIB_NAME should have been defined by the parent.

set (TEST_HEADERS
    catch_main.h
)

set (TEST_SOURCES
    catch_main.cpp
    # Add test_*.cpp files here:
    test_sanity.cpp
)

list (TRANSFORM TEST_HEADERS PREPEND "${CMAKE_CURRENT_LIST_DIR}/")
list (TRANSFORM TEST_SOURCES PREPEND "${CMAKE_CURRENT_LIST_DIR}/")
add_executable(
    ${PROJECT_TEST_NAME}
    ${TEST_HEADERS}
    ${TEST_SOURCES}
)

target_link_libraries(
    ${PROJECT_TEST_NAME}
    ${PROJECT_LIB_NAME}
    ${COMMON_LIB}::catch2
)

catch_discover_tests(${PROJECT_TEST_NAME})
