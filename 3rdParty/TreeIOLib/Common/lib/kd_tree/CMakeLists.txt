#################################################################################
# @project Common
# @brief Common libraries used throughout the project.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME KdTree)
project (${PROJECT_NAME})

add_library (
	${PROJECT_NAME} STATIC
	kd_tree/KdTree.h
	kd_tree/KdTree.cpp
	kd_tree/PriorityQueue.h
	kd_tree/PriorityQueue.cpp
	kd_tree/Vector2D.h
	kd_tree/Vector2D.cpp
	kd_tree/Vector3D.h
	kd_tree/Vector3D.cpp
)

target_include_directories (
	${PROJECT_NAME}
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kd_tree
)

add_library (
	${COMMON_LIB}::${PROJECT_NAME}
	ALIAS
	${PROJECT_NAME}
)
