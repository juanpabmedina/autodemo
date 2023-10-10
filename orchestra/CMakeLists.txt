# Set up ARGoS compilation information
include_directories(${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS})
link_directories(${ARGOS_LIBRARY_DIRS})
link_directories(
	${CMAKE_BINARY_DIR}/src
    )

add_library(template SHARED Template.h Template.cpp)
target_link_libraries(template argos3plugin_${ARGOS_BUILD_FOR}_epuck)

add_library(foraging SHARED Foraging.h Foraging.cpp)
target_link_libraries(foraging argos3plugin_${ARGOS_BUILD_FOR}_epuck)

