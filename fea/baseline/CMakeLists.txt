file(GLOB_RECURSE SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(baseline ${SRC})
target_link_libraries(baseline PUBLIC eigen)
target_include_directories(baseline PRIVATE ${CMAKE_CURRENT_LIST_DIR})
