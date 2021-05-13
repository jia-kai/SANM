set(PROJ_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)

file(GLOB_RECURSE SANM_TESTS_SRC ${PROJ_ROOT}/tests/*.cpp)
add_executable(sanm_tests ${SANM_TESTS_SRC})
target_link_libraries(sanm_tests Catch2::Catch2 sanm_lib)
