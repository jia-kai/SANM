set(PROJ_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)

add_subdirectory(baseline)
file(GLOB FEA_SRC ${PROJ_ROOT}/fea/*.cpp)
add_executable(fea ${FEA_SRC})
target_include_directories(fea SYSTEM PRIVATE
    ${PROJ_ROOT}/third_party/nlohmann-json/include)
target_link_libraries(fea sanm_lib baseline)
