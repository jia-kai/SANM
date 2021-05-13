set(PROJ_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)
file(GLOB_RECURSE SANM_LIB_SRC ${PROJ_ROOT}/libsanm/*.cpp)
set(BRENT_SRC ${PROJ_ROOT}/third_party/BRENT/brent.cpp)
set_source_files_properties(${BRENT_SRC} PROPERTIES COMPILE_FLAGS -Wno-error)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM INTERFACE
    ${PROJ_ROOT}/third_party/eigen)
target_link_libraries(eigen INTERFACE Threads::Threads)
target_compile_definitions(eigen INTERFACE
    EIGEN_USE_MKL_VML=1 EIGEN_USE_LAPACKE=1)

if (MKLROOT)
    target_include_directories(eigen INTERFACE ${MKLROOT}/include)
    target_link_options(eigen INTERFACE
        -L${MKLROOT}/lib/intel64 -Wl,-rpath=${MKLROOT}/lib/intel64
        )
    message("use MKL at ${MKLROOT}")
else()
    if (ONEAPIROOT)
        target_include_directories(eigen INTERFACE
            ${ONEAPIROOT}/mkl/latest/include)
        target_link_options(eigen INTERFACE
            -L${ONEAPIROOT}/mkl/latest/lib/intel64
            -Wl,-rpath=${ONEAPIROOT}/mkl/latest/lib/intel64
            -L${ONEAPIROOT}/compiler/latest/linux/compiler/lib/intel64_lin
            -Wl,-rpath=${ONEAPIROOT}/compiler/latest/linux/compiler/lib/intel64_lin
            )
        message("use OneAPI at ${ONEAPIROOT}")
    else()
        message(FATAL_ERROR
"Please install Intel MKL and set MKLROOT (containing include/ and \
lib/ subdirectories) or ONEAPIROOT (containing the \
mkl/latest/ subdirectory)")
    endif()
endif()

target_link_options(eigen INTERFACE
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
)

add_library(sanm_lib ${SANM_LIB_SRC} ${BRENT_SRC})
target_include_directories(sanm_lib PUBLIC ${PROJ_ROOT})
target_include_directories(sanm_lib SYSTEM PRIVATE
    ${PROJ_ROOT}/third_party/BRENT)
target_link_libraries(sanm_lib PUBLIC eigen)
