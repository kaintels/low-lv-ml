cmake_minimum_required(VERSION 3.4...3.18)
project(main VERSION 0.1.0)

include(CTest)
enable_testing()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
SET( CMAKE_MODULE_PATH "$ENV{EIGEN3_DIR}" )
find_package( Eigen3 3.3 REQUIRED)
# include_directories is needed for the compiler to know where looking for Eigen3 header files to be included 
include_directories( ${EIGEN3_INCLUDE_DIR} )

add_executable(main main.cpp layer.h activation.h layer.cpp activation.cpp)