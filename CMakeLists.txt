cmake_minimum_required(VERSION 3.24)

project(OMP VERSION 0.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(MPI REQUIRED)

# find_package(OpenMP REQUIRED)
add_definitions(-O1 -fopenmp)
# add_definitions(-O3)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
file(GLOB_RECURSE CPP_SOURCES "source/*.cpp")

option(BUILD_SHARED_LIBS "Build libs as shared" ON)

set(SOURCE_FILES "OMP.cpp" "MPI.cpp")

foreach(SOURCE_PATH ${SOURCE_FILES})
    get_filename_component(EXECUTABLE_NAME ${SOURCE_PATH} NAME_WE)

    add_executable(${EXECUTABLE_NAME} ${CPP_SOURCES} ${SOURCE_PATH})
    target_link_libraries(${EXECUTABLE_NAME} ${MPI_C_LIBRARIES})
endforeach(SOURCE_PATH ${SOURCE_FILES})
