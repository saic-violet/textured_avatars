cmake_minimum_required(VERSION 3.6)
project(stickman-drawer)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -w  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w  -O3  -march=native") #-O3

set(SOURCE_FILES stickman_test.cpp)
add_executable(stickman-drawer ${SOURCE_FILES})

target_link_libraries(stickman-drawer
        opencv_core
        opencv_highgui
        opencv_calib3d
        opencv_imgcodecs
        )