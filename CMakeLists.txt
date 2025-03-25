cmake_minimum_required(VERSION 3.8)
project(cascaded_pgo)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

include(./CMakeModules/sanitizers.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build flags. Set IS_ARM on odroid board as an environment variable.
add_compile_options(-O3 -march=native -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno 
    -funroll-loops -Wall -Wextra -Wno-unused-variable -Wno-unused-but-set-variable 
    -Wno-unknown-pragmas -Wnon-virtual-dtor -Wno-unused-function -pedantic)

add_definitions(-DGLOG_USE_GLOG_EXPORT)

if(DEFINED ENV{ARM_ARCHITECTURE})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse2 -msse3 -mssse3")
endif()


find_package(catkin REQUIRED COMPONENTS 
  eigen_catkin
  ceres_catkin
  roscpp)

catkin_package(
  CATKIN_DEPENDS  eigen_catkin  ceres_catkin roscpp
  INCLUDE_DIRS include
  LIBRARIES
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${catkin_INCLUDE_DIRS}
)

add_executable(pgo
  src/Duration.cpp
  src/geodetic_enu_converter.cpp
  src/imu_parameters.cpp
  src/ImuErrorWithGravity.cpp
  src/pgo.cpp
  src/pose_manifold.cpp
  src/Time.cpp
)

target_link_libraries(pgo PUBLIC ${catkin_LIBRARIES} gflags glog)

