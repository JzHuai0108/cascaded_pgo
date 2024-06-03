SET(PROJECT_NAME cascaded_pgo)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.3)

SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

include(./CMakeModules/sanitizers.cmake)

# Set build flags. Set IS_ARM on odroid board as environment variable
SET(CMAKE_CXX_FLAGS "-std=c++14 -Wall -D_LINUX -D_REENTRANT -march=native -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")

IF(DEFINED ENV{ARM_ARCHITECTURE})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
ENDIF()

SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Add plain cmake packages 
FIND_PACKAGE(Eigen REQUIRED)
find_package(Ceres 2.0 REQUIRED)

FIND_PACKAGE(catkin REQUIRED COMPONENTS roscpp cmake_modules)
catkin_package(
  DEPENDS Eigen
  CATKIN_DEPENDS roscpp
  INCLUDE_DIRS
  LIBRARIES
)

INCLUDE_DIRECTORIES(
  include
  ${Eigen_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

ADD_EXECUTABLE(pgo
  src/pgo.cpp
  src/pose_factors.cpp
  src/hamilton_quaternion.cpp
  include/okvis/Duration.hpp
  include/okvis/Time.hpp
  include/okvis/implementation/Duration.hpp
  include/okvis/implementation/Time.hpp
  )


TARGET_LINK_LIBRARIES(pgo PUBLIC ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${GFLAGS_LIBRARIES} glog)
