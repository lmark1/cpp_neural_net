# Find Eigen
find_package(Eigen3)
if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})
endif()

# Make executable neural net
include_directories(${EIGEN_INCLUDE_DIRS})
add_executable(neural_net main.cpp)
target_link_libraries(neural_net ${EIGEN_INCLUDE_DIRS} project_warnings project_options)