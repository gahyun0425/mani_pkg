cmake_minimum_required(VERSION 3.8)
project(mani_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)

# Eigen3 경로 추가
include_directories(/usr/include/eigen3)

# 실행 파일 추가
add_executable(manipulator_node src/mani_node.cpp)
ament_target_dependencies(manipulator_node rclcpp Eigen3)

# 실행 파일 설치 경로 설정
install(TARGETS manipulator_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

