cmake_minimum_required(VERSION 3.8)
project(so101_grasp)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Threads REQUIRED)

add_executable(pick_place_node src/pick_place_node.cpp)
target_compile_features(pick_place_node PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pick_place_node PRIVATE -Wall -Wextra -Wpedantic)
endif()
ament_target_dependencies(pick_place_node
  rclcpp moveit_core moveit_ros_planning moveit_ros_planning_interface tf2_ros)
target_link_libraries(pick_place_node Threads::Threads)

add_executable(xyz_ik_test_node src/xyz_ik_test_node.cpp)
target_compile_features(xyz_ik_test_node PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(xyz_ik_test_node PRIVATE -Wall -Wextra -Wpedantic)
endif()
ament_target_dependencies(xyz_ik_test_node
  rclcpp moveit_core moveit_ros_planning moveit_ros_planning_interface tf2_ros)
target_link_libraries(xyz_ik_test_node Threads::Threads)

install(TARGETS pick_place_node xyz_ik_test_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
