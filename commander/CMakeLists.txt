cmake_minimum_required(VERSION 3.8)
project(commander)

# Use C++17 for YOLOs-CPP compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(my_robot_interfaces REQUIRED)
find_package(OpenCV REQUIRED)

# YOLOs-CPP and ONNX Runtime paths
set(YOLOS_CPP_DIR "/root/YOLOs-CPP")
set(ONNXRUNTIME_DIR "${YOLOS_CPP_DIR}/onnxruntime-linux-x64-1.20.1")

add_executable(commander src/commander.cpp)

# Include directories for YOLOs-CPP and ONNX Runtime
target_include_directories(commander PRIVATE
  ${YOLOS_CPP_DIR}/include
  ${ONNXRUNTIME_DIR}/include
)

# Link ONNX Runtime library
target_link_libraries(commander
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
  ${OpenCV_LIBS}
)

# Set RPATH for runtime library discovery
set_target_properties(commander PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
  INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)

ament_target_dependencies(commander 
  rclcpp 
  std_msgs 
  sensor_msgs
  cv_bridge
  moveit_ros_planning_interface 
  example_interfaces 
  my_robot_interfaces
  OpenCV
)

install(TARGETS 
  commander
  DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
