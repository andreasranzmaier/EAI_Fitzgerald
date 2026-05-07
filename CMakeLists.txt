cmake_minimum_required(VERSION 3.20)

set(APP_NAME "mnist_sensehat_demo" CACHE STRING "Application target name")
project(${APP_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TFLITE_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/tensorflow-src" CACHE PATH "TensorFlow source tree")

if(NOT EXISTS "${TFLITE_SRC_DIR}/tensorflow/lite/interpreter.h")
  message(FATAL_ERROR "TensorFlow Lite source tree not found: ${TFLITE_SRC_DIR}. Run `make tflite` to fetch the source tree.")
endif()

add_subdirectory(
  "${TFLITE_SRC_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

find_package(Threads REQUIRED)

file(GLOB APP_SOURCES CONFIGURE_DEPENDS
  src/*.cpp
)

add_executable(${APP_NAME}
  ${APP_SOURCES}
)

target_include_directories(${APP_NAME} PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
)

target_link_libraries(${APP_NAME} PRIVATE
  tensorflow-lite
  Threads::Threads
  dl
  m
)

target_compile_options(${APP_NAME} PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
