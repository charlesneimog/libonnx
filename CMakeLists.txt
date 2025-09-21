cmake_minimum_required(VERSION 3.10)
project(libonnx C)

set(CMAKE_C_STANDARD 99)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

file(GLOB onnx_sources ${libonnx_SOURCE_DIR}/src/*.c
     ${libonnx_SOURCE_DIR}/src/default/*.c)

add_library(onnx STATIC ${onnx_sources})
target_include_directories(onnx PUBLIC ${libonnx_SOURCE_DIR}/src)
set_target_properties(
  onnx
  PROPERTIES C_STANDARD 99
             C_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON)

if(BUILD_TESTS)
  add_executable(tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/main.c")
  target_link_libraries(tests PRIVATE onnx m)
endif()

if(BUILD_EXAMPLES)
  # hello
  add_executable(hello "${CMAKE_CURRENT_SOURCE_DIR}/examples/hello/main.c")
  target_link_libraries(hello PRIVATE onnx m)

  # benchmark
  add_executable(benchmark
                 "${CMAKE_CURRENT_SOURCE_DIR}/examples/benchmark/main.c")
  target_link_libraries(benchmark PRIVATE onnx m)

  # mnist
  find_package(SDL2 REQUIRED)
  add_executable(mnist "${CMAKE_CURRENT_SOURCE_DIR}/examples/mnist/main.c")
  target_link_libraries(mnist PRIVATE onnx m SDL2::SDL2 SDL2_gfx)
endif()
