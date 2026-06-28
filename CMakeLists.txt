cmake_minimum_required(VERSION 3.10)
project(libonnx C)

set(CMAKE_C_STANDARD 99)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# ╭──────────────────────────────────────╮
# │               Packages               │
# ╰──────────────────────────────────────╯
set(CPM_FILE ${CMAKE_BINARY_DIR}/CPM.cmake)
set(CPM_VERSION "0.42.1")
if(NOT EXISTS "${CPM_FILE}")
    file(DOWNLOAD "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_VERSION}/CPM.cmake" ${CPM_FILE})
endif()
include(${CPM_FILE})

cpmaddpackage("gh:protobuf-c/protobuf-c@1.5.2")

set(ONNX_PROTO_FILE "${CMAKE_BINARY_DIR}/onnx.proto3")
set(ONNX_PROTO_C "${CMAKE_BINARY_DIR}/onnx.proto3.pb-c.c")
set(ONNX_PROTO_H "${CMAKE_BINARY_DIR}/onnx.proto3.pb-c.h")
if(NOT EXISTS "${ONNX_PROTO_FILE}")
    file(DOWNLOAD "https://raw.githubusercontent.com/onnx/onnx/refs/heads/main/onnx/onnx.proto3" "${ONNX_PROTO_FILE}")
endif()

find_program(PROTOC_EXECUTABLE protoc REQUIRED)
find_program(PROTOC_GEN_C_EXECUTABLE protoc-gen-c REQUIRED)

add_custom_command(
    OUTPUT "${ONNX_PROTO_C}" "${ONNX_PROTO_H}"
    COMMAND "${PROTOC_EXECUTABLE}" "-I${CMAKE_BINARY_DIR}" "--c_out=${CMAKE_BINARY_DIR}" "${ONNX_PROTO_FILE}"
    DEPENDS "${ONNX_PROTO_FILE}"
    COMMENT "Generating ONNX protobuf-c sources"
    VERBATIM)

add_library(ONNXProto STATIC "${ONNX_PROTO_C}")

target_include_directories(ONNXProto PUBLIC "${CMAKE_BINARY_DIR}")

# ╭──────────────────────────────────────╮
# │             ONNX Library             │
# ╰──────────────────────────────────────╯
file(
    GLOB
    onnx_sources
    ${libonnx_SOURCE_DIR}/src/*.c
    ${libonnx_SOURCE_DIR}/src/default/*.c)
add_library(onnx STATIC ${onnx_sources})

target_include_directories(onnx SYSTEM PUBLIC ${libonnx_SOURCE_DIR}/src)
target_link_libraries(onnx PUBLIC ONNXProto protobuf-c)
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
    add_executable(benchmark "${CMAKE_CURRENT_SOURCE_DIR}/examples/benchmark/main.c")
    target_link_libraries(benchmark PRIVATE onnx m)

endif()
