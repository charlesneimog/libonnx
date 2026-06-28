cmake_minimum_required(VERSION 3.10)
project(libonnx C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(LIBONNX_UPDATE_PROTO "Update Proto Language of ONNX" ON)

set(CPM_VERSION "0.42.1")
set(ONNX_VERSION "v1.22.0")

set(CPM_FILE "${CMAKE_BINARY_DIR}/CPM.cmake")

if(NOT EXISTS "${CPM_FILE}")
    file(DOWNLOAD "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_VERSION}/CPM.cmake" "${CPM_FILE}")
endif()

include("${CPM_FILE}")

set(onnx_sources)

if(LIBONNX_UPDATE_PROTO)
    set(ONNX_PROTO_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/onnx-proto3")
    set(PROTOBUF_C_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

    set(ONNX_PROTO_C "${ONNX_PROTO_OUT_DIR}/onnx.proto3.pb-c.c")
    set(ONNX_PROTO_H "${ONNX_PROTO_OUT_DIR}/onnx.proto3.pb-c.h")

    file(MAKE_DIRECTORY "${ONNX_PROTO_OUT_DIR}")
    file(MAKE_DIRECTORY "${PROTOBUF_C_OUT_DIR}/protobuf-c")

    cpmaddpackage("gh:protobuf-c/protobuf-c@1.5.2")

    cpmaddpackage(
        NAME
        onnx_source
        URL
        "https://github.com/onnx/onnx/archive/refs/tags/${ONNX_VERSION}.zip"
        DOWNLOAD_ONLY
        YES)

    find_program(PROTOC_EXECUTABLE protoc REQUIRED)
    find_program(PROTOC_GEN_C_EXECUTABLE protoc-gen-c REQUIRED)

    set(ONNX_PROTO_DIR "${onnx_source_SOURCE_DIR}/onnx")

    add_custom_command(
        OUTPUT "${ONNX_PROTO_C}" "${ONNX_PROTO_H}"
        COMMAND "${PROTOC_EXECUTABLE}" "-I${ONNX_PROTO_DIR}" "--plugin=protoc-gen-c=${PROTOC_GEN_C_EXECUTABLE}"
                "--c_out=${ONNX_PROTO_OUT_DIR}" "${ONNX_PROTO_DIR}/onnx.proto3"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${protobuf-c_SOURCE_DIR}/protobuf-c"
                "${PROTOBUF_C_OUT_DIR}/protobuf-c"
        DEPENDS "${ONNX_PROTO_DIR}/onnx.proto3"
        COMMENT "Generating ONNX proto and copying protobuf-c headers"
        VERBATIM)

    list(APPEND onnx_sources "${ONNX_PROTO_C}")
endif()

file(
    GLOB
    project_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/onnx-proto3/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/protobuf-c/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/default/*.c")

list(APPEND onnx_sources ${project_sources})

list(REMOVE_DUPLICATES onnx_sources)

add_library(onnx STATIC ${onnx_sources})

target_include_directories(
    onnx SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/src/onnx-proto3"
                       "${CMAKE_CURRENT_SOURCE_DIR}/src/protobuf-c")

target_compile_definitions(onnx PRIVATE HAVE_CONFIG_H)

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
    add_executable(hello "${CMAKE_CURRENT_SOURCE_DIR}/examples/hello/main.c")
    target_link_libraries(hello PRIVATE onnx m)

    add_executable(benchmark "${CMAKE_CURRENT_SOURCE_DIR}/examples/benchmark/main.c")
    target_link_libraries(benchmark PRIVATE onnx m)
endif()
