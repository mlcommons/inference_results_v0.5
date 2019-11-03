cmake_minimum_required(VERSION 3.5.1)

if(${ENABLE_SANITIZER} STREQUAL "ON")
    add_compile_options("${SANITIZER_FLAGS}")
endif()
project(HABANA_benchmark_app)
set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/source")
set(TARGET_PATH "$ENV{MLPERF_INSTALL_DIR}/app")
link_directories($ENV{MLPERF_LIB_PATH}   /usr/lib/habanalabs ${CMAKE_CURRENT_LIST_DIR}/headers ${CMAKE_CURRENT_LIST_DIR}/lib/${CMAKE_BUILD_TYPE}/)
message("LOAD GEN LIB PATH: " $ENV{MLPERF_LIB_PATH})
add_executable(HABANA_benchmark_app "${SOURCE_DIR}/main.cpp" "${SOURCE_DIR}/HABANA_runner_c_interface.cpp" "${SOURCE_DIR}/INIReader.cpp" "${SOURCE_DIR}/ini.c")
target_include_directories(HABANA_benchmark_app PUBLIC  "${CMAKE_CURRENT_LIST_DIR}/../global_headers" "${CMAKE_CURRENT_LIST_DIR}/headers/" "${CMAKE_CURRENT_LIST_DIR}/../runner/headers/" "$ENV{MLPERF_LOADGEN_PATH}" "$ENV{MLPERF_LOADGEN_PATH}/bindings" "/usr/lib/habanalabs/include")
if(${ENABLE_SANITIZER} STREQUAL "ON")

    target_link_libraries(HABANA_benchmark_app runner_lib mlperf_loadgen Synapse pthread  ${SANITIZER_FLAGS}  -fopenmp)
else()
    target_link_libraries(HABANA_benchmark_app runner_lib mlperf_loadgen Synapse pthread  -fopenmp) 
endif()
add_custom_command(TARGET HABANA_benchmark_app  POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:HABANA_benchmark_app>" "${TARGET_PATH}/${CMAKE_BUILD_TYPE}/$<TARGET_FILE_NAME:HABANA_benchmark_app>" COMMENT "Copying to output directory"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/ini/" "${TARGET_PATH}/${CMAKE_BUILD_TYPE}/" COMMENT "Copying to output directory"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/../preprocess/" "${TARGET_PATH}/${CMAKE_BUILD_TYPE}/" COMMENT "Copying to output directory"
            COMMAND "${CMAKE_COMMAND}" -E copy "$ENV{MLPERF_MODEL_DIR}/*.recipe*" "${TARGET_PATH}/${CMAKE_BUILD_TYPE}/" COMMENT "Copying to output directory"
)


