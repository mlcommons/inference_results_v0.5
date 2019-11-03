set(PROJECT_NAME "${RUNNER_LIB_NAME}")
set(PRIVATE_INCLUDES "include/") # include/${RUNNER_LIB_NAME}/

find_package(Threads REQUIRED)

add_library("${PROJECT_NAME}" 
                src/runner_lib.cpp
)

target_compile_features("${PROJECT_NAME}" PUBLIC cxx_std_14)
target_include_directories("${PROJECT_NAME}" INTERFACE "include/" )
target_link_libraries ("${PROJECT_NAME}"  PRIVATE pthread nnpi_inference )
