cmake_minimum_required(VERSION 3.15)
project(inference_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
    message(STATUS "ONNX Runtime not found via find_package, using manual paths")
    # You'll need to set ONNXRUNTIME_DIR to your ONNX Runtime installation
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
    set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif()

# Include nlohmann/json (header-only)
# Option 1: Use system installation or package manager
# Option 2: Download single header file
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/tokenizer.cpp
    src/inference_engine.cpp
    src/text_generator.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    nlohmann_json::nlohmann_json
    ${ONNXRUNTIME_LIBRARIES}
)

# Copy model files to build directory (optional, for easier testing)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models $<TARGET_FILE_DIR:${PROJECT_NAME}>/models
)
