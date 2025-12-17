cmake_minimum_required(VERSION 3.15)
project(Orchard LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Metal Shader Compilation
find_program(XC_RUN xcrun)

function(add_metal_shader TARGET_NAME SHADER_FILE)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    set(METALLIB_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SHADER_NAME}.metallib)
    
    add_custom_command(
        OUTPUT ${METALLIB_OUTPUT}
        COMMAND ${XC_RUN} -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_FILE} -o ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air
        COMMAND ${XC_RUN} -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air -o ${METALLIB_OUTPUT}
        DEPENDS ${SHADER_FILE}
        COMMENT "Compiling Metal shader ${SHADER_FILE}"
    )
    
    add_custom_target(${TARGET_NAME}_shader ALL DEPENDS ${METALLIB_OUTPUT})
endfunction()

# Source files
file(GLOB_RECURSE KERNEL_SOURCES src/kernels/*.metal)
file(GLOB_RECURSE PLATFORM_SOURCES src/platform/*.mm src/platform/*.cpp)
file(GLOB_RECURSE RUNTIME_SOURCES src/runtime/*.cpp)

# Create the library
add_library(orchard_core 
    ${PLATFORM_SOURCES}
    ${RUNTIME_SOURCES}
)

target_link_libraries(orchard_core "-framework Metal" "-framework Foundation")
target_include_directories(orchard_core PUBLIC src)

# Compile Metal shaders
foreach(SHADER ${KERNEL_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    add_metal_shader(orchard_${SHADER_NAME} ${SHADER})
endforeach()

# Test Executable (Phase 1)
add_executable(orchard_test src/main.cpp)
target_link_libraries(orchard_test orchard_core)
