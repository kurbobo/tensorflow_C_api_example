﻿cmake_minimum_required(VERSION 3.6)

project(hello_tf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(tensorflow_SYSTEM_LIBRARIES "Use system libraries" ON)
if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_VERBOSE_MAKEFILE ON)
    add_compile_options(-Wall -Wextra -pedantic-errors -Werror)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
        add_compile_options(/permissive-)
    endif()
endif()

# set the path to the library folder
#link_directories(/usr/local/lib)

# link the libraries to the executable
#target_link_libraries (tensorflow load_graph)

if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
    make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
endif()

if(UNIX AND NOT APPLE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/libtensorflow.so)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.14.0/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz
        STATUS
            status
        LOG
            log
        )
        list(GET status 0 status_code)
        list(GET status 1 status_string)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error downloading tensorflow lib: ${status_string}" "${log}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz)
    endif()

    include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include/tensorflow/c)
    link_directories(${CMAKE_SOURCE_DIR}/tensorflow/lib)
elseif(APPLE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/libtensorflow.so)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.14.0/libtensorflow-cpu-darwin-x86_64-1.14.0.tar.gz ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz
        STATUS
            status
        LOG
            log
        )
        list(GET status 0 status_code)
        list(GET status 1 status_string)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error downloading tensorflow lib: ${status_string}" "${log}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz)
    endif()

    include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include/tensorflow/c)
    link_directories(tensorflow/lib)
elseif(WIN32)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/tensorflow.dll)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.14.0/libtensorflow-cpu-windows-x86_64-1.14.0.zip ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip
        STATUS
            status
        LOG
            log
        )
        list(GET status 0 status_code)
        list(GET status 1 status_string)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error downloading tensorflow lib: ${status_string}" "${log}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip)
    endif()

    include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include/tensorflow/c)
    link_directories(tensorflow/lib)
    configure_file(tensorflow/lib/tensorflow.dll ${CMAKE_CURRENT_BINARY_DIR}/tensorflow.dll COPYONLY)
    configure_file(tensorflow/lib/tensorflow.dll ${CMAKE_CURRENT_BINARY_DIR}/test/tensorflow.dll COPYONLY)
endif()

include_directories(src/3rdparty/scope_guard)

add_executable(hello_tf src/hello_tf.cpp)
target_link_libraries(hello_tf tensorflow)

add_executable(session_run src/session_run.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(session_run tensorflow)

add_executable(load_graph src/load_graph.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(load_graph tensorflow)

add_executable(batch_interface src/batch_interface.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(batch_interface tensorflow)

configure_file(models/saved_model.pb ${CMAKE_CURRENT_BINARY_DIR}/saved_model.pb COPYONLY)

enable_testing()
add_subdirectory(test)
