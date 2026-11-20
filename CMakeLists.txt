cmake_minimum_required(VERSION 3.24)
project(poker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(poker_core INTERFACE)
target_include_directories(poker_core INTERFACE core/include)

enable_testing()
add_executable(test_eval core/tests/test_eval.cpp)
target_link_libraries(test_eval PRIVATE poker_core)
target_compile_options(test_eval PRIVATE -Wall -Wextra -O3 -march=native)
add_test(NAME eval COMMAND test_eval)
