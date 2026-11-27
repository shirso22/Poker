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

add_executable(test_engine core/tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE poker_core)
target_compile_options(test_engine PRIVATE -Wall -Wextra -O2)
add_test(NAME engine COMMAND test_engine)

add_executable(test_range core/tests/test_range.cpp)
target_link_libraries(test_range PRIVATE poker_core)
target_compile_options(test_range PRIVATE -Wall -Wextra -O3 -march=native)
add_test(NAME range COMMAND test_range)
