cmake_minimum_required(VERSION 3.20)
project(dmm_native_runner LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

if(NOT TORCH_PREFIX)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import torch, os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
endif()

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import torch; assert torch.__version__.split('.')[:2] == ['2', '13'], torch.__version__; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

set(TORCH_INCLUDE_DIRS
  "${TORCH_PREFIX}/include"
  "${TORCH_PREFIX}/include/torch/csrc/api/include")
set(TORCH_LIB_DIR "${TORCH_PREFIX}/lib")

if(NOT EXISTS "${TORCH_PREFIX}/include/torch/script.h")
  message(FATAL_ERROR "Torch headers were not found under ${TORCH_PREFIX}")
endif()

set(TORCH_LINK_LIBS "")
foreach(name torch torch_cpu torch_cuda c10 c10_cuda)
  find_library(lib_${name} ${name} PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH)
  if(lib_${name})
    list(APPEND TORCH_LINK_LIBS "${lib_${name}}")
  endif()
endforeach()

add_executable(dmm_native_runner
  src/main.cpp
  src/collision_table.cpp
  src/dist_table.cpp
  src/global_guide.cpp
  src/graph.cpp
  src/instance.cpp
  src/metrics.cpp
  src/pibt.cpp
  src/policy.cpp
  src/utils.cpp)

target_compile_features(dmm_native_runner PRIVATE cxx_std_20)
target_compile_options(dmm_native_runner PRIVATE -O3 -Wall)
target_compile_definitions(dmm_native_runner PRIVATE
  _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_include_directories(dmm_native_runner PRIVATE
  include
  src
  ${TORCH_INCLUDE_DIRS})
target_link_libraries(dmm_native_runner PRIVATE
  Threads::Threads
  OpenMP::OpenMP_CXX
  ${CMAKE_DL_LIBS}
  ${TORCH_LINK_LIBS})

if(UNIX AND NOT APPLE)
  target_link_options(dmm_native_runner PRIVATE -Wl,--no-as-needed)
  set_target_properties(dmm_native_runner PROPERTIES
    BUILD_RPATH "${TORCH_LIB_DIR}"
    INSTALL_RPATH "${TORCH_LIB_DIR}")
endif()

include(CTest)
add_test(NAME cli_rejects_missing_arguments COMMAND dmm_native_runner)
set_tests_properties(cli_rejects_missing_arguments PROPERTIES WILL_FAIL TRUE)
