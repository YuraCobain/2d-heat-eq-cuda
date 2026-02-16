cmake_minimum_required(VERSION 3.24)
project(heat2d_cuda_sdl LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# You can tune this
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(SDL2 REQUIRED)

add_executable(heat2d
  src/main.cu
)

target_include_directories(heat2d PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(heat2d PRIVATE ${SDL2_LIBRARIES})

# Reasonable defaults
target_compile_options(heat2d PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
     --use_fast_math
     --generate-line-info
>)

# Some compilers need this for SDL2 includes to work cleanly
if (TARGET SDL2::SDL2)
  target_link_libraries(heat2d PRIVATE SDL2::SDL2)
endif()
