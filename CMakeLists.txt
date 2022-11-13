cmake_minimum_required(VERSION 3.5)

project(orbits_gpu LANGUAGES CXX CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(orbits_gpu main.cu
               bodygpu.h bodygpu.cpp
               body.h
               tools.h)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
   target_link_libraries(orbits_gpu PUBLIC OpenMP::OpenMP_CXX sfml-graphics sfml-window sfml-system)
else()
    target_link_libraries(orbits_gpu sfml-graphics sfml-window sfml-system)
endif()

target_compile_features(orbits_gpu PUBLIC cxx_std_11)
set_target_properties(orbits_gpu
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET orbits_gpu
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

install(TARGETS orbits_gpu
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
