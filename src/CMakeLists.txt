add_executable(orbits_gpu main.cpp
               commons.h
               tools.h tools.cpp
               body.h body.cpp
               vector2.h vector2.cpp
               mpistate.h mpistate.cpp
               mpimsg.h
               mainstate.h mainstate.cpp)

find_package(OpenMP)
target_link_libraries(orbits_gpu PUBLIC OpenMP::OpenMP_CXX)

find_package(MPI REQUIRED)
target_link_libraries(orbits_gpu PUBLIC MPI::MPI_CXX)

target_compile_features(orbits_gpu PUBLIC cxx_std_11)

install(TARGETS orbits_gpu
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
