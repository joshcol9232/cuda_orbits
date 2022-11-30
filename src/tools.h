#pragma once
#include <mpi.h>
#include <vector>
#include <stddef.h>
#include <ostream>
#include <iostream>

namespace tools {

double mass_from_radius(double r, double density);

double inverse_volume_of_sphere(double volume);

size_t interaction_num(size_t n);

void check_mpi_err(const MPI_Status& s);

template<typename T>
void print_vector(const std::vector<T>& v) {
  std::cout << "{";
  for (auto & element : v) {
    std::cout << element << ", ";
  }
  std::cout << "}" << std::endl;
}

}  // namespace tools
