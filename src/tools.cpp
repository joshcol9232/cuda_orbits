#include <cmath>
#include <string_view>
#include <ostream>
#include <cstdlib>
#include <iostream>

#include "tools.h"
#include "commons.h"

#define DEBUG_PRINT(...) os << "DEBUG[" << rank << "] - " << __VA_ARGS__ << std::endl

namespace tools {

double mass_from_radius(double r, double density = DENSITY) {
  return 4.0/3.0 * M_PI * r * r * r * density;
}

double inverse_volume_of_sphere(double volume) {
  return pow((3.0 * volume)/(4.0 * M_PI), 1.0/3.0);
}

size_t interaction_num(size_t n) {  // Get number of interactions for N bodies
  return n * (n - 1) / 2;
}

void check_mpi_err(const MPI_Status& s) {
  switch (s.MPI_ERROR) {
    case MPI_SUCCESS:
      break;
    case MPI_ERR_COMM:
      std::cerr << "MPI Error: Invalid communicator." << std::endl;
      break;
    case MPI_ERR_TYPE:
      std::cerr << "MPI Error: Invalid datatype argument." << std::endl;
      break;
    case MPI_ERR_COUNT:
      std::cerr << "MPI Error: Invalid count argument." << std::endl;
      break;
    case MPI_ERR_TAG:
      std::cerr << "MPI Error: Invalid tag." << std::endl;
      break;
    case MPI_ERR_RANK:
      std::cerr << "MPI Error: Invalid source/destination rank." << std::endl;
      break;
    default:
      std::cerr << "Unknown MPI error: " << s.MPI_ERROR << std::endl;
  }

  if (s.MPI_ERROR != MPI_SUCCESS) {
    MPI_Abort(MPI_COMM_WORLD, s.MPI_ERROR);
  }
}

}
