#ifndef MPISTATE_H
#define MPISTATE_H

#include <vector>
#include "body.h"
#include "vector2.h"

class MPIState
{
public:
  MPIState(int my_rank, int world_size);

  void init();
  void recv_body_num();
  void recv_positions();
  void recv_masses();
  void send_forces() const;

private:
  // MPI
  int my_rank_, world_size_;
  MPI_Status mpi_status_;

  // The rank owns bodies, then has to output forces.
  int body_num_;
  std::vector<double> positions_;   // In
  std::vector<double> masses_;      // In
  std::vector<double> forces_;      // Out
};

#endif // MPISTATE_H
