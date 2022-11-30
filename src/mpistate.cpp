#include <mpi.h>
#include <iostream>
#include "tools.h"
#include "mpimsg.h"
#include "mpistate.h"

#define DEBUG_OUT std::cout << "DEBUG[" << my_rank_ << "] - "

MPIState::MPIState(int my_rank, int world_size) :
  my_rank_(my_rank), world_size_(world_size) {}

void MPIState::init() {
  recv_body_num();
  recv_positions();
  recv_masses();
}

void MPIState::recv_body_num() {
  DEBUG_OUT << "Recieving body num..." << std::endl;
  MPI_Recv(&body_num_,
           1,
           MPI_INT,
           0,
           static_cast<int>(MPIMsg::BodyNum),
           MPI_COMM_WORLD,
           &mpi_status_);
  tools::check_mpi_err(mpi_status_);
  DEBUG_OUT << "Body num recieved: " << body_num_ << std::endl;
}

void MPIState::recv_positions() {
  DEBUG_OUT << "Recieving positions..." << std::endl;
  const int expected_size = body_num_ * 2;  // x, y
  positions_ = std::vector<double>(expected_size, 0.0);

  MPI_Recv(&positions_[0],
           expected_size,
           MPI_DOUBLE,
           0,
           static_cast<int>(MPIMsg::Positions),
           MPI_COMM_WORLD,
           &mpi_status_);
  tools::check_mpi_err(mpi_status_);

  DEBUG_OUT << "Positions recieved. Num: " << positions_.size() << std::endl;
  tools::print_vector(positions_);
}

void MPIState::recv_masses() {
  DEBUG_OUT << "Recieving masses..." << std::endl;
  masses_ = std::vector<double>(body_num_, 0.0);
  MPI_Recv(&masses_[0],
           body_num_,
           MPI_DOUBLE,
           0,
           static_cast<int>(MPIMsg::Masses),
           MPI_COMM_WORLD,
           &mpi_status_);
  tools::check_mpi_err(mpi_status_);

  DEBUG_OUT << "Masses recieved. Num: " << masses_.size() << std::endl;
}

void MPIState::send_forces() const {}

