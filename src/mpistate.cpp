#include <mpi.h>
#include <iostream>
#include "mpimsg.h"
#include "mpistate.h"

MPIState::MPIState(int my_rank) : my_rank_(my_rank) {}

void MPIState::init() {
  recv_body_num();
  recv_positions();
  recv_masses();
}

void MPIState::recv_body_num() {
  std::cout << "DEBUG[" << my_rank_ << "] - Recieving body num." << std::endl;
  MPI_Recv(&body_num_,
           1,
           MPI_INT,
           0,
           static_cast<int>(MPIMsg::BodyNum),
           MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  std::cout << "DEBUG[" << my_rank_ << "] - Body num recieved: " << body_num_ << std::endl;
}

void MPIState::recv_positions() {
//  MPI_Recv();
}

void MPIState::recv_masses() {

}

void MPIState::send_forces() const {}

