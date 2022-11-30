#include <algorithm>
#include <mpi.h>
#include <iostream>

#include "mainstate.h"
#include "body.h"
#include "mpimsg.h"

#define DEBUG_OUT_MAINSTATE std::cout << "DEBUG[0 (hardc)] - "

MainState::MainState(std::vector<Body> bodies, int world_size) :
  bodies_(bodies), world_size_(world_size),
  body_num_per_rank_(bodies.size() / world_size)
{
  colliding_.assign(bodies.size(), false);
  need_removing_.assign(bodies.size(), false);
}

void MainState::update(double dt) {
  run_grav(dt);
  collision_pass();
}

void MainState::run_grav(double dt) {
  double r, f;
  Vector2 dist_vec, f_vec;
  std::vector<Vector2> forces(tools::interaction_num(bodies_.size()), Vector2());

  size_t idx = 0;
  for (size_t i = 0; i < bodies_.size()-1; ++i) {
    for (size_t j = i+1; j < bodies_.size(); ++j) {
      dist_vec = bodies_[j].x - bodies_[i].x;
      r = dist_vec.norm();
      // Magnitude of f with ratio of distance
      f = G * bodies_[i].m * bodies_[j].m / (r * r * r);
      f_vec = dist_vec * f;

      forces[idx] += f_vec;
      ++idx;
    }
  }

  // Resolve acceleration
  idx = 0;
  for (size_t i = 0; i < bodies_.size()-1; ++i) {
    for (size_t j = i+1; j < bodies_.size(); ++j) {
      bodies_[i].v += (forces[idx] / bodies_[i].m) * dt;
      bodies_[j].v -= (forces[idx] / bodies_[j].m) * dt;
      ++idx;
    }
  }

  // Resolve change in x
#pragma omp parallel for
  for (size_t i = 0; i < bodies_.size(); ++i) {
    bodies_[i].x += bodies_[i].v * dt;
  }
}

// Returns if any collide
bool MainState::check_coll() {
  return false;
}

void MainState::collision_pass() {

}

void MainState::init_mpi_data() const {
  for (int r = 1; r < world_size_; ++r) {
    send_body_num(r);
    send_positions(r);
    send_masses(r);
  }
}

void MainState::send_body_num(int rank) const {
  DEBUG_OUT_MAINSTATE << "Sending body num to " << rank << std::endl;
  MPI_Send(&body_num_per_rank_,
           1,
           MPI_INT,
           rank,
           static_cast<int>(MPIMsg::BodyNum),
           MPI_COMM_WORLD);
  DEBUG_OUT_MAINSTATE << "Body num sent to " << rank << std::endl;
}

std::vector<double> MainState::serialize_positions(int rank) const {
  int expected_size = (bodies_.size() / world_size_) * 2;
  std::vector<double> pos(expected_size);

  const size_t i_offset = body_num_per_rank_ * rank;
  for (size_t i = 0; i < body_num_per_rank_; ++i) {
    pos.push_back(bodies_[i + i_offset].x.x);
    pos.push_back(bodies_[i + i_offset].x.y);
  }
  return pos;
}

void MainState::send_positions(int rank) const {
  DEBUG_OUT_MAINSTATE << "Sending positions to " << rank << std::endl;
  std::vector<double> pos = serialize_positions(rank);
  DEBUG_OUT_MAINSTATE << "POS SIZE: " << pos.size() * sizeof(double) << std::endl;
  MPI_Send(&pos[0],
           body_num_per_rank_ * 2,
           MPI_DOUBLE,
           rank,
           static_cast<int>(MPIMsg::Positions),
           MPI_COMM_WORLD);
  DEBUG_OUT_MAINSTATE << "Positions sent to " << rank << std::endl;
}

void MainState::send_masses(int rank) const {
  DEBUG_OUT_MAINSTATE << "Sending masses to " << rank << std::endl;

  std::vector<double> masses(body_num_per_rank_);

  const size_t i_offset = body_num_per_rank_ * rank;
  for (size_t i = 0; i < body_num_per_rank_; ++i) {
    masses.push_back(bodies_[i + i_offset].m);
  }

  MPI_Send(&masses[0],
           body_num_per_rank_,
           MPI_DOUBLE,
           rank,
           static_cast<int>(MPIMsg::Masses),
           MPI_COMM_WORLD);

  DEBUG_OUT_MAINSTATE << "Masses sent to " << rank << std::endl;
}
