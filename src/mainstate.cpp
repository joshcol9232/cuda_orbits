#include <algorithm>
#include <mpi.h>
#include <iostream>

#include "mainstate.h"
#include "body.h"
#include "mpimsg.h"


MainState::MainState(std::vector<Body> bodies) :
  bodies_(bodies)
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

void MainState::send_body_num(int rank) const {
  int body_num = bodies_.size();
  std::cout << "DEBUG [0(hardc)] - Sending body num to " << rank << std::endl;
  MPI_Send(&body_num,
           1,
           MPI_INT,
           rank,
           static_cast<int>(MPIMsg::BodyNum),
           MPI_COMM_WORLD);
  std::cout << "DEBUG[0(hardc)] - Body num sent." << std::endl;
}
