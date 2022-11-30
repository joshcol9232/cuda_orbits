#ifndef MAINSTATE_H
#define MAINSTATE_H

#include <vector>

#include "body.h"

class MainState {
public:
  MainState(std::vector<Body> bodies, int world_size);

  void update(double dt);
  void run_grav(double dt);
  bool check_coll();
  void collision_pass();

  const std::vector<Body>& bodies() const { return bodies_; }
  size_t body_num() const { return bodies_.size(); }

  // MPI
  void init_mpi_data() const;
  void send_body_num(int rank) const;   // to rank
  void send_positions(int rank) const;
  void send_masses(int rank) const;
  void recv_forces(int rank);           // from rank

  std::vector<double> serialize_positions(int rank) const;

private:
  int world_size_, body_num_per_rank_;
  std::vector<Body> bodies_;
  std::vector<bool> colliding_;
  std::vector<bool> need_removing_;
};

#endif // MAINSTATE_H
