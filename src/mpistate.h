#ifndef MPISTATE_H
#define MPISTATE_H

#include <vector>
#include "body.h"
#include "vector2.h"

class MPIState
{
public:
  MPIState(int my_rank);

  void init();
  void recv_body_num();
  void recv_positions();
  void recv_masses();
  void send_forces() const;

private:
  int my_rank_;
  // The rank owns bodies, then has to output forces.
  int body_num_;
  std::vector<Vector2> positions_;   // In
  std::vector<Vector2> masses_;      // In
  std::vector<Vector2> forces_;      // Out
};

#endif // MPISTATE_H
