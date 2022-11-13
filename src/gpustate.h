#ifndef GPUSTATE_H
#define GPUSTATE_H

#include <vector>
#include <stddef.h>
#include "body.h"
#include "bodygpu.h"

class GPUState {
public:
  GPUState(const size_t Nbodies_);
  ~GPUState();

  void copy_to_device_buffers(const std::vector<Body>& bodies);
  void fetch_result();

  void resize(const int Nbodies_);

  // Host
  size_t Ninteractions, Nbodies, interaction_bytes, body_bytes;
  std::vector<size_t> interactions;
  std::vector<double> f_x, f_y, dist; // Interactions
  std::vector<BodyGPU> body_gpu_data;
  // GPU
  BodyGPU *d_b;
  double *d_f_x, *d_f_y, *d_dist;
  size_t *d_interactions;

private:
  void alloc(const int Nbodies_);
  void free();
};


#endif // GPUSTATE_H
