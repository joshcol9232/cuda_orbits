#include <iostream>

#include "gpustate.h"
#include "bodygpu.h"
#include "tools.h"

GPUState::GPUState(const size_t Nbodies_) :
  Ninteractions(tools::interaction_num(Nbodies_)), Nbodies(Nbodies_) {
  std::cout << "GPUState::GPUState starting." << std::endl;

  alloc(Nbodies);

  std::cout << "GPUState::GPUState finished." << std::endl;
}

GPUState::~GPUState() {
  std::cout << "GPUState::~GPUState starting." << std::endl;
  // Free memory on device
  free();
  std::cout << "GPUState::~GPUState finished." << std::endl;
}

void GPUState::copy_to_device_buffers(const std::vector<Body>& bodies) {
  for (size_t i = 0; i < bodies.size(); ++i) {
    body_gpu_data[i] = bodies[i];
  }
  cudaMemcpy(d_b, body_gpu_data.data(), body_bytes, cudaMemcpyHostToDevice);
}

void GPUState::fetch_result() {
  cudaMemcpy(f_x.data(), d_f_x, interaction_bytes/2, cudaMemcpyDeviceToHost);
  cudaMemcpy(f_y.data(), d_f_y, interaction_bytes/2, cudaMemcpyDeviceToHost);
  cudaMemcpy(dist.data(), d_dist, interaction_bytes/2, cudaMemcpyDeviceToHost);
}

void GPUState::resize(const int Nbodies_) {
  free();
  alloc(Nbodies_);
}

void GPUState::alloc(const int Nbodies_) {
  std::cout << "GPUState::alloc starting." << std::endl;
  Nbodies = Nbodies_;
  Ninteractions = tools::interaction_num(Nbodies);

  // Allocate local memory
  f_x = std::vector<double>(Ninteractions, 0.0);
  f_y = std::vector<double>(Ninteractions, 0.0);
  dist = std::vector<double>(Ninteractions, 0.0);

  body_gpu_data.resize(Nbodies);

  body_bytes = sizeof(BodyGPU) * Nbodies_;
  interaction_bytes = sizeof(size_t) * Ninteractions * 2;
  // Allocate memory on gpu
  cudaMalloc(&d_b, body_bytes);
  cudaMalloc(&d_f_x, interaction_bytes / 2);
  cudaMalloc(&d_f_y, interaction_bytes / 2);
  cudaMalloc(&d_dist, interaction_bytes / 2);

  // Setup interactions
  cudaMalloc(&d_interactions, interaction_bytes);

  interactions = std::vector<size_t>(Ninteractions * 2, 0);

  size_t idx = 0;
  for (size_t i = 0; i < Nbodies - 1; ++i) {
    for (size_t j = i + 1; j < Nbodies; ++j) {
      interactions[idx] = i;
      interactions[idx + Ninteractions] = j;
      ++idx;
    }
  }
  // Copy interactions to device
  cudaMemcpy(d_interactions, interactions.data(), interaction_bytes, cudaMemcpyHostToDevice);

  std::cout << "GPUState::alloc finished." << std::endl;
}

void GPUState::free() {
  std::cout << "GPUState::free starting." << std::endl;

  cudaFree(d_b);
  cudaFree(d_dist);
  cudaFree(d_f_x);
  cudaFree(d_f_y);
  cudaFree(d_interactions);
  std::cout << "GPUState::free finished." << std::endl;
}
