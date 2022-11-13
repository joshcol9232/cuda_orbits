#include <algorithm>
#include <cassert>
#include <iostream>
#include<stdio.h>
#include<stdlib.h>
#include <vector>
#include <unistd.h>

#include "body.h"

#define DT 0.016666666666666666

size_t interaction_num(size_t n) {
  return n * (n - 1) / 2;
}

// --- CUDA ---
__global__ void grav(const float *__restrict x1, const float *__restrict y1,
                     const float *__restrict x2, const float *__restrict y2,
                     const float *__restrict m1,  // masses
                     const float *__restrict m2,
                     float *__restrict f_x, float *__restrict f_y, // Force from 1 -> 2
                     const int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= N) return;

  // F_vec = m a_vec
  // F_vec = (GMm/r^3) * r_vec

  const float xdist = x2[tid] - x1[tid];
  const float ydist = y2[tid] - y1[tid];
  const float r = sqrt(xdist * xdist + ydist * ydist);
  // Magnitude of f with ratio of distance
  const float f = 100.0 * m1[tid] * m2[tid] / (r * r * r);

  // Force for this interaction
  f_x[tid] = f * xdist;
  f_y[tid] = f * ydist;
}

struct GPUState {
  int N, bytes;
  // Host
  std::vector<float> x1, y1, x2, y2, m1, m2, f_x, f_y;
  // GPU
  float *d_x1, *d_y1, *d_x2, *d_y2, *d_m1, *d_m2, *d_f_x, *d_f_y;

  GPUState(const int N_, const int bytes_) : N(N_), bytes(bytes_) {
    std::cout << "GPUState::GPUState starting." << std::endl;
    x1 = std::vector<float>(N, 0.0);
    y1 = std::vector<float>(N, 0.0);
    x2 = std::vector<float>(N, 0.0);
    y2 = std::vector<float>(N, 0.0);
    m1 = std::vector<float>(N, 0.0);
    m2 = std::vector<float>(N, 0.0);
    f_x = std::vector<float>(N, 0.0);
    f_y = std::vector<float>(N, 0.0);

    // Allocate memory on gpu
    cudaMalloc(&d_x1, bytes);
    cudaMalloc(&d_y1, bytes);
    cudaMalloc(&d_x2, bytes);
    cudaMalloc(&d_y2, bytes);
    cudaMalloc(&d_m1, bytes);
    cudaMalloc(&d_m2, bytes);
    cudaMalloc(&d_f_x, bytes);
    cudaMalloc(&d_f_y, bytes);
    std::cout << "GPUState::GPUState finished." << std::endl;
  }

  ~GPUState() {
    std::cout << "GPUState::~GPUState starting." << std::endl;
    // Free memory on device
    cudaFree(d_x1);
    cudaFree(d_y1);
    cudaFree(d_x2);
    cudaFree(d_y2);
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_f_x);
    cudaFree(d_f_y);
    std::cout << "GPUState::~GPUState finished." << std::endl;
  }

  void copy_to_device_buffers() {
    std::cout << "GPUState::copy_to_device_buffers starting." << std::endl;
    cudaMemcpy(d_x1, x1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, y1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, x2.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, y2.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m1, m1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2.data(), bytes, cudaMemcpyHostToDevice);
    std::cout << "GPUState::copy_to_device_buffers finished." << std::endl;
  }

  void get_result() {
    std::cout << "GPUState::get_result starting." << std::endl;
    cudaMemcpy(f_x.data(), d_f_x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(f_y.data(), d_f_y, bytes, cudaMemcpyDeviceToHost);
    std::cout << "GPUState::get_result finished." << std::endl;
  }
};

void run_grav(GPUState& gpu, std::vector<Body>& bodies) {
  std::cout << "Running grav..." << std::endl;

  size_t idx = 0;
  for (size_t i = 0; i < bodies.size()-1; ++i) {
    for (size_t j = i+1; j < bodies.size(); ++j) {
      gpu.x1[idx] = bodies[i].x;
      gpu.y1[idx] = bodies[i].y;
      gpu.x2[idx] = bodies[j].x;
      gpu.y2[idx] = bodies[j].y;
      gpu.m1[idx] = bodies[i].m;
      gpu.m2[idx] = bodies[j].m;
      ++idx;
    }
  }

  // Copy data to GPU buffers
  gpu.copy_to_device_buffers();

  // Threads per CTA
  int NUM_THREADS = 1024;

  // CTAs per Grid
  // We need to launch at LEAST as many threads as we have elements
  // This equation pads an extra CTA to the grid if N cannot evenly be divided
  // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
  int NUM_BLOCKS = (gpu.N + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU
  grav<<<NUM_BLOCKS, NUM_THREADS>>>(gpu.d_x1, gpu.d_y1, gpu.d_x2, gpu.d_y2,
                                    gpu.d_m1, gpu.d_m2, gpu.d_f_x, gpu.d_f_y,
                                    gpu.N);

  gpu.get_result();

  // Apply acceleration and update positions
  idx = 0;
  float df_x, df_y;  // Delta force in this interaction

  #pragma omp parallel for
  for (size_t i = 0; i < bodies.size()-1; ++i) {
    for (size_t j = i+1; j < bodies.size(); ++j) {
      // v = integrate f/m dt
      // x = integrate v dt
      df_x = gpu.f_x[idx] * DT;
      df_y = gpu.f_y[idx] * DT;
      // Apply acceleration
      bodies[i].vx += df_x / bodies[i].m;
      bodies[i].vy += df_y / bodies[i].m;
      bodies[j].vx -= df_x / bodies[j].m; // Opposite & equal force
      bodies[j].vy -= df_y / bodies[j].m;

      ++idx;
    }
  }

  #pragma omp parallel for
  for (size_t i = 0; i < bodies.size(); ++i) {
    bodies[i].x += bodies[i].vx * DT;
    bodies[i].y += bodies[i].vy * DT;
  }

  std::cout << "New positions:\n"
            << "Body 0:\t" << bodies[0].x << ", " << bodies[0].y << std::endl
            << "Body 1:\t" << bodies[1].x << ", " << bodies[1].y << std::endl
            << "Body 2:\t" << bodies[2].x << ", " << bodies[2].y << std::endl;
}

// ------------

int main() {
  std::vector<Body> bodies = {
    Body(0.01, 0.01),
    Body(50.0, 50.0),
    Body(100.0, 0.1),
  };

  const int N = interaction_num(bodies.size());
  const int bytes = sizeof(float) * N;
  GPUState gpu(N, bytes);

  for (size_t i = 0; i < 5; ++i) {
    std::cout << "i: " << i << std::endl;
    run_grav(gpu, bodies);
    sleep(0.1);
  }

  std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

  return 0;
}
