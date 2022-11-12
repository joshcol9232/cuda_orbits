#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "body.h"


size_t interaction_num(size_t n) {
  return n * (n - 1) / 2;
}

// --- CUDA ---
// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void grav(const double *__restrict x1, const double *__restrict y1,
                     const double *__restrict x2, const double *__restrict y2,
                     const double *__restrict m1,  // masses
                     const double *__restrict m2,
                     double *__restrict a_x, double *__restrict a_y, // Acceleration from 1 -> 2
                     const int N) {
  constexpr double G = 100.0;

  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= N) return;

  // F_vec = m a_vec
  // F_vec = (GMm/r^3) * r_vec
  // a_vec = (GM/r^3) * r_vec for m1 = m

  const double xdist = x2[tid] - x1[tid];
  const double ydist = y2[tid] - y1[tid];
  const double r = sqrt(xdist * xdist + ydist * ydist);
  // Magnitude of a/distance
  const double a = G * m2[tid] / (r * r * r);

  a_x[tid] = a * xdist;
  a_y[tid] = a * ydist;
}

struct GPUState {
  int N, bytes;
  // Host
  std::vector<double> x1, y1, x2, y2, m1, m2, a_x, a_y;
  // GPU
  double *d_x1, *d_y1, *d_x2, *d_y2, *d_m1, *d_m2, *d_a_x, *d_a_y;

  GPUState(const int N_, const int bytes_) : N(N_), bytes(bytes_) {
    std::cout << "GPUState::GPUState starting." << std::endl;
    x1 = std::vector<double>(N, 0.0);
    y1 = std::vector<double>(N, 0.0);
    x2 = std::vector<double>(N, 0.0);
    y2 = std::vector<double>(N, 0.0);
    m1 = std::vector<double>(N, 0.0);
    m2 = std::vector<double>(N, 0.0);
    a_x = std::vector<double>(N, 0.0);
    a_y = std::vector<double>(N, 0.0);

    // Allocate memory on gpu
    cudaMalloc(&d_x1, bytes);
    cudaMalloc(&d_y1, bytes);
    cudaMalloc(&d_x2, bytes);
    cudaMalloc(&d_y2, bytes);
    cudaMalloc(&d_m1, bytes);
    cudaMalloc(&d_m2, bytes);
    cudaMalloc(&d_a_x, bytes);
    cudaMalloc(&d_a_y, bytes);
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
    cudaFree(d_a_x);
    cudaFree(d_a_y);
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
    cudaMemcpy(a_x.data(), d_a_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(a_y.data(), d_a_y, bytes, cudaMemcpyHostToDevice);
    std::cout << "GPUState::get_result finished." << std::endl;
  }
};

void run_grav(GPUState gpu, std::vector<Body> bodies) {
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
  int NUM_THREADS = 256;

  // CTAs per Grid
  // We need to launch at LEAST as many threads as we have elements
  // This equation pads an extra CTA to the grid if N cannot evenly be divided
  // by NUM_THREADS (e.g. N = 257, NUM_THREADS = 256)
  int NUM_BLOCKS = (gpu.N + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU
  grav<<<NUM_BLOCKS, NUM_THREADS>>>(gpu.d_x1, gpu.d_y1, gpu.d_x2, gpu.d_y2,
                                    gpu.d_m1, gpu.d_m2, gpu.d_a_x, gpu.d_a_y,
                                    gpu.N);

  gpu.get_result();


}

// ------------

int main() {
  std::vector<Body> bodies = {
    Body(0.0, 0.0),
    Body(1.0, 0.0),
  };

  const int N = interaction_num(bodies.size());
  const int bytes = sizeof(double) * N;
  GPUState gpu(N, bytes);

  run_grav(gpu, bodies);

  std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

  return 0;
}
