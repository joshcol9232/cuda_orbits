#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>

#include <SFML/Graphics.hpp>

#include "body.h"

#define G 100.0

size_t interaction_num(size_t n) {  // Get number of interactions for N bodies
  return n * (n - 1) / 2;
}

// --- CUDA ---
__global__ void grav(const double *__restrict x1, const double *__restrict y1,
                     const double *__restrict x2, const double *__restrict y2,
                     const double *__restrict m1,  // masses
                     const double *__restrict m2,
                     double *__restrict f_x, double *__restrict f_y, // Force from 1 -> 2
                     const int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= N) return;

  // F_vec = m a_vec
  // F_vec = (GMm/r^3) * r_vec

  const double xdist = x2[tid] - x1[tid];
  const double ydist = y2[tid] - y1[tid];
  const double r = sqrt(xdist * xdist + ydist * ydist);
  // Magnitude of f with ratio of distance
  const double f = G * m1[tid] * m2[tid] / (r * r * r);

  // Force for this interaction
  f_x[tid] = f * xdist;
  f_y[tid] = f * ydist;
}

struct GPUState {
  int N, bytes;
  // Host
  std::vector<double> x1, y1, x2, y2, m1, m2, f_x, f_y;
  // GPU
  double *d_x1, *d_y1, *d_x2, *d_y2, *d_m1, *d_m2, *d_f_x, *d_f_y;

  GPUState(const int N_, const int bytes_) : N(N_), bytes(bytes_) {
    std::cout << "GPUState::GPUState starting." << std::endl;
    x1 = std::vector<double>(N, 0.0);
    y1 = std::vector<double>(N, 0.0);
    x2 = std::vector<double>(N, 0.0);
    y2 = std::vector<double>(N, 0.0);
    m1 = std::vector<double>(N, 0.0);
    m2 = std::vector<double>(N, 0.0);
    f_x = std::vector<double>(N, 0.0);
    f_y = std::vector<double>(N, 0.0);

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
    cudaMemcpy(d_x1, x1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, y1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, x2.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, y2.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m1, m1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2.data(), bytes, cudaMemcpyHostToDevice);
  }

  void get_result() {
    cudaMemcpy(f_x.data(), d_f_x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(f_y.data(), d_f_y, bytes, cudaMemcpyDeviceToHost);
  }
};

void run_grav(GPUState& gpu, std::vector<Body>& bodies, double dt) {
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
  double df_x, df_y;  // Delta force in this interaction

  #pragma omp parallel for
  for (size_t i = 0; i < bodies.size()-1; ++i) {
    for (size_t j = i+1; j < bodies.size(); ++j) {
      // v = integrate f/m dt
      // x = integrate v dt
      df_x = gpu.f_x[idx] * dt;
      df_y = gpu.f_y[idx] * dt;
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
    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
  }
}

// ------------

void spawn_galaxy(std::vector<Body>& bodies, const double x,
                  const double y, const double inner_mass,
                  const double inner_rad, const double outer_rad,
                  const size_t num) {
  constexpr double M = 10.0;

  bodies.reserve(bodies.size() + num + 1);

  // Central body
  bodies.emplace_back(Body(x, y, inner_mass));

  // Make random gen
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(inner_rad, outer_rad);
  std::uniform_real_distribution<double> angle_distribution(0.0, M_PI * 2);

  double r, v, theta;
  Body b;
  for (size_t n = 0; n < num; ++n) {
    r = distribution(generator);
    theta = angle_distribution(generator);
    // sqrt(GM/r) = v
    v = sqrt(G * inner_mass / r);

    b = Body(r * cos(theta) + x, r * sin(theta) + y,
             v * cos(theta + M_PI/2.0), v * sin(theta + M_PI/2.0), M);
    bodies.emplace_back(b);
  }
}

void update(GPUState& gpu, std::vector<Body>& bodies, double dt) {
//  std::cout << "FPS: " << 1.0/dt << std::endl;
  run_grav(gpu, bodies, dt);
}

int main() {
  std::vector<Body> bodies;

  spawn_galaxy(bodies, 400.0, 400.0, 10000.0,
               110.0, 300.0, 100);

  // Setup
  const int N = interaction_num(bodies.size());
  const int bytes = sizeof(double) * N;
  GPUState gpu(N, bytes);

  sf::Clock clock;
  sf::RenderWindow window(sf::VideoMode(800, 800), "SFML works!");
  sf::CircleShape body_shape(1.f);
  body_shape.setFillColor(sf::Color::White);

  // Main loop
  while (window.isOpen()) {
    sf::Event event;

    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        window.close();
      }
    }

    sf::Time elapsed = clock.restart();

    update(gpu, bodies, static_cast<double>(elapsed.asSeconds()));

    window.clear();
    for (const auto & b : bodies) {
      body_shape.setPosition(b.x, b.y);
      window.draw(body_shape);
    }
    window.display();
  }

  return 0;
}
