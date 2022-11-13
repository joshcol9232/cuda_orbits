#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>

#include <SFML/Graphics.hpp>

#include "bodygpu.h"
#include "body.h"

#define G 0.01

size_t interaction_num(size_t n) {  // Get number of interactions for N bodies
  return n * (n - 1) / 2;
}


// TODO: Pass C++ objects
// --- CUDA ---
// TODO: Could just pass an index map, and just copy body data with no duplicates
// interactions: []
__global__ void grav(const BodyGPU *__restrict bodies,
                     const size_t *__restrict interactions, // 2D Index map. len = 2N interaction
                     double *__restrict f_x, double *__restrict f_y, // Force from 1 -> 2. len = N interaction
                     const size_t N) {
  // Calculate global thread ID
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= N) return;

  const BodyGPU& body1 = bodies[interactions[tid]];
  const BodyGPU& body2 = bodies[interactions[tid + N]];

  printf("[DEVICE %d] Body1 pos y: %f\n", tid, body1.y);
  printf("[DEVICE %d] Body2 pos y: %f\n", tid, body1.y);
  // F_vec = m a_vec
  // F_vec = (GMm/r^3) * r_vec

  const double xdist = body2.x - body1.x;
  const double ydist = body2.y - body1.y;
  const double r = sqrt(xdist * xdist + ydist * ydist);
  // Magnitude of f with ratio of distance
  const double f = G * body1.m * body2.m / (r * r * r);

  // Force for this interaction
  f_x[tid] = f * xdist;
  f_y[tid] = f * ydist;
}

struct GPUState {
  // Host
  size_t Ninteractions, Nbodies, interaction_bytes, body_bytes;
  std::vector<double> f_x, f_y; // Interaction forces
  std::vector<BodyGPU> body_gpu_data;
  // GPU
  BodyGPU *d_b;
  double *d_f_x, *d_f_y;
  size_t *d_interactions;

  GPUState(const size_t Nbodies_, std::vector<Body>& bodies) :
    Ninteractions(interaction_num(Nbodies_)), Nbodies(Nbodies_) {
    std::cout << "GPUState::GPUState starting." << std::endl;

    // Allocate local memory
    f_x = std::vector<double>(Ninteractions, 0.0);
    f_y = std::vector<double>(Ninteractions, 0.0);

    body_gpu_data.reserve(bodies.size());

    body_bytes = sizeof(BodyGPU) * Nbodies_;
    interaction_bytes = sizeof(size_t) * Ninteractions * 2;
    // Allocate memory on gpu
    cudaMalloc(&d_b, body_bytes);
    cudaMalloc(&d_f_x, interaction_bytes / 2);
    cudaMalloc(&d_f_y, interaction_bytes / 2);

    // Setup interactions
    cudaMalloc(&d_interactions, interaction_bytes);

    std::vector<size_t> interactions(Ninteractions * 2, 0);

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

    std::cout << "GPUState::GPUState finished." << std::endl;
  }

  ~GPUState() {
    std::cout << "GPUState::~GPUState starting." << std::endl;
    // Free memory on device
    cudaFree(d_b);
    cudaFree(d_f_x);
    cudaFree(d_f_y);
    cudaFree(d_interactions);
    std::cout << "GPUState::~GPUState finished." << std::endl;
  }

  void copy_to_device_buffers(const std::vector<Body>& bodies) {
    for (size_t i = 0; i < bodies.size(); ++i) {
      body_gpu_data[i] = bodies[i];
    }
    cudaMemcpy(d_b, body_gpu_data.data(), body_bytes, cudaMemcpyHostToDevice);
  }

  void fetch_result() {
    cudaMemcpy(f_x.data(), d_f_x, interaction_bytes/2, cudaMemcpyDeviceToHost);
    cudaMemcpy(f_y.data(), d_f_y, interaction_bytes/2, cudaMemcpyDeviceToHost);
  }
};

void run_grav(GPUState& gpu, std::vector<Body>& bodies, double dt) {
  // Copy data to GPU buffers
  gpu.copy_to_device_buffers(bodies);

  // Threads per CTA
  size_t NUM_THREADS = 1024;

  // CTAs per Grid
  // We need to launch at LEAST as many threads as we have elements
  // This equation pads an extra CTA to the grid if N cannot evenly be divided
  // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
  size_t NUM_BLOCKS = (gpu.Ninteractions + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU
  grav<<<NUM_BLOCKS, NUM_THREADS>>>(gpu.d_b, gpu.d_interactions,
                                    gpu.d_f_x, gpu.d_f_y,
                                    gpu.Ninteractions);

  gpu.fetch_result();

  // Apply acceleration and update positions
  size_t idx = 0;
  double df_x, df_y;  // Delta force in this interaction

  #pragma omp parallel for
  for (size_t i = 0; i < bodies.size()-1; ++i) {
    for (size_t j = i+1; j < bodies.size(); ++j) {
      // v = integrate f/m dt
      // x = integrate v dt
      df_x = gpu.f_x[idx] * dt;
      df_y = gpu.f_y[idx] * dt;
      std::cout << "f_y: " << gpu.f_y[idx] << std::endl;
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

//  spawn_galaxy(bodies, 400.0, 400.0, 10000.0,
//               110.0, 400.0, 1000);

  bodies = {
    Body(100.0, 100.0, 10.0),
    Body(100.0, 200.0, 10.0),
  };

  // Setup
  GPUState gpu(bodies.size(), bodies);

  sf::Clock clock;
  sf::RenderWindow window(sf::VideoMode(800, 800), "SFML");
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
      body_shape.setScale(b.r, b.r);
      window.draw(body_shape);
    }
    window.display();
  }

  return 0;
}
