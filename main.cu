#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>
#include <sstream>

#include <SFML/Graphics.hpp>

#include "bodygpu.h"
#include "body.h"
#include "tools.h"

#define G 0.005

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

class GPUState {
public:
  GPUState(const size_t Nbodies_) :
    Ninteractions(interaction_num(Nbodies_)), Nbodies(Nbodies_) {
    std::cout << "GPUState::GPUState starting." << std::endl;

    alloc(Nbodies);

    std::cout << "GPUState::GPUState finished." << std::endl;
  }

  ~GPUState() {
    std::cout << "GPUState::~GPUState starting." << std::endl;
    // Free memory on device
    free();
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

  void resize(const int Nbodies_) {
    free();
    alloc(Nbodies_);
  }

  // Host
  size_t Ninteractions, Nbodies, interaction_bytes, body_bytes;
  std::vector<size_t> interactions;
  std::vector<double> f_x, f_y; // Interaction forces
  std::vector<BodyGPU> body_gpu_data;
  // GPU
  BodyGPU *d_b;
  double *d_f_x, *d_f_y;
  size_t *d_interactions;

private:
  void alloc(const int Nbodies_) {
    std::cout << "GPUState::alloc starting." << std::endl;
    Nbodies = Nbodies_;
    Ninteractions = interaction_num(Nbodies);

    // Allocate local memory
    f_x = std::vector<double>(Ninteractions, 0.0);
    f_y = std::vector<double>(Ninteractions, 0.0);

    body_gpu_data.resize(Nbodies);

    body_bytes = sizeof(BodyGPU) * Nbodies_;
    interaction_bytes = sizeof(size_t) * Ninteractions * 2;
    // Allocate memory on gpu
    cudaMalloc(&d_b, body_bytes);
    cudaMalloc(&d_f_x, interaction_bytes / 2);
    cudaMalloc(&d_f_y, interaction_bytes / 2);

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

  void free() {
    std::cout << "GPUState::free starting." << std::endl;

    cudaFree(d_b);
    cudaFree(d_f_x);
    cudaFree(d_f_y);
    cudaFree(d_interactions);
    std::cout << "GPUState::free finished." << std::endl;
  }
};

inline void run_grav(GPUState& gpu, std::vector<Body>& bodies, double dt) {
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
  size_t idx0, idx1;
  double df_x, df_y;  // Delta force in this interaction

  #pragma omp parallel for
  for (size_t i = 0; i < gpu.Ninteractions; ++i) {
    idx0 = gpu.interactions[i];
    idx1 = gpu.interactions[i + gpu.Ninteractions];
    // v = integrate f/m dt
    // x = integrate v dt
    df_x = gpu.f_x[i] * dt;
    df_y = gpu.f_y[i] * dt;
    // Apply acceleration
    bodies[idx0].vx += df_x / bodies[idx0].m;
    bodies[idx0].vy += df_y / bodies[idx0].m;
    bodies[idx1].vx -= df_x / bodies[idx1].m;  // Opposite & equal force
    bodies[idx1].vy -= df_y / bodies[idx1].m;
  }

  #pragma omp parallel for
  for (size_t i = 0; i < bodies.size(); ++i) {
    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
  }
}

// ------------
// ----- Collisions -----

std::vector<bool> check_coll(const std::vector<size_t>& interactions,
                             std::vector<Body>& bodies) {
  const size_t Ninteractions = interactions.size()/2;
  std::vector<bool> colliding(Ninteractions, false);

  size_t idx0, idx1;
  double dist_x, dist_y, dist;
  for (size_t i = 0; i < Ninteractions; ++i) {
    idx0 = interactions[i];
    idx1 = interactions[i + Ninteractions];

    dist_x = bodies[idx1].x - bodies[idx0].x;
    dist_y = bodies[idx1].y - bodies[idx0].y;
    dist = sqrt(dist_x * dist_x + dist_y * dist_y);

    colliding[i] = dist < bodies[idx0].r + bodies[idx1].r;
  }

  return colliding;
}

void collision_pass(GPUState& gpu, std::vector<Body>& bodies) {
  const std::vector<bool> collisions = check_coll(gpu.interactions, bodies);
  std::vector<bool> need_removing(bodies.size(), false);

  bool at_least_one_needs_deleting = false;
  size_t idx0, idx1;
  for (size_t i = 0; i < gpu.Ninteractions; ++i) {
    if (collisions[i]) {
      std::cout << "COLLISION FOUND" << std::endl;
      idx0 = gpu.interactions[i];
      idx1 = gpu.interactions[i + gpu.Ninteractions];
      // TODO: Work out collision stuff

      // Clean them up
      need_removing[idx0] = true;
      need_removing[idx1] = true;
      at_least_one_needs_deleting = true;
    }
  }

  if (at_least_one_needs_deleting) {
    bodies.erase(std::remove_if(bodies.begin(),
                                bodies.end(),
                                [&need_removing, &bodies](const Body& i) {
                                  return need_removing.at(&i - bodies.data());
                                }),
                                bodies.end());

    gpu.resize(bodies.size());
  }
}

// --------------------
// ----- Spawning -----
void spawn_galaxy(std::vector<Body>& bodies, const double x,
                  const double y, const double inner_body_rad,
                  const double inner_density,
                  const double inner_rad, const double outer_rad,
                  const double outer_body_rad_min,
                  const double outer_body_rad_max,
                  const size_t num) {
  bodies.reserve(bodies.size() + num + 1);

  // Central body
  Body inner(x, y, inner_body_rad);
  inner.m = tools::mass_from_radius(inner_body_rad, inner_density);
  const double inner_mass = inner.m;
  bodies.emplace_back(inner);

  // Make random gen
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(inner_rad, outer_rad);
  std::uniform_real_distribution<double> size_distribution(outer_body_rad_min, outer_body_rad_max);
  std::uniform_real_distribution<double> angle_distribution(0.0, M_PI * 2);

  double r, v, theta, moon_radius;
  Body b;
  for (size_t n = 0; n < num; ++n) {
    r = distribution(generator);
    theta = angle_distribution(generator);
    // sqrt(GM/r) = v
    v = sqrt(G * inner_mass / r);
    moon_radius = size_distribution(generator);

    b = Body(r * cos(theta) + x, r * sin(theta) + y,
             v * cos(theta + M_PI/2.0), v * sin(theta + M_PI/2.0), moon_radius);
    bodies.emplace_back(b);
  }
}

void spawn_random_uniform(std::vector<Body>& bodies,
                          const double x_min, const double x_max,
                          const double y_min, const double y_max,
                          const double r_min, const double r_max,
                          const size_t num) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> xdistr(x_min, x_max);
  std::uniform_real_distribution<double> ydistr(y_min, y_max);
  std::uniform_real_distribution<double> rdistr(r_min, r_max);

  bodies.reserve(num);

  double x, y, r;
  for (size_t n = 0; n < num; ++n) {
    x = xdistr(generator);
    y = ydistr(generator);
    r = rdistr(generator);

    bodies.push_back(Body(x, y, r));
  }
}

// ------------

inline void update(GPUState& gpu, std::vector<Body>& bodies, double dt) {
  run_grav(gpu, bodies, dt);
  collision_pass(gpu, bodies);
}

int main() {
  std::vector<Body> bodies;


  spawn_galaxy(bodies, 1280.0/2, 400.0,
               20.0, 1000000.0,
               50.0, 180.0,
               0.25, 2.0, 400);

  /*
  spawn_random_uniform(bodies,
                       0.0, 1280.0/5,
                       0.0, 800.0/5,
                       1.0, 1.0,
                       1000);
  */

  /*
  bodies = {
    Body(100.0, 100.0, 10.0),
    Body(100.0, 200.0, 10.0),
  };
  */

  // Setup
  GPUState gpu(bodies.size());

  sf::Clock clock;
  sf::ContextSettings settings;
  settings.antialiasingLevel = 8;

  sf::RenderWindow window(sf::VideoMode(1280, 800), "SFML", sf::Style::Default, settings);

  sf::Font font;
  font.loadFromFile("/usr/share/fonts/ubuntu/Ubuntu-R.ttf");

  sf::Text fps_text;
  fps_text.setFont(font); // font is a sf::Font
  fps_text.setCharacterSize(16);
  fps_text.setFillColor(sf::Color::Green);

  sf::CircleShape body_shape(1.f);
  body_shape.setFillColor(sf::Color::White);

  sf::Time elapsed;
  double dt;

  // Main loop
  while (window.isOpen()) {
    sf::Event event;

    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        window.close();
      }
    }

    elapsed = clock.restart();
    dt = static_cast<double>(elapsed.asSeconds());

    update(gpu, bodies, dt);

    window.clear();
    for (const auto & b : bodies) {
      body_shape.setPosition(b.x - b.r, b.y - b.r);
      body_shape.setScale(b.r, b.r);
      window.draw(body_shape);
    }

    std::stringstream os;
    os << "FPS: " << 1.0/dt << std::endl
       << "Bodies: " << bodies.size();
    fps_text.setString(os.str());
    window.draw(fps_text);
    window.display();
  }

  return 0;
}
