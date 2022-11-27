#include <algorithm>

#include "mainstate.h"
#include "body.h"

/*
// --- CUDA ---
__global__ void grav(const Body *__restrict bodies,
                     const size_t *__restrict interactions, // 2D Index map. len = 2N interaction
                     double *__restrict f_x, double *__restrict f_y, // Force from 1 -> 2. len = N interaction
                     double *__restrict dist,
                     const size_t N) {
  // Calculate global thread ID
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= N) return;

  const Body& body1 = bodies[interactions[tid]];
  const Body& body2 = bodies[interactions[tid + N]];

  // F_vec = m a_vec
  // F_vec = (GMm/r^3) * r_vec

  const double xdist = body2.x - body1.x;
  const double ydist = body2.y - body1.y;
  const double r = sqrt(xdist * xdist + ydist * ydist);
  dist[tid] = r;
  // Magnitude of f with ratio of distance
  const double f = G * body1.m * body2.m / (r * r * r);

  // Force for this interaction
  f_x[tid] = f * xdist;
  f_y[tid] = f * ydist;
}

// --------------------
*/

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

}

// Returns if any collide
bool MainState::check_coll() {
  return false;
}

void MainState::collision_pass() {

}
