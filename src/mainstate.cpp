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
  double xdist, ydist, r, f, fx, fy;
  for (size_t i = 0; i < bodies_.size()-1; ++i) {
    for (size_t j = i+1; j < bodies_.size(); ++j) {
      xdist = bodies_[j].x - bodies_[i].x;
      ydist = bodies_[j].y - bodies_[i].y;
      r = sqrt(xdist * xdist + ydist * ydist);
      // Magnitude of f with ratio of distance
      f = G * bodies_[i].m * bodies_[j].m / (r * r * r);
      fx = f * xdist;
      fy = f * ydist;

      bodies_[i].r_fx += fx;
      bodies_[i].r_fy += fy;
      bodies_[j].r_fx -= fx;
      bodies_[j].r_fy -= fy;
    }
  }

  for (size_t i = 0; i < bodies_.size(); ++i) {
    bodies_[i].vx += (bodies_[i].r_fx / bodies_[i].m) * dt;
    bodies_[i].vy += (bodies_[i].r_fy / bodies_[i].m) * dt;
    bodies_[i].x += bodies_[i].vx * dt;
    bodies_[i].y += bodies_[i].vy * dt;
    bodies_[i].r_fx = 0.0;
    bodies_[i].r_fy = 0.0;
  }
}

// Returns if any collide
bool MainState::check_coll() {
  return false;
}

void MainState::collision_pass() {

}
