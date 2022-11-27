#include <algorithm>

#include "mainstate.h"
#include "body.h"

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


MainState::MainState(std::vector<Body> bodies) :
   gpu_(bodies.size()), bodies_(bodies)
{
  colliding_.assign(bodies.size(), false);
  need_removing_.assign(bodies.size(), false);
}

void MainState::update(double dt) {
  run_grav(dt);
  collision_pass();
}

void MainState::run_grav(double dt) {
  // Copy data to GPU buffers
  gpu_.copy_to_device_buffers(bodies_);

  // Threads per CTA
  size_t NUM_THREADS = 1024;

  // CTAs per Grid
  // We need to launch at LEAST as many threads as we have elements
  // This equation pads an extra CTA to the grid if N cannot evenly be divided
  // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
  size_t NUM_BLOCKS = (gpu_.Ninteractions + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU
  grav<<<NUM_BLOCKS, NUM_THREADS>>>(gpu_.d_b, gpu_.d_interactions,
                                    gpu_.d_f_x, gpu_.d_f_y,
                                    gpu_.d_dist,
                                    gpu_.Ninteractions);

  gpu_.fetch_result();

  // Apply acceleration and update positions
  size_t idx0, idx1;
  double df_x, df_y;  // Delta force in this interaction

  #pragma omp parallel for
  for (size_t i = 0; i < gpu_.Ninteractions; ++i) {
    idx0 = gpu_.interactions[i];
    idx1 = gpu_.interactions[i + gpu_.Ninteractions];
    // v = integrate f/m dt
    // x = integrate v dt
    df_x = gpu_.f_x[i] * dt;
    df_y = gpu_.f_y[i] * dt;
    // Apply acceleration
    bodies_[idx0].vx += df_x / bodies_[idx0].m;
    bodies_[idx0].vy += df_y / bodies_[idx0].m;
    bodies_[idx1].vx -= df_x / bodies_[idx1].m;  // Opposite & equal force
    bodies_[idx1].vy -= df_y / bodies_[idx1].m;
  }

  #pragma omp parallel for
  for (size_t i = 0; i < bodies_.size(); ++i) {
    bodies_[i].x += bodies_[i].vx * dt;
    bodies_[i].y += bodies_[i].vy * dt;
  }
}

// Returns if any collide
bool MainState::check_coll() {
  colliding_.assign(gpu_.Ninteractions, false);

  size_t idx0, idx1;
  bool any_colliding, col;

  #pragma omp parallel for
  for (size_t i = 0; i < gpu_.Ninteractions; ++i) {
    idx0 = gpu_.interactions[i];
    idx1 = gpu_.interactions[i + gpu_.Ninteractions];

    col = gpu_.dist[i] < bodies_[idx0].r + bodies_[idx1].r;
    colliding_[i] = col;
    if (col) any_colliding = true;
  }

  return any_colliding;
}

void MainState::collision_pass() {
  const bool any_colliding = check_coll();

  size_t idx0, idx1;

  for (size_t i = 0; i < gpu_.Ninteractions; ++i) {
    if (colliding_[i]) {
      idx0 = gpu_.interactions[i];
      idx1 = gpu_.interactions[i + gpu_.Ninteractions];

      bodies_[idx0].collide_with_no_join(bodies_[idx1]);

//      colliding_[i] = false;
    }
  }
}

//void MainState::collision_pass() {
//  const bool any_colliding = check_coll();
//  need_removing_.assign(bodies_.size(), false);

//  bool at_least_one_needs_deleting = false;
//  size_t idx0, idx1;

//  for (size_t i = 0; i < gpu_.Ninteractions; ++i) {
//    if (colliding_[i]) {
//      idx0 = gpu_.interactions[i];
//      idx1 = gpu_.interactions[i + gpu_.Ninteractions];

//      bodies_[idx0].collide_with(bodies_[idx1]);
//      // Clean the second one up (first is new bigger planet)
//      need_removing_[idx1] = true;
//      at_least_one_needs_deleting = true;
//    }
//  }

//  if (at_least_one_needs_deleting) {
//    bodies_.erase(std::remove_if(bodies_.begin(),
//                                 bodies_.end(),
//                                 [this](const Body& i) {
//                                   return need_removing_.at(&i - bodies_.data());
//                                 }),
//                                 bodies_.end());

//    gpu_.resize(bodies_.size());
//  }
//}
