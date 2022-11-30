#include <cassert>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>
#include <sstream>
#include <utility>
#include <fstream>

#include <mpi.h>

#include "commons.h"
#include "body.h"
#include "mainstate.h"
#include "vector2.h"
#include "mpistate.h"


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
  const Vector2 centre(x, y);
  Body inner(centre, inner_body_rad, inner_density);
  const double& inner_mass = inner.m;
  bodies.emplace_back(inner);

  // Make random gen
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(inner_rad, outer_rad);
  std::uniform_real_distribution<double> size_distribution(outer_body_rad_min, outer_body_rad_max);
  std::uniform_real_distribution<double> angle_distribution(0.0, M_PI * 2);

  double r, mag_v, theta, moon_radius;
  Vector2 d_centre;
  Body b;
  for (size_t n = 0; n < num; ++n) {
    r = distribution(generator);
    theta = angle_distribution(generator);
    // sqrt(GM/r) = v
    mag_v = sqrt(G * inner_mass / r);
    moon_radius = size_distribution(generator);
    d_centre.x = r * cos(theta);
    d_centre.y = r * sin(theta);

    b = Body(centre + d_centre,
             Vector2(mag_v * cos(theta + M_PI/2.0),
                     mag_v * sin(theta + M_PI/2.0)),
             moon_radius);

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


int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

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
    Body(150.0, 155.1, 25.0)
  };
  */

  // Setup
  int my_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  std::cout << my_rank << "/" << world_size << std::endl;

  if (my_rank == 0) {
    std::vector<Body> bodies;

    spawn_galaxy(bodies, 1280.0/2, 400.0,
                 20.0, 1000000.0,
                 50.0, 300.0,
                 0.25, 2.0, 800);

    MainState state(bodies, world_size);
    state.init_mpi_data();

    // Output file
    std::cout << "Opening file..." << std::endl;
    std::ofstream out_file("/tmp/jcolclou/out.txt",
                           std::ofstream::out);
    std::cout << "File opened" << std::endl;

    const double DT = 1.0/60.0;
    const size_t max_loop = 3;
    double t = 0.0;
    size_t f_num = 0;

    // Main loop
    while (f_num < max_loop) {
      std::cout << "Doing frame " << f_num << std::endl;
      state.update(DT);

      for (const auto & b : state.bodies()) {
        out_file << b << "!";
      }
      out_file << std::endl;

      ++f_num;
    }

    out_file.close();
  } else {
    std::cout << "HELLO FROM RANK " << my_rank << " :)" << std::endl;

    MPIState my_state(my_rank, world_size);
    my_state.init();
  }

  MPI_Finalize();

  return 0;
}
