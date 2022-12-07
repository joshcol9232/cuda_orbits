#include <iostream>
#include <fstream>
#include <vector>

#include <mpi.h>

#include "vector2.h"
#include "tools.h"


#define DEBUG_OUT_MAINSTATE std::cout << "DEBUG[0 (hardc)] - "
#define DEBUG_OUT std::cout << "DEBUG[" << my_rank << "] - "

constexpr double DT = 1.0/60.0;
constexpr size_t MAX_FRAME = 100;

/* IDEAS:
 * - Body mapping can be done using min() - find lowest index. Then arrange bodies in body list
 *   sent to rank in that order. Easily maps between rank & main rank
 *
 *
 */

void write_frame_to_file(std::ofstream& f,
                         const std::vector<Vector2>& x,
                         const std::vector<double>& r) {

  for (size_t i = 0; i < x.size(); ++i) {
    f << x[i] << ";" << r[i] << ";!";
  }
  f << std::endl;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // Setup
  int my_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<Vector2> x;
  std::vector<Vector2> v;
  std::vector<double> m;
  std::vector<double> r;

  std::vector<size_t> my_interactions;

  std::ofstream out_file;

  if (my_rank == 0) {
    DEBUG_OUT_MAINSTATE << "My rank is 0 :)" << std::endl;

    DEBUG_OUT_MAINSTATE << "Opening file..." << std::endl;
    out_file = std::ofstream("/tmp/jcolclou/out.txt",
                             std::ofstream::out);
    DEBUG_OUT_MAINSTATE << "File opened" << std::endl;


    // Make bodies
    x = { Vector2(0.0, 0.0), Vector2(100.0, 0.01), Vector2(50.0, 50.0), Vector2(150.0, 150.0) };
    v = { Vector2(0.0, 1.0), Vector2(10.0, 0.1), Vector2(), Vector2(-100.0, -100.0) };
    m = { 1000.0, 1000.0, 1000.0, 1500.0 };
    r = { 10.0, 10.0, 10.0, 15.0 };

    // Partition interactions
    const size_t interaction_num = tools::interaction_num(x.size());

    const size_t partition_size = interaction_num / world_size;

    std::vector<std::vector<size_t>> interactions(world_size,
                                                  std::vector<size_t>(partition_size * 2, 0));

    std::vector<std::vector<size_t>> mappings(world_size,
                                              std::vector<size_t>(partition_size, 0));

    size_t idx = 0;
    int destination_rank = -1;
    for (size_t i = 0; i < x.size()-1; ++i) {
      for (size_t j = i + 1; j < x.size(); ++j) {
        if ((idx/2) % partition_size == 0) {
          ++destination_rank;
        }
        interactions[destination_rank][idx % (2 * partition_size)] = i;
        interactions[destination_rank][idx % (2 * partition_size) + 1] = j;

        idx += 2;
      }
    }

    size_t i = 0;
    for (auto & inter : interactions) {
      DEBUG_OUT_MAINSTATE << "Rank " << i << " interactions: ";
      tools::print_vector(inter);
      ++i;
    }

    my_interactions = interactions[0];

    write_frame_to_file(out_file, x, r);
  }


  // Do some things on all ranks
  DEBUG_OUT << "Hi from this rank" << std::endl;

  if (my_rank == 0) {
    out_file.close();
  }

  return 0;
}
