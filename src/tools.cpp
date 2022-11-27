#include <cmath>

#include "tools.h"
#include "commons.h"

namespace tools {

double mass_from_radius(double r, double density = DENSITY) {
  return 4.0/3.0 * M_PI * r * r * r * density;
}

double inverse_volume_of_sphere(double volume) {
  return pow((3.0 * volume)/(4.0 * M_PI), 1.0/3.0);
}

size_t interaction_num(size_t n) {  // Get number of interactions for N bodies
  return n * (n - 1) / 2;
}

}
