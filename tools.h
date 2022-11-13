#ifndef TOOLS_H
#define TOOLS_H

#include <cmath>

#define DENSITY 1000

namespace tools {

double mass_from_radius(double r, double density = DENSITY) {
  return 4.0/3.0 * M_PI * r * r *r * DENSITY;
}

}  // namespace tools

#endif // TOOLS_H
