#ifndef BODY_H
#define BODY_H

#include <cmath>

#include "bodygpu.h"

#define DENSITY 1000

namespace {
double mass_from_radius(double r) {
  return 4.0/3.0 * M_PI * r * r *r * DENSITY;
}
}

class Body : public BodyGPU
{
public:
  Body() {}
  Body(double x, double y, double r) :
    BodyGPU(x, y, mass_from_radius(r)), vx(0.0), vy(0.0), r(r) {}
  Body(double x, double y, double vx, double vy, double r) :
    BodyGPU(x, y, mass_from_radius(r)), vx(vx), vy(vy), r(r) {}

  BodyGPU * get() {
    return this;
  }
  const BodyGPU * get() const {
    return this;
  }

  double vx, vy;
  double r;
};

#endif // BODY_H
