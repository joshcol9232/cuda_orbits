#ifndef BODY_H
#define BODY_H

#include <cmath>

#include "tools.h"
#include "bodygpu.h"

class Body : public BodyGPU
{
public:
  Body() {}
  Body(double x, double y, double r) :
    BodyGPU(x, y, tools::mass_from_radius(r)), vx(0.0), vy(0.0), r(r) {}
  Body(double x, double y, double vx, double vy, double r) :
    BodyGPU(x, y, tools::mass_from_radius(r)), vx(vx), vy(vy), r(r) {}

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
