#ifndef BODY_H
#define BODY_H

#include <cmath>

#include "tools.h"
#include "bodygpu.h"
#include "commons.h"

class Body : public BodyGPU
{
public:
  Body() {}
  Body(double x, double y, double r, double density = DENSITY) :
    BodyGPU(x, y, tools::mass_from_radius(r, density)), vx(0.0), vy(0.0), r(r), density(density) {}
  Body(double x, double y, double vx, double vy, double r, double density = DENSITY) :
    BodyGPU(x, y, tools::mass_from_radius(r, density)), vx(vx), vy(vy), r(r), density(density) {}

  void collide_with(const Body& other);

  BodyGPU * get() {
    return this;
  }
  const BodyGPU * get() const {
    return this;
  }

  double vx, vy;
  double density;
  double r;
};

#endif // BODY_H
