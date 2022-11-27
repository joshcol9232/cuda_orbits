#ifndef BODY_H
#define BODY_H

#include <cmath>

#include "tools.h"
#include "commons.h"

class Body
{
public:
  Body() {}
  Body(double x, double y, double r, double density = DENSITY) :
    x(x), y(y), m(tools::mass_from_radius(r, density)), vx(0.0), vy(0.0), r(r), density(density) {}
  Body(double x, double y, double vx, double vy, double r, double density = DENSITY) :
    x(x), y(y), m(tools::mass_from_radius(r, density)), vx(vx), vy(vy), r(r), density(density) {}

  void collide_with(const Body& other);
  void collide_with_no_join(Body& other);

  double x, y;
  double vx, vy;
  double density;
  double m;
  double r;
};

#endif // BODY_H
