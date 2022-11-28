#ifndef BODY_H
#define BODY_H

#include <cmath>
#include <ostream>

#include "tools.h"
#include "commons.h"
#include "vector2.h"

class Body
{
public:
  Body() {}
  Body(Vector2 x, double r, double density = DENSITY) :
    x(x), m(tools::mass_from_radius(r, density)), v(0.0),
    r(r), density(density) {}
  Body(Vector2 x, Vector2 v, double r, double density = DENSITY) :
    x(x), m(tools::mass_from_radius(r, density)), v(v),
    r(r), density(density) {}

  void collide_with(const Body& other);
  void collide_with_no_join(Body& other);

  Vector2 x;
  Vector2 v;
  double density;
  double m;
  double r;
};

std::ostream& operator<<(std::ostream& out, const Body& b);

#endif // BODY_H
