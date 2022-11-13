#include "body.h"
#include "tools.h"

// Updates current body with new info
void Body::collide_with(const Body& other) {
  // Elastic collision
  // Conservation of momentum
  const double total_mass = m + other.m;
  const double total_momentum_x = m * vx + other.m * other.vx;
  const double total_momentum_y = m * vy + other.m * other.vy;
  r = tools::inverse_volume_of_sphere(total_mass/density);

  // Use centre of mass as new position
  x = (x * m + other.x * other.m)/total_mass;
  y = (y * m + other.y * other.m)/total_mass;
  vx = total_momentum_x/total_mass;
  vy = total_momentum_y/total_mass;
  m = total_mass;
}

