#include "body.h"
#include "tools.h"

// Updates current body with new info
void Body::collide_with(const Body& other) {
  // Elastic collision
  // Conservation of momentum
  const double total_mass = m + other.m;
  const Vector2 total_momentum = v * m + other.v * m;
  r = tools::inverse_volume_of_sphere(total_mass/density);

  // Use centre of mass as new position
  x = (x * m + other.x * other.m)/total_mass;
  v = total_momentum/total_mass;
  m = total_mass;
}

void Body::collide_with_no_join(Body& other) {
  // Elastic collision


}

std::ostream& operator<<(std::ostream& out, const Body& b) {
  out << b.x << ";" << b.r << ";";

  return out;
}
