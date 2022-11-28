#include <cmath>
#include "vector2.h"

Vector2::Vector2(): x(0.0), y(0.0) {}

Vector2::Vector2(double s): x(s), y(s) {}

Vector2::Vector2(double x_, double y_): x(x_), y(y_) {}

Vector2 Vector2::fromAngle(double angle) {
  return Vector2(std::cos(angle), std::sin(angle));
}

double Vector2::normSquared() const {
  return x*x + y*y;
}

double Vector2::norm() const {
  return std::sqrt(normSquared());
}

double Vector2::dot(const Vector2& rhs) const {
  return x * rhs.x + y * rhs.y;
}

double Vector2::cross(const Vector2& rhs) const {
  return x * rhs.y - y * rhs.x;
}


std::ostream& operator<<(std::ostream& os, const Vector2& v) {
  os << v.x << ", " << v.y;
  return os;
}

Vector2 Vector2::operator+(const Vector2& rhs) const {
  Vector2 out;
  out.x = x + rhs.x;
  out.y = y + rhs.y;
  return out;
}

Vector2 Vector2::operator-(const Vector2& rhs) const {
  Vector2 out;
  out.x = x - rhs.x;
  out.y = y - rhs.y;
  return out;
}

Vector2 Vector2::operator*(double scalar) const {
  Vector2 out;
  out.x = x * scalar;
  out.y = y * scalar;
  return out;
}

Vector2 Vector2::operator/(double scalar) const {
  Vector2 out;
  out.x = x / scalar;
  out.y = y / scalar;
  return out;
}

void Vector2::operator+=(const Vector2& rhs) {
  x += rhs.x;
  y += rhs.y;
}

void Vector2::operator-=(const Vector2& rhs) {
  x -= rhs.x;
  y -= rhs.y;
}

void Vector2::operator*=(double scalar) {
  x *= scalar;
  y *= scalar;
}

void Vector2::operator/=(double scalar) {
  x /= scalar;
  y /= scalar;
}

Vector2 operator*=(double scalar, const Vector2& rhs) {
  return rhs * scalar;
}

Vector2 operator/=(double scalar, const Vector2& rhs) {
  return rhs / scalar;
}
