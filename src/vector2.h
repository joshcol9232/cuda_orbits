#ifndef VECTOR2_H
#define VECTOR2_H

#include <ostream>

class Vector2
{
public:
    double x = 0, y = 0;
    Vector2();
    Vector2(double s);
    Vector2(double x_, double y_);
    static Vector2 fromAngle(double angle);
    double normSquared() const;
    double norm() const;
    double dot(const Vector2& rhs) const;
    double cross(const Vector2& rhs) const;  // 2D cross product - matrix det

    Vector2 operator+(const Vector2& rhs) const;
    Vector2 operator-(const Vector2& rhs) const;
    Vector2 operator*(double scalar) const;
    Vector2 operator/(double scalar) const;
    void operator+=(const Vector2& rhs);
    void operator-=(const Vector2& rhs);
    void operator*=(double scalar);
    void operator/=(double scalar);
};

std::ostream& operator<<(std::ostream& os, const Vector2& v);
Vector2 operator*=(double scalar, const Vector2& rhs);
Vector2 operator/=(double scalar, const Vector2& rhs);

#endif // VECTOR2_H
