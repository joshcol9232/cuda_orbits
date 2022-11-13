#ifndef BODY_H
#define BODY_H


class Body
{
public:
  Body(double x, double y, double m = 1.0) :
    x(x), y(y), vx(0.0), vy(0.0), m(m) {}
  Body(double x, double y, double vx, double vy, double m = 1.0) :
    x(x), y(y), vx(vx), vy(vy), m(m) {}

  double x, y;
  double vx, vy;
  double m;
};

#endif // BODY_H
