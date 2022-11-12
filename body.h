#ifndef BODY_H
#define BODY_H


class Body
{
public:
  Body(float x, float y, float m = 1.0) :
    x(x), y(y), m(m) {}
  Body(float x, float y, float vx, float vy, float m = 1.0) :
    x(x), y(y), vx(vx), vy(vy), m(m) {}

  float x, y;
  float vx, vy;
  float m;
};

#endif // BODY_H
