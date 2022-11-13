#ifndef BODYGPU_H
#define BODYGPU_H

// Only data GPU needs
class BodyGPU
{
public:
  BodyGPU() {}
  BodyGPU(double x, double y, double m) :
    x(x), y(y), m(m) {}

  double x, y;
  double m;
};

#endif // BODYGPU_H
