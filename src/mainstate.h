#ifndef MAINSTATE_H
#define MAINSTATE_H

#include <vector>
#include <SFML/Graphics.hpp>

#include "body.h"

class MainState {
public:
  MainState(std::vector<Body> bodies);

  void update(double dt);
  void run_grav(double dt);
  bool check_coll();
  void collision_pass();

  const std::vector<Body>& bodies() const { return bodies_; }
  size_t body_num() const { return bodies_.size(); }
private:
  std::vector<Body> bodies_;
  std::vector<bool> colliding_;
  std::vector<bool> need_removing_;
};

#endif // MAINSTATE_H
