#include <cassert>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>
#include <sstream>
#include <utility>

#include <omp.h>
#include <SFML/Graphics.hpp>

#include "commons.h"
#include "bodygpu.h"
#include "body.h"
#include "mainstate.h"


// ----- Spawning -----
void spawn_galaxy(std::vector<Body>& bodies, const double x,
                  const double y, const double inner_body_rad,
                  const double inner_density,
                  const double inner_rad, const double outer_rad,
                  const double outer_body_rad_min,
                  const double outer_body_rad_max,
                  const size_t num) {
  bodies.reserve(bodies.size() + num + 1);

  // Central body
  Body inner(x, y, inner_body_rad, inner_density);
  const double& inner_mass = inner.m;
  bodies.emplace_back(inner);

  // Make random gen
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(inner_rad, outer_rad);
  std::uniform_real_distribution<double> size_distribution(outer_body_rad_min, outer_body_rad_max);
  std::uniform_real_distribution<double> angle_distribution(0.0, M_PI * 2);

  double r, v, theta, moon_radius;
  Body b;
  for (size_t n = 0; n < num; ++n) {
    r = distribution(generator);
    theta = angle_distribution(generator);
    // sqrt(GM/r) = v
    v = sqrt(G * inner_mass / r);
    moon_radius = size_distribution(generator);

    b = Body(r * cos(theta) + x, r * sin(theta) + y,
             v * cos(theta + M_PI/2.0), v * sin(theta + M_PI/2.0), moon_radius);
    bodies.emplace_back(b);
  }
}

void spawn_random_uniform(std::vector<Body>& bodies,
                          const double x_min, const double x_max,
                          const double y_min, const double y_max,
                          const double r_min, const double r_max,
                          const size_t num) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> xdistr(x_min, x_max);
  std::uniform_real_distribution<double> ydistr(y_min, y_max);
  std::uniform_real_distribution<double> rdistr(r_min, r_max);

  bodies.reserve(num);

  double x, y, r;
  for (size_t n = 0; n < num; ++n) {
    x = xdistr(generator);
    y = ydistr(generator);
    r = rdistr(generator);

    bodies.push_back(Body(x, y, r));
  }
}

// ------------


int main() {
  std::cout << "MAX THREADS: " << omp_get_max_threads() << std::endl;

  std::vector<Body> bodies;

  spawn_galaxy(bodies, 1280.0/2, 400.0,
               20.0, 1000000.0,
               50.0, 300.0,
               0.25, 2.0, 800);

  /*
  spawn_random_uniform(bodies,
                       0.0, 1280.0/5,
                       0.0, 800.0/5,
                       1.0, 1.0,
                       1000);
  */

  /*
  bodies = {
    Body(100.0, 100.0, 10.0),
    Body(100.0, 200.0, 10.0),
    Body(150.0, 155.1, 25.0)
  };
  */

  // Setup
  MainState state(bodies);

  sf::Clock clock;
  sf::ContextSettings settings;
  settings.antialiasingLevel = 8;

  sf::RenderWindow window(sf::VideoMode(1280, 800), "SFML", sf::Style::Default, settings);
  window.setFramerateLimit(144);
  window.setVerticalSyncEnabled(true);

  sf::Font font;
  font.loadFromFile("/usr/share/fonts/ubuntu/Ubuntu-R.ttf");

  sf::Text fps_text;
  fps_text.setFont(font); // font is a sf::Font
  fps_text.setCharacterSize(16);
  fps_text.setFillColor(sf::Color::Green);

  sf::CircleShape body_shape(1.f);
  body_shape.setFillColor(sf::Color::White);

  sf::Time elapsed;
  double dt;

  // Main loop
  while (window.isOpen()) {
    sf::Event event;

    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        window.close();
      }
    }

    elapsed = clock.restart();
    dt = static_cast<double>(elapsed.asSeconds());

    state.update(dt);

    window.clear();
    for (const auto & b : state.bodies()) {
      body_shape.setPosition(b.x - b.r, b.y - b.r);
      body_shape.setScale(b.r, b.r);
      window.draw(body_shape);
    }

    std::stringstream os;
    os << "FPS: " << 1.0/dt << std::endl
       << "Bodies: " << state.body_num();
    fps_text.setString(os.str());
    window.draw(fps_text);
    window.display();
  }

  return 0;
}
