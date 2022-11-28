#ifndef MPIMSG_H
#define MPIMSG_H

// MESSAGE TAGS
enum class MPIMsg : int {
  BodyNum   = 0,
  Positions = 1,
  Masses    = 2,
};

#endif // MPIMSG_H
