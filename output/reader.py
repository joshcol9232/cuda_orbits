import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sys import argv

INPUT_F = "/tmp/jcolclou/out.txt"

fr = 0

positions = []
radii = []

with open(INPUT_F, "r") as of:
  frame = of.readline()
  
  while frame:
    print("Doing frame %d..." % fr)
    pos = []  # positions for this frame
    r = []

    # body_posx, body_posy;radius;!body2_posx...
    bodies = frame.split("!")[:-1]

    for body in bodies:
      b = body.split(";")[:-1]
      pos.append(np.array(b[0].split(", "), dtype=float))
      r.append(float(b[-1]))


    positions.append(pos)
    radii.append(r)
    fr += 1

    frame = of.readline()


positions = np.array(positions)
radii = np.array(radii)

print(len(positions))

fig, ax = plt.subplots()

displayed = ax.scatter(np.zeros(len(positions[0])), np.zeros(len(positions[0])), s=np.ones(len(positions[0])), c="k")

def init():
  ax.set_xlim(200, 1000)
  ax.set_ylim(0, 800)
  return displayed,

def update(frame):
  displayed.set_offsets(positions[frame])
  displayed.set_sizes(radii[frame])
  return displayed,

interval = 10
if len(argv) > 1:
  interval = int(argv[1])

ani = FuncAnimation(fig, update, frames=range(len(positions)), init_func=init,
                    blit=True, interval=interval)

plt.gca().set_aspect("equal")
plt.show()


