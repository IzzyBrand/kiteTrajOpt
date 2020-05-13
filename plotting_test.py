import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from util import load_trajectory, calc_power
from simple_kite_dynamics import Kite
from vis import setup_ax
import sys


def set_3d_data(line, data):
    line.set_data(data.T[0:2])
    line.set_3d_properties(data.T[2])

def update(i, traj, lines):
    for line, data in zip(lines, get_data(traj, i)):
        set_3d_data(line, data)

# get a list of lines to be drawn for timestep i
def get_data(traj, i):
    p, q, qd, qdd, u, h = traj
    w = np.array([6,0,0])
    to_return = []

    # a line along the trajectory
    trace = p[:i] 
    to_return.append(trace)

    # a line from the origin to the kite
    cable = np.vstack([np.zeros(3), p[i]]) 
    to_return.append(cable)

    # the border of the kite
    x_i = np.concatenate([q[i],qd[i]])
    e_l = kite.e_l(x_i, u[i], w)
    e_t = kite.e_t(x_i, u[i], w)
    s = 2
    border = np.vstack([p[i] + 2*s*e_l,
                        p[i] + s*e_t,
                        p[i] - s*e_l,
                        p[i] - s*e_t,
                        p[i] + 2*s*e_l,])
    to_return.append(border)

    return to_return

# Attaching 3D axis to the figure
fig = plt.figure()
# ax = p3.Axes3D(fig)
ax = fig.gca(projection='3d')

# load in the trajectory and preprocess to get euclidean p
q, qd, qdd, u, h = load_trajectory(sys.argv[1])
print(calc_power(qd, u))

kite = Kite()
p = np.array([kite.p(x) for x in np.hstack([q, qd])])
traj = (p, q, qd, qdd, u, h)

T = u.shape[0]

lines = []
for data in get_data(traj, T-1):
    lines.extend(ax.plot(*data.T))


setup_ax(ax)

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update, T, fargs=(traj, lines),
                                   interval=30, blit=False)

plt.show()