import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from util import load_trajectory, calc_power
from simple_kite_dynamics import Kite

def setup(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([0, 2*plot_radius])

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    # ax.grid(False)


    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')


    ax.set_title('Kite trajectory')

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
q, qd, qdd, u, h = load_trajectory('opt_120.npy')
print(calc_power(qd, u))

kite = Kite()
p = np.array([kite.p(x) for x in np.hstack([q, qd])])
traj = (p, q, qd, qdd, u, h)

T = u.shape[0]

lines = []
for data in get_data(traj, T-1):
    lines.extend(ax.plot(*data.T))


setup(ax)

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update, T, fargs=(traj, lines),
                                   interval=50, blit=False)

plt.show()