import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from simple_kite_dynamics import Kite
# TODO: this shouldn't be in the global scope. wrap in a class
kite = Kite()

def setup_ax(ax):
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # TODO: set aspect ratio
    # TODO: this should be related to the max tether length
    ax.set_xlim(-5, 60)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-5, 60)

def plot(ax, x, u, w):
    # get the kite state
    # position of the origin
    o = np.zeros(3)

    # draw a line from the origin to the center of the kite
    tether = np.array([o,kite.p(x)])
    ax.plot3D(*tether.T, 'gray', alpha=0.5)

    # draw a line at the kite center  in the direction of e_t
    e_t = np.array([o, 10 * kite.e_t(x, u, w)]) + kite.p(x)
    ax.plot3D(*e_t.T, label='e_t')

    # draw a line at the kite center  in the direction of e_t
    e_l = np.array([o, 10 * kite.e_l(x, u, w)]) + kite.p(x)
    ax.plot3D(*e_l.T, label='e_l')

    # plot the axes
    for v, c in zip(kite.get_string_basis(x), ['red', 'green', 'blue']):
        ax.plot3D(*np.array([o,20 * v]).T, color=c)

    # draw a dot at the origin
    ax.scatter3D(*o)

def update(t, ax, X, U, W):
    # get the command
    x = X[t]
    w = W[t]
    u = U[t]
    # plot the kite
    ax.cla()
    setup_ax(ax)
    plot(ax, x, u, w)
    ax.legend()

    dt = 0.05 # TODO: Make this a parameter?
    ax.set_title('t = {:8.4f}"'.format(t*dt))

def animate_trajectory(X, U, W, save_file=None):
    T = X.shape[0]

    assert U.shape[0] == T
    assert W.shape[0] == T

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    setup_ax(ax)

    line_ani = animation.FuncAnimation(fig, update, frames=T, fargs=(ax, X, U, W),
                                       interval=33, blit=False)

    if save_file is None:
        plt.show()
    else:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        line_ani.save(save_file, writer=writer)

def plot_3d_trajectory(q, qd, u=None, ax=None, show=True):
    kite = Kite()
    p = np.array([kite.p(x) for x in np.hstack([q, qd])])

    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

    # we need to use a scatterplot to have color
    if u is not None:
        ts = np.arange(p.shape[0]) % u.shape[0]
        power = qd[:,2] * u[ts,1]
        power = power / np.abs(power).max()/2 + 0.5
        color = plt.cm.jet(power)
        ax.scatter(*p.T, color=color)
    else:
        plt.plot(*p.T)

    if show:
        plt.show()
    if not show:
        return ax

def main():
    T = 100 # how many frames
    dt = 0.05

    # trajectectory of wind vectors
    W = np.ones([T,3]) * np.array([6, 0, 0])[None,:]
    # trajectory of control inputs
    U = np.ones(T) * -np.pi/8


    theta = np.pi/4
    phi = np.pi/4
    thetadot = 0
    phidot = 0
    x_0 = np.array([theta, phi, thetadot, phidot])

    # simulate the kite in time
    X = [x_0]
    for t in range(T-1):
        dx = kite.f(X[t], U[t], W[t])
        X.append(X[t] + dx*dt)
    X = np.array(X)

    animate_trajectory(X, U, W)

if __name__ == '__main__':
    main()
