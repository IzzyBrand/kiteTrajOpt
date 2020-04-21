import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def setup_ax(ax):
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # TODO: this should be related to the max tether length 
    ax.set_xlim(-5, 60)  
    ax.set_ylim(-5, 60)
    ax.set_zlim(-5, 60)

def plot(ax, kite):


    # get the kite state
    x = kite.state
    # position of the origin
    o = np.zeros(3)

    # draw a line from the origin to the center of the kite
    tether = np.array([o,kite.p(x)])
    print(tether.shape)
    ax.plot3D(*tether.T, 'gray')

    # draw a dot at the origin
    ax.scatter3D(*o)

def update(num, ax, kite, U, w):
    # get the command
    u = U[num]
    # dynamics update
    kite.state += 0.01 * kite.f(kite.state, u, w)
    # plot the kite
    ax.cla()
    setup_ax(ax)
    plot(ax, kite)


if __name__ == '__main__':
    theta = 1e-5
    phi = np.pi/4
    thetadot = 0
    phidot = 0

    x_0 = np.array([theta, phi, thetadot, phidot])
    w = np.array([6, 0, 0])
    u = -np.pi/8
    T = 100 # how many frames
    U = [u]*T # trajectory of controls

    from simple_kite_dynamics import Kite
    kite = Kite()
    kite.state = x_0

    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    setup_ax(ax)

    line_ani = animation.FuncAnimation(fig, update, frames=T, fargs=(ax, kite, U, w),
                                       interval=50, blit=False)

    plt.show()
