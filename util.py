import numpy as np
from simple_kite_dynamics import Kite
from vis import plot_3d_trajectory

def create_mirrored_loop(q, qd, qdd, u):
    """ Create a mirrored (phi and roll) copy of the trajecotory and append
    """
    q_mirrored = q * np.array([[1,-1,1]])
    qd_mirrored = qd * np.array([[1,-1,1]])
    qdd_mirrored = qdd * np.array([[1,-1,1]])
    u_mirrored = u * np.array([[-1, 1]])

    q_full = np.vstack([q[:-1], q_mirrored])
    qd_full = np.vstack([qd[:-1], qd_mirrored])
    qdd_full = np.vstack([qdd, qdd_mirrored])
    u_full = np.vstack([u, u_mirrored])
    return q_full, qd_full, qdd_full, u_full

def get_circle_guess_trajectory(T):
    """ Generate an intital trajectory guess (a circle in theta and phi)
    of T timesteps. This only half a trajectory (one circle instead of two)
    because we are optimizing a symmetric trajectory
    """
    t = np.linspace(0,np.pi*2,T+1)
    s = 20
    n = 0.1
    q_guess = np.random.randn(T+1,3)*n
    qd_guess = np.random.randn(T+1,3)*n
    qdd_guess = np.random.randn(T,3)*n
    u_guess = np.random.randn(T,2)*n
    q_guess[:,0] += s*np.sin(t) + 45
    q_guess[:,1] += s*np.cos(t) - s
    q_guess[:,2] += 40
    qd_guess[:,0] += s*np.cos(t)
    qd_guess[:,1] += -s*np.sin(t)
    qd_guess[:,2] += 0
    qdd_guess[:,0] += -s*np.sin(t)[:-1]
    qdd_guess[:,1] += -s*np.cos(t)[:-1]
    qdd_guess[:,2] += 0

    q_guess[:,:2] = np.radians(q_guess[:,:2])
    qd_guess[:,:2] = np.radians(qd_guess[:,:2])
    qdd_guess[:,:2] = np.radians(qdd_guess[:,:2])

    return q_guess, qd_guess, qdd_guess, u_guess

def get_lemniscate_guess_trajectory(T, num_loops=1):
    t = np.linspace(0,np.pi*2*num_loops,T+1) + np.pi/2
    s = np.radians(40)
    n = 0.001
    q_guess = np.random.randn(T+1,3)*n
    qd_guess = np.random.randn(T+1,3)*n
    qdd_guess = np.random.randn(T+1,3)*n
    u_guess = np.random.randn(T,2)*n
    # phi
    q_guess[:,1] += s*np.cos(t)/(1+np.sin(t)**2)
    qd_guess[:,1] += -s*(9*np.sin(t) + np.sin(3*t))/(-3 + np.cos(2*t))**2
    qdd_guess[:,1] += s*(2*np.cos(t) + 45*np.cos(3*t) + np.cos(5*t))/(2*(-3 + np.cos(2*t))**3)
    # theta
    q_guess[:,0] += s/2*np.cos(t)*np.sin(t)/(1+np.sin(t)**2) + np.pi/2-s/2 - np.radians(10)
    qd_guess[:,0] += s/2*(-2 + 6*np.cos(2*t))/(-3 + np.cos(2*t))**2
    qdd_guess[:,0] += s/2*(28*np.sin(2*t) + 6*np.sin(4*t))/(-3 + np.cos(2*t))**3


    q_guess[:,2] += 30
    # q_guess[:,2] += np.linspace(20,40,T+1)
    # q_guess[:,2] += np.sin(np.linspace(0,np.pi, T+1))*20+40

    # plt.plot(*np.degrees(q_guess[:,[1,0]].T))
    # plt.show()
    return q_guess, qd_guess, qdd_guess[:-1], u_guess

def load_trajectory(name):
    """ load all the components of a trajectory given a <name>
    trajectories are stored in the data folder

            'data/q_<name>'
            'data/qd_<name>'
            'data/qdd_<name>'
            'data/u_<name>'
            'data/h_<name>'
    """
    to_return = []

    for prefix in ['q', 'qd', 'qdd', 'u', 'h']:
        filename = 'data/' + prefix + '_' + name
        to_return.append(np.load(filename))

    return to_return

def save_trajectory(name, q, qd, qdd, u, h):
    """ save all the components of a trajectory given a <name>
    trajectories are stored in the data folder

            'data/q_<name>'
            'data/qd_<name>'
            'data/qdd_<name>'
            'data/u_<name>'
            'data/h_<name>'
    """
    prefix_list = ['q', 'qd', 'qdd', 'u', 'h']
    traj_list = [q, qd, qdd, u, h]

    for prefix, traj in zip(prefix_list, traj_list):
        filename = 'data/' + prefix + '_' + name
        np.save(filename, traj)


def retime(dt, q, qd, qdd, u, h):
    """ take an existing trajectory and a desired timestep size, and
    retime that trajectory using linear interpolation to the new timestep
    """
    old_T = u.shape[0] # number of timesteps in the original traj

    if h.size == 1:
        # if h is a single value, make a sequence of h's
        onevec = np.ones(old_T)
        onevec[0] = 0
        h = onevec * h.item()
        #h = np.ones(old_T) * h.item()

    duration = h.sum() # duration of the initial trajectory
    new_T = int(duration / dt) # desired number of steps
    new_ts = np.linspace(0, duration, new_T) # new timesteps
    old_ts = np.cumsum(h)

    to_return = []

    for traj in [q, qd, qdd, u]:
        retimed = np.array([np.interp(new_ts, old_ts, traj_dim) \
            for traj_dim in np.atleast_2d(traj[:old_T].T)]).T

        to_return.append(retimed)

    hnew = np.ones(to_return[-1].shape[0])*dt
    to_return.append(hnew)

    return to_return

def calc_power(qd, u):
    """ get the power of a trajectory
    """
    T = u.shape[0]
    return qd[:T,2].dot(u[:,1])/T

def calc_dynamics_error(q, qd, qdd, u, h, w=np.array([6,0,0])):
    """ sum the total dynamics error of a trajectory (with backwards)
    euler integration
    """
    kite = Kite()
    T = q.shape[0] - 1
    error = 0
    for t in range(T):
        dt = h[t] if h.size == T else h[0]
        x = np.concatenate([q[t+1], qd[t+1]])
        xprev = np.concatenate([q[t], qd[t]])
        xd = np.concatenate([qd[t+1], qdd[t]])
        error += np.linalg.norm(kite.f(x,u[t],w) - xd)
        error += np.linalg.norm(x - (xprev + xd*dt))

    return error

def summarize(name=None, traj=None, plot=True):
    """ Load a trajectory, print summary statistics
    You can pass in a trajectory by name or just as a tuple
    """
    if name is not None:
        q, qd, qdd, u, h = load_trajectory(name)
    if traj is not None:
        q, qd, qdd, u, h = traj

    duration = h[0]*u.shape[0] if h.size==1 else h.sum()
    print(f'Duration\t{duration}')
    print(f'Power\t{calc_power(qd,u)}')
    print(f'Error\t{calc_dynamics_error(q,qd,qdd,u,h)}')
    if plot: plot_3d_trajectory(q,qd)

if __name__ == '__main__':
    # q,qd,qdd,u,h = load_trajectory('asymmetric_opt_100.npy')
    q,qd,qdd,u = get_lemniscate_guess_trajectory(800, 3)
    h=np.array([0])
    summarize(traj=(q,qd,qdd,u,h), plot=False)
    plot_3d_trajectory(q,qd, title='T=100 orbit (6.9 W over 29 sec)')
