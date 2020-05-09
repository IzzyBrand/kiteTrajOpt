import numpy as np

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
    n = 0.01
    q_guess = np.random.randn(T+1,3)*n
    qd_guess = np.random.randn(T+1,3)*n
    qdd_guess = np.random.randn(T+1,3)*n
    u_guess = np.random.randn(T,2)*n
    # phi
    q_guess[:,1] += s*np.cos(t)/(1+np.sin(t)**2)
    qd_guess[:,1] += -s*(9*np.sin(t) + np.sin(3*t))/(-3 + np.cos(2*t))**2
    qdd_guess[:,1] += s*(2*np.cos(t) + 45*np.cos(3*t) + np.cos(5*t))/(2*(-3 + np.cos(2*t))**3)
    # theta
    q_guess[:,0] += s*np.cos(t)*np.sin(t)/(1+np.sin(t)**2) + s
    qd_guess[:,0] += s*(-2 + 6*np.cos(2*t))/(-3 + np.cos(2*t))**2
    qdd_guess[:,0] += s*(28*np.sin(2*t) + 6*np.sin(4*t))/(-3 + np.cos(2*t))**3

    q_guess[:,2] += 40

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
        h = np.ones(old_T) * h.item()

    duration = h.sum() # duration of the initial trajectory
    new_T = int(duration / dt) # desired number of steps
    new_ts = np.linspace(0, duration, new_T) # new timesteps
    old_ts = np.cumsum(h)

    to_return = []

    for traj in [q, qd, qdd, u, h]:
        retimed = np.array([np.interp(new_ts, old_ts, traj_dim) \
            for traj_dim in np.atleast_2d(traj[:old_T].T)]).T

        to_return.append(retimed)

    return to_return

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    q,_,_,_ = get_lemniscate_guess_trajectory(100, 1.5)
    plt.plot(*q[:,[1,0]].T)
    plt.show()
