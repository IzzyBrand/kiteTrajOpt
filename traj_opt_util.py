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


