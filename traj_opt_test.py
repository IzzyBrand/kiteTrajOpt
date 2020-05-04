#!/usr/bin/python3
from drake_kite import DrakeKite
from pydrake.all import eq, MathematicalProgram, Solve, Variable, SnoptSolver
from simple_kite_dynamics import Kite
import numpy as np
from matplotlib import pyplot as plt
import sys
###############################################################################
# Constants
###############################################################################
# number of dimensions of state and control
nq = 3
nu = 2
# time steps in the trajectory optimization
T = 100
# minimum and maximum time interval is seconds
h_min = 5./T
h_max = 50./T
w = np.array([6, 0, 0])

###############################################################################
# Set up the mathematical program
###############################################################################
# initialize program
prog = MathematicalProgram()
# vector of the time intervals
# (distances between the T + 1 break points)
h = prog.NewContinuousVariables(T, name='h')
# system configuration, generalized velocities, and accelerations
q = prog.NewContinuousVariables(rows=T+1, cols=nq, name='q')
qd = prog.NewContinuousVariables(rows=T+1, cols=nq, name='qd')
qdd = prog.NewContinuousVariables(rows=T, cols=nq, name='qdd')
# control inputs
u = prog.NewContinuousVariables(rows=T, cols=nu, name='u')

###############################################################################
# Add constraints
###############################################################################
# lower and upper bound on the time steps for all t
prog.AddBoundingBoxConstraint([h_min] * T, [h_max] * T, h)

# dynamics equations for all t (implicit Euler)
kite = Kite()
def dynamics(args):
    x = args[:nq*2]
    xd = args[nq:nq*3]
    u = args[-nu:]
    # TODO: wind should be a function of time
    return xd - kite.f(x, u, w)

# link the configurations, velocities, and accelerations
# uses implicit Euler method, https://en.wikipedia.org/wiki/Backward_Euler_method
for t in range(T):
    prog.AddConstraint(eq(q[t+1], q[t] + h[t] * qd[t]))
    prog.AddConstraint(eq(qd[t+1], qd[t] + h[t] * qdd[t]))

    args = np.concatenate((q[t], qd[t], qdd[t], u[t]))
    prog.AddConstraint(dynamics, lb=[0]*nq*2, ub=[0]*nq*2, vars=args)

    # prog.AddConstraint(eq(q[t+1], q[t] + h[t] * qd[t+1])) # backward euler
    # prog.AddConstraint(eq(qd[t+1], qd[t] + h[t] * qdd[t])) # forward euler

for t in range(T+1):

    #prog.AddLinearConstraint(q[t,1] >= 0) # stay off the ground
    #prog.AddLinearConstraint(q[t,1] <= np.pi/2) # behind the anchor
    #prog.AddLinearConstraint(q[t,0] <= np.pi/2) # behind the anchor
    #prog.AddLinearConstraint(q[t,1] >= -np.pi/2) # behind the anchor

    #prog.AddLinearConstraint(q[t,0] <= (np.pi / 2)) # stay off the ground
    #prog.AddLinearConstraint(q[t,0] >= (-np.pi / 2)) # stay off the ground

    prog.AddLinearConstraint(q[t,0] <= np.radians(75)) # stay off the ground
    # prog.AddLinearConstraint(q[t,2] == 50) # keep the kite at 50m
    prog.AddLinearConstraint(q[t,2] >= 40)
    prog.AddLinearConstraint(q[t,2] <= 60)

    #prog.AddLinearConstraint(q[t,0] >= np.radians(2)) # stay out of vertical singularity

    #prog.AddLinearConstraint(q[t,1] <= np.pi/2) # behind the anchor
    #prog.AddLinearConstraint(q[t,0] <= np.pi/2) # behind the anchor
    #prog.AddLinearConstraint(q[t,1] >= -np.pi/2) # behind the anchor

# the trajectory must be a closed circuit
prog.AddLinearConstraint(eq(q[0], q[-1]))
prog.AddLinearConstraint(eq(qd[0], qd[-1]))

# penalize large control inputs and nonsmooth control inputs
for t in range(T):
    prog.AddLinearConstraint(u[t,1] <= 0) # you can't push the kite
    prog.AddLinearConstraint(u[t,1] >= -10) # limit generator torque
    prog.AddLinearConstraint(u[t,0] <= np.degrees(20))
    prog.AddLinearConstraint(u[t,0] >= -np.degrees(20))

    prog.AddQuadraticCost(u[t, 0]*u[t, 0]) # penalize roll inputs

    # control smoothing constraint
    if t < T - 1:
        prog.AddQuadraticCost((u[t+1, 0] - u[t, 0])*(u[t+1, 0] - u[t, 0]))
        prog.AddQuadraticCost((u[t+1, 1] - u[t, 1])*(u[t+1, 1] - u[t, 1]))

for t in range(T):
    prog.AddQuadraticCost(qd[t,2]*u[t,1]) # maximize power
# prog.AddCost(qd[:-1,2].dot(u[:,1]))
###############################################################################
# Initial guess
###############################################################################
# vector of the initial guess

def retime(h, X, T):
    ts = np.linspace(0,h.sum(),T)
    if h.shape[0] == X.shape[0]:
        return np.array([np.interp(ts, np.cumsum(h), x)\
            for x in np.atleast_2d(X.T)]).T
    elif h.shape[0]+1 == X.shape[0]:
        return np.vstack([retime(h,X[:-1],T), X[-1]])
    else:
        print('retime: h and X are different lengths')
        return None


initial_guess = np.empty(prog.num_vars())

q_guess = np.load(f'data/q_opt_{T}.npy')
qd_guess = np.load(f'data/qd_opt_{T}.npy')
qdd_guess = np.load(f'data/qdd_opt_{T}.npy')
u_guess = np.load(f'data/u_opt_{T}.npy')
h_guess = np.load(f'data/h_opt_{T}.npy')

# q_guess = retime(h_guess, q_guess, T)
# qd_guess = retime(h_guess, qd_guess, T)
# qdd_guess = retime(h_guess, qdd_guess, T)
# u_guess = retime(h_guess, u_guess, T)
# h_guess = retime(h_guess, h_guess, T)

prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qd, qd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qdd, qdd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(u, u_guess, initial_guess)
prog.SetDecisionVariableValueInVector(h, h_guess, initial_guess)

###############################################################################
# Solve and get the solution
###############################################################################
# solve mathematical program with initial guess
solver = SnoptSolver()
result = solver.Solve(prog, initial_guess)

# ensure solution is found
print(f'Solution found? {result.is_success()}.')
# print(result.get_solution_result())
# print(result.get_solver_details().info)

# get optimal solution
h_opt = result.GetSolution(h)
q_opt = result.GetSolution(q)
qd_opt = result.GetSolution(qd)
qdd_opt = result.GetSolution(qdd)
u_opt = result.GetSolution(u)

# stack states
x_opt = np.hstack((q_opt, qd_opt)).T


x,y,r = q_guess.T
#print('Theta, Phi:', x[0], y[0])
#plt.plot(*q_guess.T, label='Guess')
plt.plot(np.degrees(y),np.degrees(x), label='Guess')
#plt.plot(*q_opt.T, label='Opt')
x,y,r = q_opt.T
plt.plot(np.degrees(y), np.degrees(x), label='Opt')
plt.legend()
plt.xlabel('phi')
plt.ylabel('theta')

plt.figure()
plt.plot(np.degrees(u_opt))

x,y,r = qd_opt.T
#print('Thetadot, Phidot:', x[0], y[0])

#print(np.cumsum(h_opt), u_opt)
u_interp = np.interp(np.linspace(0, np.sum(h_opt[:-1]), 5*(T-1)), np.concatenate((np.array([0]), np.cumsum(h_opt)[:-1])), u_opt[:, 0])
np.save('fig8_openloop_control.npy', u_interp)
np.save('fig8_openloop_times.npy', np.arange(len(u_interp)) * np.sum(h_opt) / len(u_interp))
#print(u_interp)
plt.plot(np.arange(len(u_interp))/5, np.degrees(u_interp))

plt.xlabel('t')
plt.ylabel('u')
plt.figure()
plt.plot(np.arange(len(u_interp)) * np.sum(h_opt) / len(u_interp), np.degrees(u_interp))

plt.figure()
t, p, r = q_opt.T
plt.title('Theta, Phi')
plt.plot(t)
plt.plot(p)
plt.legend(['Theta', 'Phi'])
plt.xlabel('Time')
plt.ylabel('Radians')

plt.figure()
t, p, r = qd_opt.T
plt.title('dot Theta, Phi')
plt.plot(t)
plt.plot(p)
plt.legend(['dTheta', 'dPhi'])
plt.xlabel('Time')
plt.ylabel('Radians / s')

plt.show()
