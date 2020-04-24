from drake_kite import DrakeKite
from pydrake.all import eq, MathematicalProgram, Solve, Variable, SnoptSolver
from simple_kite_dynamics import Kite
import numpy as np
###############################################################################
# Constants
###############################################################################
# number of dimensions of state and control
nq = 2
nu = 1
# time steps in the trajectory optimization
T = 40
# minimum and maximum time interval is seconds
h_min = 1./T
h_max = 10./T
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

    # args = np.concatenate((q[t+1], qd[t+1], qdd[t], u[t]))
    # prog.AddConstraint(dynamics, lb=[0]*nq*2, ub=[0]*nq*2, vars=args)

for t in range(T+1):
    prog.AddLinearConstraint(q[t,1] >= 0) # stay off the ground
    prog.AddLinearConstraint(q[t,1] <= np.pi/2) # behind the anchor
    prog.AddLinearConstraint(q[t,0] <= np.pi/2) # behind the anchor
    prog.AddLinearConstraint(q[t,1] >= -np.pi/2) # behind the anchor


# mirror initial and final configuration to force a loop
prog.AddLinearConstraint(eq(q[0], - q[-1]))
prog.AddLinearConstraint(eq(qd[0], - qd[-1]))

# penalize large control inputs
for t in range(T):
    prog.AddQuadraticCost(u[t].dot(u[t]))
###############################################################################
# Initial guess
###############################################################################
# vector of the initial guess
initial_guess = np.empty(prog.num_vars())

# initial guess for the time step
h_guess = h_max
prog.SetDecisionVariableValueInVector(h, [h_guess] * T, initial_guess)


q_guess = np.zeros([T+1,nq])
qd_guess = np.zeros([T+1,nq])
qdd_guess = np.zeros([T+1,nq])

# a = np.linspace(0, np.pi*2, T+1)
# q_guess = np.stack([np.sin(a), np.cos(a)+2]).T*np.pi/8
# qd_guess = np.stack([np.cos(a), -np.sin(a)]).T*np.pi/8
# qdd_guess = np.stack([-np.sin(a), -np.cos(a)]).T*np.pi/8
prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qd, qd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qdd, qdd_guess[:-1], initial_guess)

u_guess = np.zeros([T,nu])
prog.SetDecisionVariableValueInVector(u, u_guess, initial_guess)

###############################################################################
# Solve and get the solution
###############################################################################
# solve mathematical program with initial guess
solver = SnoptSolver()
result = solver.Solve(prog, initial_guess)

# ensure solution is found
print(f'Solution found? {result.is_success()}.')

# get optimal solution
h_opt = result.GetSolution(h)
q_opt = result.GetSolution(q)
qd_opt = result.GetSolution(qd)
qdd_opt = result.GetSolution(qdd)
u_opt = result.GetSolution(u)

# stack states
x_opt = np.hstack((q_opt, qd_opt)).T

from matplotlib import pyplot as plt
plt.plot(*q_guess.T, label='Guess')
plt.plot(*q_opt.T, label='Opt')
plt.legend()
plt.show()
