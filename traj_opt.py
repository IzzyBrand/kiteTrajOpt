#!/usr/bin/python3
from drake_kite import DrakeKite
from pydrake.all import eq, MathematicalProgram, Solve, Variable, SnoptSolver
from pydrake.solvers.mathematicalprogram import SolverType
from simple_kite_dynamics import Kite
from util import *
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
T = 50
# minimum and maximum time interval is seconds
h_min = 5./T
h_max = 20./T
w = np.array([6, 0, 0])

###############################################################################
# Set up the mathematical program
###############################################################################
# initialize program
prog = MathematicalProgram()
# vector of the time intervals
# (distances between the T + 1 break points)
h = prog.NewContinuousVariables(1, name='h')
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
prog.AddBoundingBoxConstraint([h_min], [h_max], h)

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
    # prog.AddConstraint(eq(q[t+1], q[t] + h[0] * qd[t]))
    # prog.AddConstraint(eq(qd[t+1], qd[t] + h[0] * qdd[t]))

    # args = np.concatenate((q[t], qd[t], qdd[t], u[t])) # forward euler
    args = np.concatenate((q[t+1], qd[t+1], qdd[t], u[t])) # backward euler
    prog.AddConstraint(dynamics, lb=[0]*nq*2, ub=[0]*nq*2, vars=args)

    prog.AddConstraint(eq(q[t+1], q[t] + h[0] * qd[t+1])) # backward euler
    prog.AddConstraint(eq(qd[t+1], qd[t] + h[0] * qdd[t])) # backward euler

for t in range(T+1):
    prog.AddLinearConstraint(q[t,0] <= np.radians(75)) # stay off the ground
    prog.AddLinearConstraint(q[t,0] >= np.radians(2)) # stay out of vertical singularity
    # prog.AddLinearConstraint(q[t,1] <= 0) # stay to one side of the symmetry line
    # prog.AddLinearConstraint(q[t,2] == 40) # keep the kite at a fixed length
    prog.AddLinearConstraint(q[t,2] >= 10) # minimum tether length
    prog.AddLinearConstraint(q[t,2] <= 60) # maximum tether length


# the trajectory must be a closed circuit
prog.AddLinearConstraint(q[0,1] == 0)
prog.AddLinearConstraint(eq(q[0], q[-1]))
prog.AddLinearConstraint(qd[0,0] == qd[-1,0])
prog.AddLinearConstraint(qd[0,1] == -qd[-1,1])
prog.AddLinearConstraint(qd[0,2] == qd[-1,2])

# penalize large control inputs and nonsmooth control inputs
for t in range(T):
    prog.AddLinearConstraint(u[t,1] <= 0) # you can't push the kite
    prog.AddLinearConstraint(u[t,1] >= -10) # limit generator torque
    prog.AddLinearConstraint(u[t,0] <= np.degrees(20))
    prog.AddLinearConstraint(u[t,0] >= -np.degrees(20))

    # prog.AddQuadraticCost(u[t, 0]*u[t, 0]) # penalize roll inputs

# control smoothing constraint
for t in range(T-1):
    prog.AddQuadraticCost((u[t+1, 0] - u[t, 0])*(u[t+1, 0] - u[t, 0]))
    prog.AddQuadraticCost((u[t+1, 1] - u[t, 1])*(u[t+1, 1] - u[t, 1]))

# control smoothing constraint across the last timestep
prog.AddQuadraticCost((u[0, 0] + u[-1, 0])*(u[0, 0] + u[-1, 0]))
prog.AddQuadraticCost((u[0, 1] - u[-1, 1])*(u[0, 1] - u[-1, 1]))

# for t in range(T):
#     prog.AddQuadraticCost(qd[t,2]*u[t,1]) # maximize power
prog.AddCost(0.01 * qd[:-1,2].dot(u[:,1]))
###############################################################################
# Initial guess
###############################################################################
initial_guess = np.empty(prog.num_vars())

q_guess, qd_guess, qdd_guess, u_guess = get_circle_guess_trajectory(T)
h_guess = [h_min]

prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qd, qd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qdd, qdd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(u, u_guess, initial_guess)
prog.SetDecisionVariableValueInVector(h, h_guess, initial_guess)

###############################################################################
# Solve and get the solution
###############################################################################
# solve mathematical program with initial guess
prog.SetSolverOption(SolverType.kSnopt, "Major iterations limit", 10000)
solver = SnoptSolver()
result = solver.Solve(prog, initial_guess)

# ensure solution is found
print(f'Solution found? {result.is_success()}.')
print(result.get_solution_result())
print(result.get_solver_details().info)

# get optimal solution
h_opt = result.GetSolution(h)
q_opt = result.GetSolution(q)
qd_opt = result.GetSolution(qd)
qdd_opt = result.GetSolution(qdd)
u_opt = result.GetSolution(u)

# mirror the results for viewing
q_opt, qd_opt, qdd_opt, u_opt = create_mirrored_loop(q_opt, qd_opt, qdd_opt, u_opt)
q_guess, qd_guess, qdd_guess, u_guess = create_mirrored_loop(q_guess, qd_guess, qdd_guess, u_guess)
T *= 2

print(f'Duration: {T*h_opt}')
print(f'Power: {qd_opt[:-1,2].dot(u_opt[:,1])/(T*h_opt[0])}')

# convert to euclidean coordinates
x_guess = np.hstack([q_guess, qd_guess])
euclidean_guess = np.array([kite.p(x_t) for x_t in x_guess])
x_opt = np.hstack([q_opt, qd_opt])
euclidean_opt = np.array([kite.p(x_t) for x_t in x_opt])

# plot the trajectory in 3d
fig = plt.figure() 
ax = plt.axes(projection='3d') 
ax.plot3D(*euclidean_guess.T, label='Guess') 
ax.plot3D(*euclidean_opt.T, label='Opt') 
ax.legend()

# plot the roll control
plt.figure()
plt.plot(u_guess[:,0], label='Guess')
plt.plot(u_opt[:,0], label='Opt')
plt.title('Roll Control')
plt.legend()

# plot the pitch control
plt.figure()
plt.plot(u_guess[:,1], label='Guess')
plt.plot(u_opt[:,1], label='Opt')
plt.title('Tether Control')
plt.legend()

# plot the power output
plt.figure()
plt.plot(qd_opt[:-1,2]*u_opt[:,1])
plt.title('Power')

plt.show()


