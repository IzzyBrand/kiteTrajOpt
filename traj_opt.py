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
T = 300
# minimum and maximum time interval is seconds
h_min = 10./T
h_max = 30./T
w = np.array([6, 0, 0])

symmetric = True

###############################################################################
# Set up the mathematical program
###############################################################################
# initialize program
prog = MathematicalProgram()
# vector of the time intervals
# (distances between the T + 1 break points)
h = [30/T]
# h = prog.NewContinuousVariables(1, name='h')
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
# prog.AddBoundingBoxConstraint([h_min], [h_max], h)

# dynamics equations for all t (implicit Euler)
kite = Kite()
def dynamics(args):
    x = args[:nq*2]
    xd = args[nq:nq*3]
    u = args[-nu:]
    # TODO: wind should be a function of time
    return xd - kite.f(x, u, w)


# return an internal dynamics model quantity which we want to constraint
# so we can pass through arcsin without errors
def wind_projection_ratio_bound(args):
    x = args[:nq*2]
    u = args[-nu:]
    return [kite.e_t(x, u, w, get_wind_projection_ratio=True)]



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

    # add a constaint on internal dynamics model quantity
    # prog.AddConstraint(wind_projection_ratio_bound, lb=[-.99], ub=[.99], vars=args)

for t in range(T+1):
    prog.AddLinearConstraint(q[t,0] <= np.radians(75)) # stay off the ground
    prog.AddLinearConstraint(q[t,0] >= np.radians(5)) # stay out of vertical singularity
    prog.AddLinearConstraint(q[t,1] <= np.radians(80)) # stay behind the anchor
    prog.AddLinearConstraint(q[t,1] >= -np.radians(80)) # stay behind the anchor
    prog.AddLinearConstraint(q[t,2] >= 20) # minimum tether length
    prog.AddLinearConstraint(q[t,2] <= 40) # maximum tether length
    # prog.AddLinearConstraint(q[t,2] == 40)

if symmetric:
    # the trajectory must be a closed circuit (symmetric)
    prog.AddLinearConstraint(q[0,1] == 0)
    prog.AddLinearConstraint(eq(q[0], q[-1]))
    prog.AddLinearConstraint(qd[0,0] == qd[-1,0])
    prog.AddLinearConstraint(qd[0,1] == -qd[-1,1])
    prog.AddLinearConstraint(qd[0,2] == qd[-1,2])
else:
    # the trajectory must be a closed circuit (not symmetric)
    prog.AddLinearConstraint(eq(q[0], q[-1]))
    prog.AddLinearConstraint(eq(qd[0], qd[-1]))

# penalize large control inputs and nonsmooth control inputs
for t in range(T):
    prog.AddLinearConstraint(u[t,1] <= -0.001) # you can't push the kite
    prog.AddLinearConstraint(u[t,1] >= -200) # limit generator torque
    prog.AddLinearConstraint(u[t,0] <= np.radians(20))
    prog.AddLinearConstraint(u[t,0] >= -np.radians(20))


# prog.AddLinearConstraint(q[T//2,1] >= np.radians(30))

tether_smoothness = 0.5 #5. * (T-5)/T # Newtons
roll_smoothness = 100. #0.5 * (T-5)/T  # radians
power_cost_scale = 0.1    # watts

# control smoothing constraint
for t in range(T-1):
    prog.AddQuadraticCost(roll_smoothness*(u[t+1, 0] - u[t, 0])*(u[t+1, 0] - u[t, 0]))
    prog.AddQuadraticCost(tether_smoothness*(u[t+1, 1] - u[t, 1])*(u[t+1, 1] - u[t, 1]))

if symmetric:
    # control smoothing constraint across the last timestep (for symmetric trajectory)
    prog.AddQuadraticCost(roll_smoothness*(u[0, 0] + u[-1, 0])*(u[0, 0] + u[-1, 0]))
    prog.AddQuadraticCost(tether_smoothness*(u[0, 1] - u[-1, 1])*(u[0, 1] - u[-1, 1]))

# power generation costs
prog.AddQuadraticCost(power_cost_scale * qd[:-1,2].dot(u[:,1]))
###############################################################################
# Initial guess
###############################################################################
initial_guess = np.empty(prog.num_vars())


# q_guess, qd_guess, qdd_guess, u_guess, h_guess = retime(20/T,*load_trajectory('strong_opt_250.npy'))
# q_guess = q_guess[:T+1]
# qd_guess = qd_guess[:T+1]
# qdd_guess = qdd_guess[:T]
# u_guess = u_guess[:T]
# h_guess = [h_guess[0]]

# q_guess, qd_guess, qdd_guess, u_guess = get_circle_guess_trajectory(T)
# h_guess = [h_min]


q_guess, qd_guess, qdd_guess, u_guess =\
    get_lemniscate_guess_trajectory(T, num_loops=1.5)
# h_guess = [h_min]

prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qd, qd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qdd, qdd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(u, u_guess, initial_guess)
# prog.SetDecisionVariableValueInVector(h, h_guess, initial_guess)

###############################################################################
# Solve and get the solution
###############################################################################
# print out a title so we can keep track of multiple experiments
traj_opt_title = "short 3 loops"
description = f"""
roll_smoothness = {roll_smoothness}
tether_smoothness = {tether_smoothness}
power_cost_scale = {power_cost_scale}
T = {T}"""
print(traj_opt_title)
print(description)

# solve mathematical program with initial guess
prog.SetSolverOption(SolverType.kSnopt, "Print file", "snopt.out")
# prog.SetSolverOption(SolverType.kSnopt, "Major iterations limit", 100000)
# prog.SetSolverOption(SolverType.kSnopt, "Minor iterations limit", 1e12)

# found the iteration limit code here
# https://github.com/benthomsen/mit-6832-project/blob/master/airplane_system.py
it_limit = int(max(20000, 50*prog.num_vars()))
prog.SetSolverOption(SolverType.kSnopt, 'Iterations limit', it_limit)

solver = SnoptSolver()
result = solver.Solve(prog, initial_guess)

# ensure solution is found
print(f'Solution found? {result.is_success()}.')
print(result.get_solution_result())
print(result.get_solver_details().info)

# get optimal solution
h_opt = h
# h_opt = result.GetSolution(h)
q_opt = result.GetSolution(q)
qd_opt = result.GetSolution(qd)
qdd_opt = result.GetSolution(qdd)
u_opt = result.GetSolution(u)

# mirror the results for viewing
if symmetric:
    q_opt, qd_opt, qdd_opt, u_opt = create_mirrored_loop(q_opt, qd_opt, qdd_opt, u_opt)
    q_guess, qd_guess, qdd_guess, u_guess = create_mirrored_loop(q_guess, qd_guess, qdd_guess, u_guess)
    T *= 2

# print trajectory statistics
summarize(traj=(q_opt, qd_opt, qdd_opt, u_opt, h_opt), plot=False)

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
ax.set_title(traj_opt_title) # the title that we printed at the start of the run
ax.legend()

# # plot the roll control
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle(traj_opt_title)
ax1.plot(np.degrees(u_guess[:,0]), label='Guess')
ax1.plot(np.degrees(u_opt[:,0]), label='Opt')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Degrees')
ax1.set_title('Roll Control')

# plot the pitch control
ax2.plot(u_guess[:,1], label='Guess')
ax2.plot(u_opt[:,1], label='Opt')
ax2.set_xlabel('Timestep')
ax2.set_ylabel('Newtons')
ax2.set_title('Tether Control')

# # plot the power output
ax3.plot(qd_opt[:-1,2]*u_opt[:,1])
ax3.set_title('Power')

plt.legend()
plt.show()


