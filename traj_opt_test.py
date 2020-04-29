#!/usr/bin/python3
from drake_kite import DrakeKite
from pydrake.all import eq, MathematicalProgram, Solve, Variable, SnoptSolver
from simple_kite_dynamics import Kite
import numpy as np
from matplotlib import pyplot as plt
###############################################################################
# Constants
###############################################################################
# number of dimensions of state and control
nq = 3
nu = 2
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

for t in range(T+1):

    #prog.AddLinearConstraint(q[t,1] >= 0) # stay off the ground
    #prog.AddLinearConstraint(q[t,1] <= np.pi/2) # behind the anchor
    #prog.AddLinearConstraint(q[t,0] <= np.pi/2) # behind the anchor
    #prog.AddLinearConstraint(q[t,1] >= -np.pi/2) # behind the anchor

    #prog.AddLinearConstraint(q[t,0] <= (np.pi / 2)) # stay off the ground
    #prog.AddLinearConstraint(q[t,0] >= (-np.pi / 2)) # stay off the ground

    prog.AddLinearConstraint(q[t,0] <= np.radians(75)) # stay off the ground
    # prog.AddLinearConstraint(q[t,2] == 50) # keep the kite at 50m

    #prog.AddLinearConstraint(q[t,0] >= np.radians(2)) # stay out of vertical singularity

    #prog.AddLinearConstraint(q[t,1] <= np.pi/2) # behind the anchor
    #prog.AddLinearConstraint(q[t,0] <= np.pi/2) # behind the anchor
    #prog.AddLinearConstraint(q[t,1] >= -np.pi/2) # behind the anchor



prog.AddLinearConstraint(eq(q[0], q[-1]))
prog.AddLinearConstraint(eq(qd[0], qd[-1]))

# penalize large control inputs and nonsmooth control inputs
for t in range(T):
    prog.AddLinearConstraint(u[t,1] <= 0) # you can't push the kite
    prog.AddLinearConstraint(u[t,1] >= -7) # limit generator torque
    prog.AddLinearConstraint(u[t,0] <= np.degrees(20))
    prog.AddLinearConstraint(u[t,0] >= -np.degrees(20))

    # prog.AddQuadraticCost(u[t, 0]*u[t, 0]) # limit roll control
    # control smoothing constraint
    if t < T - 1:
        prog.AddQuadraticCost( (u[t+1, 0] - u[t, 0])*(u[t+1, 0] - u[t, 0])) 
        prog.AddQuadraticCost( (u[t+1, 1] - u[t, 1])*(u[t+1, 1] - u[t, 1])) 

for t in range(T):
    prog.AddQuadraticCost(qd[t,2]*u[t,1]) # maximize power
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

#q_guess = np.random.random(q_guess.shape)
#qd_guess = np.random.random(qd_guess.shape)
#qdd_guess = np.random.random(qdd_guess.shape)

# Initial guess for a figure-8
# First, chooses initial conditions for 8 time steps
# Then, interpolate linearly
# There's probably a better way to do this.
q_guess[0] = np.array([np.radians(60), np.radians(0), 50])
q_guess[5] = np.array([np.radians(70), np.radians(30), 50])
q_guess[10] = np.array([np.radians(60), np.radians(40), 50])
q_guess[15] = np.array([np.radians(45), np.radians(25), 50])
q_guess[20] = np.array([np.radians(60), np.radians(0), 50])
q_guess[25] = np.array([np.radians(70), np.radians(-35), 50])
q_guess[30] = np.array([np.radians(55), np.radians(-45), 50])
q_guess[35] = np.array([np.radians(50), np.radians(-30), 50])
q_guess[39] = np.array([np.radians(60), np.radians(0), 50])

q_guess[0:5] = np.linspace(q_guess[0], q_guess[5], 5)
q_guess[5:10] = np.linspace(q_guess[5], q_guess[10], 5)
q_guess[10:15] = np.linspace(q_guess[10], q_guess[15], 5)
q_guess[15:20] = np.linspace(q_guess[15], q_guess[20], 5)
q_guess[20:25] = np.linspace(q_guess[20], q_guess[25], 5)
q_guess[25:30] = np.linspace(q_guess[25], q_guess[30], 5)
q_guess[30:35] = np.linspace(q_guess[30], q_guess[35], 5)
q_guess[35:39] = np.linspace(q_guess[35], q_guess[39], 4)
q_guess[40] = q_guess[39]


qd_guess[0] = np.array([np.radians(25), np.radians(25), 0])
qd_guess[5] = np.array([np.radians(0), np.radians(25), 0])
qd_guess[10] = np.array([np.radians(-25), np.radians(0), 0])
qd_guess[15] = np.array([np.radians(0), np.radians(-25), 0])
qd_guess[20] = np.array([np.radians(25), np.radians(-25), 0])
qd_guess[25] = np.array([np.radians(0), np.radians(-25), 0])
qd_guess[30] = np.array([np.radians(-25), np.radians(0), 0])
qd_guess[35] = np.array([np.radians(0), np.radians(25), 0])
qd_guess[39] = np.array([np.radians(25), np.radians(25), 0])

qd_guess[0:5] =   np.linspace(qd_guess[0],  qd_guess[5], 5)
qd_guess[5:10] =  np.linspace(qd_guess[5],  qd_guess[10], 5)
qd_guess[10:15] = np.linspace(qd_guess[10], qd_guess[15], 5)
qd_guess[15:20] = np.linspace(qd_guess[15], qd_guess[20], 5)
qd_guess[20:25] = np.linspace(qd_guess[20], qd_guess[25], 5)
qd_guess[25:30] = np.linspace(qd_guess[25], qd_guess[30], 5)
qd_guess[30:35] = np.linspace(qd_guess[30], qd_guess[35], 5)
qd_guess[35:39] = np.linspace(qd_guess[35], qd_guess[39], 4)
qd_guess[40] = qd_guess[39]



# a = np.linspace(0, np.pi*2, T+1)
# q_guess = np.stack([np.sin(a), np.cos(a)+2]).T*np.pi/8
# qd_guess = np.stack([np.cos(a), -np.sin(a)]).T*np.pi/8
# qdd_guess = np.stack([-np.sin(a), -np.cos(a)]).T*np.pi/8
prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qd, qd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qdd, qdd_guess[:-1], initial_guess)

#u_guess = np.zeros([T,nu])
# u_guess = np.radians(10*np.sin(np.linspace([0]*2, [2*np.pi]*2, T)))
u_guess = np.load('u_opt.npy')
plt.show()
prog.SetDecisionVariableValueInVector(u, u_guess, initial_guess)

###############################################################################
# Solve and get the solution
###############################################################################
# solve mathematical program with initial guess
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

# stack states
x_opt = np.hstack((q_opt, qd_opt)).T


x,y,r = q_guess.T
print('Theta, Phi:', x[0], y[0])
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
print('Thetadot, Phidot:', x[0], y[0])

print(np.cumsum(h_opt), u_opt)
u_interp = np.interp(np.linspace(0, np.sum(h_opt[:-1]), 5*(T-1)), np.concatenate((np.array([0]), np.cumsum(h_opt)[:-1])), u_opt[:, 0])
np.save('fig8_openloop_control.npy', u_interp)
np.save('fig8_openloop_times.npy', np.arange(len(u_interp)) * np.sum(h_opt) / len(u_interp))
print(u_interp)
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
