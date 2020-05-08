#!/usr/bin/python3
from drake_kite import DrakeKite
from pydrake.all import eq, MathematicalProgram, Solve, Variable, SnoptSolver
from pydrake.solvers.mathematicalprogram import SolverType
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
    if t < T - 1:
        prog.AddQuadraticCost((u[t+1, 0] - u[t, 0])*(u[t+1, 0] - u[t, 0]))
        prog.AddQuadraticCost((u[t+1, 1] - u[t, 1])*(u[t+1, 1] - u[t, 1]))

# for t in range(T):
#     prog.AddQuadraticCost(qd[t,2]*u[t,1]) # maximize power
prog.AddCost(0.01 * qd[:-1,2].dot(u[:,1]))
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

def create_mirrored_loop(q, qd, qdd, u):
    q_mirrored = q * np.array([[1,-1,1]])
    qd_mirrored = qd * np.array([[1,-1,1]])
    qdd_mirrored = qdd * np.array([[1,-1,1]])
    u_mirrored = u * np.array([[-1, 1]])

    q_full = np.vstack([q[:-1], q_mirrored])
    qd_full = np.vstack([qd[:-1], qd_mirrored])
    qdd_full = np.vstack([qdd[:-1], qdd_mirrored])
    u_full = np.vstack([u, u_mirrored])
    return q_full, qd_full, qdd_full, u_full


initial_guess = np.empty(prog.num_vars())


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


# q_guess = np.load(f'data/q_opt_{40}.npy')
# qd_guess = np.load(f'data/qd_opt_{40}.npy')
# qdd_guess = np.load(f'data/qdd_opt_{40}.npy')
# u_guess = np.load(f'data/u_opt_{40}.npy')
# h_guess = np.load(f'data/h_opt_{40}.npy')

# q_guess = retime(h_guess, q_guess, T)
# qd_guess = retime(h_guess, qd_guess, T)
# qdd_guess = retime(h_guess, qdd_guess, T)
# u_guess = retime(h_guess, u_guess, T)
# h_guess = retime(h_guess, h_guess, T)

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


fig = plt.figure() 
ax = plt.axes(projection='3d') 
ax.plot3D(*euclidean_guess.T, label='Guess') 
ax.plot3D(*euclidean_opt.T, label='Opt') 
ax.legend()

plt.figure()
plt.plot(u_guess[:,0], label='Guess')
plt.plot(u_opt[:,0], label='Opt')
plt.title('Roll Control')
plt.legend()

plt.figure()
plt.plot(u_guess[:,1], label='Guess')
plt.plot(u_opt[:,1], label='Opt')
plt.title('Tether Control')
plt.legend()

plt.figure()
plt.plot(qd_opt[:-1,2]*u_opt[:,1])
plt.title('Power')

plt.show()


