from drake_kite import DrakeKite
from pydrake.all import eq, MathematicalProgram, Solve, Variable, SnoptSolver
from simple_kite_dynamics import Kite
import numpy as np
from matplotlib import pyplot as plt

# number of dimensions of state and control
nq = 3
nu = 2
w = np.array([6, 0, 0])

class MPC:
    def __init__(self, T):
        self.T = T # number of planning steps
        self.kite = Kite()

    def dynamics(self, args):
        x = args[:nq*2]
        xd = args[nq:nq*3]
        u = args[-nu:]
        # TODO: wind should be a function of time
        return xd - self.kite.f(x, u, w)

    def setup_optimization_variables(self):
        # initialize program
        self.prog = MathematicalProgram()
        # system configuration, generalized velocities, and accelerations
        q = self.prog.NewContinuousVariables(rows=self.T+1, cols=nq, name='q')
        qd = self.prog.NewContinuousVariables(rows=self.T+1, cols=nq, name='qd')
        qdd = self.prog.NewContinuousVariables(rows=self.T, cols=nq, name='qdd')
        # control inputs
        u = self.prog.NewContinuousVariables(rows=self.T, cols=nu, name='u')
        # vector of the time intervals
        h = self.prog.NewContinuousVariables(self.T, name='h')

        return q, qd, qdd, u, h

    def add_constraints(self, variables):
        q, qd, qdd, u, h = variables

        # lower and upper bound on the time steps for all t
        self.prog.AddBoundingBoxConstraint([1./self.T] * self.T,
            [10./self.T] * self.T, h)

        # dynamics constraints
        for t in range(self.T):
            self.prog.AddConstraint(eq(q[t+1], q[t] + h[t] * qd[t]))
            self.prog.AddConstraint(eq(qd[t+1], qd[t] + h[t] * qdd[t]))

            args = np.concatenate((q[t], qd[t], qdd[t], u[t]))
            self.prog.AddConstraint(self.dynamics,
                lb=[0]*nq*2, ub=[0]*nq*2, vars=args)

        # bound tether and add ground constraint
        for t in range(self.T+1):
            self.prog.AddLinearConstraint(q[t,0] <= np.radians(80))
            self.prog.AddLinearConstraint(q[t,2] == 50)
            # self.prog.AddLinearConstraint(q[t,2] >= 40)
            # self.prog.AddLinearConstraint(q[t,2] <= 60)

        # control input constrinats
        for t in range(self.T):
            self.prog.AddLinearConstraint(u[t,1] <= 0) # you can't push the kite
            self.prog.AddLinearConstraint(u[t,1] >= -10) # limit generator torque
            self.prog.AddLinearConstraint(u[t,0] <= np.degrees(20))
            self.prog.AddLinearConstraint(u[t,0] >= -np.degrees(20))

    def add_costs(self, variables):
        q, qd, qdd, u, h = variables

        # control smoothness cost
        for t in range(self.T-1):
            self.prog.AddQuadraticCost((u[t+1, 0] - u[t, 0])*(u[t+1, 0] - u[t, 0]))
            self.prog.AddQuadraticCost((u[t+1, 1] - u[t, 1])*(u[t+1, 1] - u[t, 1]))

        # energy generation cost
        # self.prog.AddCost(qd[:-1,2].dot(u[:,1]))

        # TODO: add trajectory tracking cost

    def set_initial_guess(self, variables):
        q, qd, qdd, u, h = variables

        initial_guess = np.empty(self.prog.num_vars())

        q_guess = np.load(f'data/q_opt_{self.T}.npy')
        qd_guess = np.load(f'data/qd_opt_{self.T}.npy')
        qdd_guess = np.load(f'data/qdd_opt_{self.T}.npy')
        u_guess = np.load(f'data/u_opt_{self.T}.npy')
        h_guess = np.load(f'data/h_opt_{self.T}.npy')

        self.prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
        self.prog.SetDecisionVariableValueInVector(qd, qd_guess, initial_guess)
        self.prog.SetDecisionVariableValueInVector(qdd, qdd_guess, initial_guess)
        self.prog.SetDecisionVariableValueInVector(u, u_guess, initial_guess)
        self.prog.SetDecisionVariableValueInVector(h, h_guess, initial_guess)

        return initial_guess

    def optimize(self, variables, initial_guess):
        q, qd, qdd, u, h = variables
        # solve mathematical program with initial guess
        solver = SnoptSolver()
        result = solver.Solve(self.prog, initial_guess)

        # ensure solution is found
        print(f'Solution found? {result.is_success()}.')
        print(result.get_solution_result())
        print(result.get_solver_details().info)

        # get optimal solution
        q_opt = result.GetSolution(q)
        qd_opt = result.GetSolution(qd)
        qdd_opt = result.GetSolution(qdd)
        u_opt = result.GetSolution(u)
        h_opt = result.GetSolution(h)

        return q_opt, qd_opt, qdd_opt, u_opt, h_opt

    def plan(self):
        variables = self.setup_optimization_variables()
        self.add_constraints(variables)
        self.add_costs(variables)
        initial_guess = self.set_initial_guess(variables)
        result = self.optimize(variables, initial_guess)
        return result


if __name__ == '__main__':
    # q_ref = np.load(f'data/q_opt_{T}.npy')
    # qd_ref = np.load(f'data/qd_opt_{T}.npy')
    # qdd_ref = np.load(f'data/qdd_opt_{T}.npy')
    # u_ref = np.load(f'data/u_opt_{T}.npy')
    # h_ref = np.load(f'data/h_opt_{T}.npy')

    mpc = MPC(40)
    q_opt, qd_opt, qdd_opt, u_opt, h_opt = mpc.plan()

