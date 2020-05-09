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
    def __init__(self, T, ref_traj, dt):
        self.T = T # number of planning steps
        self.ref_traj = ref_traj
        self.ref_T = self.ref_traj[0].shape[0] - 1
        self.kite = Kite()
        self.dt = dt
        self.gamma = 0.8 # cost decay along trajectory

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

        return q, qd, qdd, u

    def add_constraints(self, variables, q_0, qd_0):
        q, qd, qdd, u = variables

        # dynamics constraints
        for t in range(self.T):
            # forward euler
            # self.prog.AddConstraint(eq(q[t+1], q[t] + self.dt * qd[t]))
            # self.prog.AddConstraint(eq(qd[t+1], qd[t] + self.dt * qdd[t]))

            # args = np.concatenate((q[t], qd[t], qdd[t], u[t]))
            # self.prog.AddConstraint(self.dynamics,
            #     lb=[0]*nq*2, ub=[0]*nq*2, vars=args)
            args = np.concatenate((q[t+1], qd[t+1], qdd[t], u[t])) # backward euler
            self.prog.AddConstraint(self.    dynamics, lb=[0]*nq*2, ub=[0]*nq*2, vars=args)

            self.prog.AddConstraint(eq(q[t+1], q[t] + self.dt * qd[t+1])) # backward euler
            self.prog.AddConstraint(eq(qd[t+1], qd[t] + self.dt * qdd[t])) # backward euler

        # bound tether and add ground constraint
        # for t in range(self.T+1):
        #     self.prog.AddLinearConstraint(q[t,0] <= np.radians(80))
        #     self.prog.AddLinearConstraint(q[t,2] == 50)
            # self.prog.AddLinearConstraint(q[t,2] >= 40)
            # self.prog.AddLinearConstraint(q[t,2] <= 60)

        # control input constrinats
        for t in range(self.T):
            self.prog.AddLinearConstraint(u[t,1] <= 0) # you can't push the kite
            self.prog.AddLinearConstraint(u[t,1] >= -10) # limit generator torque
            self.prog.AddLinearConstraint(u[t,0] <= np.degrees(20))
            self.prog.AddLinearConstraint(u[t,0] >= -np.degrees(20))

        # trajectory starts at the current position
        self.prog.AddLinearConstraint(eq(q[0], q_0))
        self.prog.AddLinearConstraint(eq(qd[0], qd_0))

    def add_costs(self, variables, start_t=0):
        q, qd, qdd, u = variables
        q_ref, qd_ref, qdd_ref, u_ref = self.ref_traj

        # control smoothness cost
        # for t in range(self.T-1):
        #     self.prog.AddQuadraticCost((u[t+1, 0] - u[t, 0])*(u[t+1, 0] - u[t, 0]))
        #     self.prog.AddQuadraticCost((u[t+1, 1] - u[t, 1])*(u[t+1, 1] - u[t, 1]))

        # energy generation cost
        # self.prog.AddCost(qd[:-1,2].dot(u[:,1]))

        for t in range(1, self.T):
            ref_t = (t + start_t) % self.ref_T
            self.prog.AddQuadraticCost(self.gamma**(self.T - t - 1)*(q[t] - q_ref[ref_t]).T.dot(q[t] - q_ref[ref_t]))
            self.prog.AddQuadraticCost(self.gamma**(self.T - t - 1)*(qd[t] - qd_ref[ref_t]).T.dot(qd[t] - qd_ref[ref_t]))
            # NOTE: should we be penalizing acceleration
            # self.prog.AddQuadraticCost((qdd[t] - qdd_ref[t]).T.dot(qdd[t] - qdd_ref[t]))

    def set_initial_guess(self, variables, start_t):
        q, qd, qdd, u = variables
        q_ref, qd_ref, qdd_ref, u_ref = self.ref_traj

        initial_guess = np.empty(self.prog.num_vars())

        ts = (start_t + np.arange(self.T+1, dtype=int)) % self.ref_T

        self.prog.SetDecisionVariableValueInVector(q, q_ref[ts], initial_guess)
        self.prog.SetDecisionVariableValueInVector(qd, qd_ref[ts], initial_guess)
        self.prog.SetDecisionVariableValueInVector(qdd, qdd_ref[ts[:-1]], initial_guess)
        self.prog.SetDecisionVariableValueInVector(u, u_ref[ts[:-1]], initial_guess)

        return initial_guess

    def optimize(self, variables, initial_guess):
        q, qd, qdd, u = variables
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

        return q_opt, qd_opt, qdd_opt, u_opt


    def visualize_plan(self, result):
        q_opt,_,_,_ = result
        q_ref,_,_,_ = self.ref_traj
        plt.plot(*q_ref[:,[1,0]].T)
        plt.plot(*q_opt[:,[1,0]].T)

        plt.show()

    def plan(self, start_t, state):
        q_0 = state[:3]
        qd_0 = state[3:]

        variables = self.setup_optimization_variables()
        self.add_constraints(variables, q_0, qd_0)
        self.add_costs(variables, start_t)
        initial_guess = self.set_initial_guess(variables, start_t)
        result = self.optimize(variables, initial_guess)
        self.visualize_plan(result)
        return result


if __name__ == '__main__':
    q_ref = np.load(f'data/q_opt_{T}.npy')
    qd_ref = np.load(f'data/qd_opt_{T}.npy')
    qdd_ref = np.load(f'data/qdd_opt_{T}.npy')
    u_ref = np.load(f'data/u_opt_{T}.npy')
    ref_traj = (q_ref, qd_ref, qdd_ref, u_ref)

    mpc = MPC(40, ref_traj)
    q_opt, qd_opt, qdd_opt, u_opt, h_opt = mpc.plan()

