#!/usr/bin/python3

from matplotlib import pyplot as plt
import numpy as np

from pydrake.all import Variable, SymbolicVectorSystem
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput
from pydrake.systems.drawing import plot_system_graphviz

from drake_kite import DrakeKite
from mpc_drake import MPCDrake
from mpc import MPC
from vis import animate_trajectory

# h: original timesteps
# X: original states
# T: desired number of timesteps
def retime(h, X, dt):
    T = int(h.sum() / dt)
    ts = np.linspace(0, h.sum(), T)
    if h.shape[0] == X.shape[0]:
        return np.array([np.interp(ts, np.cumsum(h), x)\
            for x in np.atleast_2d(X.T)]).T
    elif h.shape[0]+1 == X.shape[0]:
        return np.vstack([retime(h,X[:-1],T), X[-1]])
    else:
        print('retime: h and X are different lengths')
        return None



# create simple block diagram containing our system
builder = DiagramBuilder()

kite_system = builder.AddSystem(DrakeKite())
kite_system.set_name("kite")

# TODO: Load reference trajectory
q_guess = np.load(f'data/q_opt_40.npy')
qd_guess = np.load(f'data/qd_opt_40.npy')
qdd_guess = np.load(f'data/qdd_opt_40.npy')
u_guess = np.load(f'data/u_opt_40.npy')
h_guess = np.load(f'data/h_opt_40.npy')

mpc_hz = 20
dt = 1 / mpc_hz

q_guess = retime(h_guess, q_guess, dt)
qd_guess = retime(h_guess, qd_guess, dt)
qdd_guess = retime(h_guess, qdd_guess, dt)
u_guess = retime(h_guess, u_guess, dt)

mpc_lookahead = 5

kite_mpc = MPC(mpc_lookahead, (q_guess, qd_guess, qdd_guess, u_guess), dt) # drake mathematical program for mpc
mpc_controller = MPCDrake(kite_mpc) # Drake system wrapping of our mpc
controller = builder.AddSystem(mpc_controller) # mpc after adding to system diagram
controller.set_name("controller")

builder.Connect(controller.get_output_port(0), kite_system.get_input_port(0))

builder.Connect(kite_system.get_output_port(0), controller.get_input_port(0))

logger_kite = LogOutput(kite_system.get_output_port(0), builder)
logger_kite.set_name("kite_logger")

logger_control = LogOutput(controller.get_output_port(0), builder)
logger_control.set_name("control_logger")

diagram = builder.Build()
diagram.set_name("diagram")

plt.figure()
plot_system_graphviz(diagram, max_depth=2)
plt.show()


theta = 1.04719
phi = 0.0
thetadot = 0.33634
phidot = 0.46218
x0 = [theta, phi, thetadot, phidot]

context = diagram.CreateDefaultContext()
kite_context = diagram.GetMutableSubsystemContext(kite_system, context)
kite_context.SetContinuousState(x0)

controller_context = diagram.GetMutableSubsystemContext(controller, context)
#controller_context.SetContinuousState([0])
controller_context.SetDiscreteState([0])

simulator = Simulator(diagram, context)
simulator.AdvanceTo(10)

(_, T) = logger_kite.data().shape
W = np.ones([T,3]) * np.array([6, 0, 0])[None,:]
U = logger_control.data().transpose()
X = logger_kite.data().transpose()
expected_control_times = np.load('fig8_openloop_times.npy')


plt.figure()
plt.plot(logger_kite.sample_times(), logger_kite.data().transpose())
plt.legend(['theta', 'phi', 'thetadot', 'phidot'])
plt.figure()
plt.plot(expected_control_times, olc.u)
plt.plot(logger_kite.sample_times(), U)
plt.show()

animate_trajectory(X, U, W)
