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
from util import *

# create simple block diagram containing our system
builder = DiagramBuilder()

kite_system = builder.AddSystem(DrakeKite())
kite_system.set_name("kite")


mpc_hz = 20
dt = 1 / mpc_hz

q_ref, qd_ref, qdd_ref, u_ref, _ = retime(dt, *load_trajectory('opt_100.npy'))

mpc_lookahead = 5

kite_mpc = MPC(mpc_lookahead, (q_ref, qd_ref, qdd_ref, u_ref), dt) # drake mathematical program for mpc
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

# plt.figure()
# plot_system_graphviz(diagram, max_depth=2)
# plt.show()


x0 = np.concatenate([q_ref[0], qd_ref[0]])

context = diagram.CreateDefaultContext()
kite_context = diagram.GetMutableSubsystemContext(kite_system, context)
kite_context.SetContinuousState(x0)

controller_context = diagram.GetMutableSubsystemContext(controller, context)
#controller_context.SetContinuousState([0])
controller_context.SetDiscreteState([0])

simulator = Simulator(diagram, context)
simulator.AdvanceTo(10)

# (_, T) = logger_kite.data().shape
# W = np.ones([T,3]) * np.array([6, 0, 0])[None,:]
# U = logger_control.data().transpose()
# X = logger_kite.data().transpose()
# expected_control_times = np.load('fig8_openloop_times.npy')


# plt.figure()
# plt.plot(logger_kite.sample_times(), logger_kite.data().transpose())
# plt.legend(['theta', 'phi', 'thetadot', 'phidot'])
# plt.figure()
# plt.plot(expected_control_times, olc.u)
# plt.plot(logger_kite.sample_times(), U)
# plt.show()

# animate_trajectory(X, U, W)
