#!/usr/bin/python3

from matplotlib import pyplot as plt
import numpy as np

from pydrake.all import Variable, SymbolicVectorSystem
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput
from pydrake.systems.drawing import plot_system_graphviz

from drake_kite import DrakeKite
from ol_control import OlControl
from vis import animate_trajectory


x = Variable("x")
const_u_sys = SymbolicVectorSystem(state=[x], dynamics=[0], output=[x])

# create simple block diagram containing our system
builder = DiagramBuilder()

kite_system = builder.AddSystem(DrakeKite())
kite_system.set_name("kite")

#controller = builder.AddSystem(const_u_sys)
olc = OlControl()
controller = builder.AddSystem(olc)
controller.set_name("controller")

builder.Connect(controller.get_output_port(0), kite_system.get_input_port(0))

logger_kite = LogOutput(kite_system.get_output_port(0), builder)
logger_kite.set_name("kite_logger")

logger_control = LogOutput(controller.get_output_port(0), builder)
logger_control.set_name("control_logger")

diagram = builder.Build()
diagram.set_name("diagram")

# plt.figure()
# plot_system_graphviz(diagram, max_depth=2)
# plt.show()


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

plt.figure()
plt.plot(logger_kite.sample_times(), logger_kite.data().transpose())
plt.legend(['theta', 'phi', 'thetadot', 'phidot'])
plt.figure()
plt.plot(olc.u)
plt.show()

(_, T) = logger_kite.data().shape
W = np.ones([T,3]) * np.array([6, 0, 0])[None,:]
U = logger_control.data().transpose()
X = logger_kite.data().transpose()
animate_trajectory(X, U, W)
