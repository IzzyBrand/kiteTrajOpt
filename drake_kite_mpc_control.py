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
from vis import *
from util import *

# create simple block diagram containing our system
builder = DiagramBuilder()

kite_system = builder.AddSystem(DrakeKite())
kite_system.set_name("kite")


mpc_hz = 5
dt = 1 / mpc_hz
traj_name = 'strong_opt_400'
q_ref, qd_ref, qdd_ref, u_ref, h_ref = retime(dt, *load_trajectory('%s.npy' % traj_name))
summarize(traj=(q_ref, qd_ref, qdd_ref, u_ref, h_ref), plot=False)
mpc_lookahead = 10

kite_mpc = MPC(mpc_lookahead, (q_ref, qd_ref, qdd_ref, u_ref), dt) # drake mathematical program for mpc
mpc_controller = MPCDrake(kite_mpc) # Drake system wrapping of our mpc
mpc_controller.U[0] = u_ref[0]
mpc_controller.U[1] = u_ref[0]

controller = builder.AddSystem(mpc_controller) # mpc after adding to system diagram
controller.set_name("controller")

builder.Connect(controller.get_output_port(0), kite_system.get_input_port(0))

builder.Connect(kite_system.get_output_port(0), controller.get_input_port(0))

logger_kite = LogOutput(kite_system.get_output_port(0), builder)
logger_kite.set_name("kite_logger")

logger_control = LogOutput(controller.get_output_port(0), builder)
logger_control.set_name("control_logger")

diagram = builder.Build()
diagram.set_name("Kite MPC Diagram")

#plt.figure()
#plot_system_graphviz(diagram, max_depth=2)
#plt.show()


x0 = np.concatenate([q_ref[0], qd_ref[0]])

context = diagram.CreateDefaultContext()
kite_context = diagram.GetMutableSubsystemContext(kite_system, context)
kite_context.SetContinuousState(x0)

controller_context = diagram.GetMutableSubsystemContext(controller, context)
#controller_context.SetContinuousState([0])
controller_context.SetDiscreteState([0])

simulator = Simulator(diagram, context)
simulator.AdvanceTo(120)

(_, T) = logger_kite.data().shape
W = np.ones([T,3]) * np.array([6, 0, 0])[None,:]
U = logger_control.data().transpose()
X = logger_kite.data().transpose()

q_mpc = X[:,:3]
qd_mpc = X[:,3:]
ax = plot_3d_trajectory(q_mpc, qd_mpc, show=False)
plot_3d_trajectory(q_ref, qd_ref, ax=ax)

ref_times = np.cumsum(h_ref) - h_ref[0]

plt.figure()
plt.plot(ref_times, q_ref[:,:2])
plt.plot(logger_kite.sample_times(), logger_kite.data().transpose()[:,:2])
plt.legend(['theta_ref','phi_ref', 'theta', 'phi'])
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Kite Angles')
plt.savefig('kite_angles_%s_%dhz.png' % (traj_name, mpc_hz))

plt.figure()
plt.plot(ref_times, q_ref[:,2])
plt.plot(logger_kite.sample_times(), logger_kite.data().transpose()[:,2])
plt.legend(['r_ref', 'r'])
plt.xlabel('Time (s)')
plt.ylabel('Kite Distance (m)')
plt.title('Kite Tether Length')
plt.savefig('kite_tether_len_%s_%dhz.png' % (traj_name, mpc_hz))


plt.figure()
plt.plot(ref_times, qd_ref[:,:2])
plt.plot(logger_kite.sample_times(), logger_kite.data().transpose()[:,3:5])
plt.legend(['thetadot_ref', 'phidot_ref', 'thetadot', 'phidot'])
plt.title('Angular Velocities')
plt.xlabel('Time(s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.savefig('angular_velocity_%s_%dhz.png' % (traj_name, mpc_hz))

plt.figure()
plt.plot(ref_times, qd_ref[:,2])
plt.plot(logger_kite.sample_times(), logger_kite.data().transpose()[:,5])
plt.legend(['rdot_ref', 'rdot'])
plt.title('Generator Rate')
plt.xlabel('Time(s)')
plt.ylabel('Tether rate (m/s)')
plt.savefig('tether_rate_%s_%dhz.png' % (traj_name, mpc_hz))


plt.figure()
plt.plot(ref_times, u_ref[:,0])
plt.plot(logger_kite.sample_times(), U[:,0])
plt.legend(['roll_ref', 'roll_real'])
plt.title('Kite Roll')
plt.xlabel('Time(s)')
plt.ylabel('Kite Roll (rad)')
plt.savefig('roll_%s_%dhz.png' % (traj_name, mpc_hz))

plt.figure()
plt.plot(ref_times, u_ref[:,1])
plt.plot(logger_kite.sample_times(), U[:,1])
plt.legend(['torque_ref', 'torque_real'])
plt.title('Tether Tension')
plt.xlabel('Time(s)')
plt.ylabel('Tether Tension (N)')
plt.savefig('tension_%s_%dhz.png' % (traj_name, mpc_hz))

plt.show()

# animate_trajectory(X, U, W)
