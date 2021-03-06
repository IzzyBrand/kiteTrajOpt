#!/usr/bin/python3

import os

import matplotlib.pyplot as plt
import numpy as np
from util import plot_3d_trajectory, retime, load_trajectory

name = 'weak_opt_200'

if not os.path.exists('data/%s' % name):
    os.mkdir('data/%s' % name)

mpc_hz = 4
dt = 1 / mpc_hz
traj_name = 'weak_opt_200'
q_ref, qd_ref, qdd_ref, u_ref, h_ref = retime(dt, *load_trajectory('%s.npy' % name))


n_repeat = 3

q_ref_extended = np.tile(q_ref[:-1].T, n_repeat).T
qd_ref_extended = np.tile(qd_ref[:-1].T, n_repeat).T
qdd_ref_extended = np.tile(qdd_ref[:-1].T, n_repeat).T
u_ref_extended = np.tile(u_ref.T, n_repeat).T
h_ref_extended = np.tile(h_ref.T, n_repeat).T



sample_times = np.load('%s_sample_times.npy' % name)
kite_data = np.load('%s_logger_kite.npy' % name).T
control_data = np.load('%s_logger_control.npy' % name).T

(_, T) = kite_data.shape
W = np.ones([T,3]) * np.array([6, 0, 0])[None,:]
U = control_data
X = kite_data

q_mpc = X[:,:3]
qd_mpc = X[:,3:]
ax = plot_3d_trajectory(q_mpc, qd_mpc, show=False)
plot_3d_trajectory(q_ref, qd_ref, ax=ax, show=False)
plt.legend(['Real Trajectory', 'Reference Trajectory'])
plt.show()

#ref_times = np.cumsum(h_ref) - h_ref[0]
#ref_times = np.concatenate(([0], np.cumsum(h_ref) - h_ref[0]))
ref_times = np.concatenate(([0], np.cumsum(h_ref_extended) - h_ref[0]))

# Kite Angles
plt.figure()
plt.plot(ref_times[:-1], q_ref_extended[:,:2])
plt.plot(sample_times, kite_data[:,:2])
plt.legend(['theta_ref','phi_ref', 'theta', 'phi'],loc='upper right')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Kite Angles')
plt.savefig('data/%s/kite_angles_%s_%dhz.png' % (traj_name, traj_name, mpc_hz),bbox_inches='tight')

plt.figure()
resampled_theta = np.interp(ref_times[:-1], sample_times, kite_data[:,0])
resampled_phi = np.interp(ref_times[:-1], sample_times, kite_data[:,1])
plt.plot(ref_times[:-1], resampled_theta - q_ref_extended[:,0])
plt.plot(ref_times[:-1], resampled_phi - q_ref_extended[:,1])
plt.legend(['Theta Error', 'Phi Error'], loc='upper right')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Angular Error')
plt.savefig('data/%s/kite_angles_error_%s_%dhz.png' % (traj_name, traj_name, mpc_hz),bbox_inches='tight')

# Kite tether
plt.figure()
plt.plot(ref_times[:-1], q_ref_extended[:,2])
plt.plot(sample_times, kite_data[:,2])
plt.legend(['r_ref', 'r'], loc='upper right')
plt.xlabel('Time (s)')
plt.ylabel('Kite Distance (m)')
plt.title('Kite Tether Length')
plt.savefig('data/%s/kite_tether_len_%s_%dhz.png' % (traj_name, traj_name, mpc_hz), bbox_inches='tight')

plt.figure()
resampled_r = np.interp(ref_times[:-1], sample_times, kite_data[:,2])
plt.plot(ref_times[:-1], resampled_r - q_ref_extended[:,2])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Tether Error')
plt.savefig('data/%s/kite_tether_len_error_%s_%dhz.png' % (traj_name, traj_name, mpc_hz), bbox_inches='tight')


# Angular Velocities
plt.figure()
plt.plot(ref_times[:-1], qd_ref_extended[:,:2])
plt.plot(sample_times, kite_data[:,3:5])
plt.legend(['thetadot_ref', 'phidot_ref', 'thetadot', 'phidot'], loc='upper right')
plt.title('Angular Velocities')
plt.xlabel('Time(s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.savefig('data/%s/angular_velocity_%s_%dhz.png' % (traj_name, traj_name, mpc_hz), bbox_inches='tight')

plt.figure()
resampled_thetadot = np.interp(ref_times[:-1], sample_times, kite_data[:,3])
resampled_phidot = np.interp(ref_times[:-1], sample_times, kite_data[:,4])
plt.plot(ref_times[:-1], resampled_thetadot - qd_ref_extended[:,0])
plt.plot(ref_times[:-1], resampled_phidot - qd_ref_extended[:,1])
plt.legend(['Thetadot Error', 'Phidot Error'], loc='upper right')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Angular Velocity Error')
plt.savefig('data/%s/angular_velocity_error_%s_%dhz.png' % (traj_name, traj_name, mpc_hz), bbox_inches='tight')

# Tether Velocity
plt.figure()
plt.plot(ref_times[:-1], qd_ref_extended[:,2])
plt.plot(sample_times, kite_data[:,5])
plt.legend(['rdot_ref', 'rdot'], loc='upper right')
plt.title('Generator Rate')
plt.xlabel('Time(s)')
plt.ylabel('Tether rate (m/s)')
plt.savefig('data/%s/tether_rate_%s_%dhz.png' % (traj_name, traj_name, mpc_hz), bbox_inches='tight')

plt.figure()
resampled_rdot = np.interp(ref_times[:-1], sample_times, kite_data[:,5])
plt.plot(ref_times[:-1], resampled_rdot - qd_ref_extended[:,2])
plt.xlabel('Time (s)')
plt.ylabel('Rate (m/s)')
plt.title('Tether Velocity Error')
plt.savefig('data/%s/tether_rate_error_%s_%dhz.png' % (traj_name, traj_name, mpc_hz), bbox_inches='tight')

# Control Inputs
plt.figure()
plt.plot(ref_times[:-1], u_ref_extended[:,0])
plt.plot(sample_times, U[:,0])
plt.legend(['roll_ref', 'roll_real'], loc='upper right')
plt.title('Kite Roll')
plt.xlabel('Time(s)')
plt.ylabel('Kite Roll (rad)')
plt.savefig('data/%s/roll_%s_%dhz.png' % (traj_name, traj_name, mpc_hz), bbox_inches='tight')

plt.figure()
plt.plot(ref_times[:-1], u_ref_extended[:,1])
plt.plot(sample_times, U[:,1])
plt.legend(['torque_ref', 'torque_real'], loc='upper right')
plt.title('Tether Tension')
plt.xlabel('Time(s)')
plt.ylabel('Tether Tension (N)')
plt.savefig('data/%s/tension_%s_%dhz.png' % (traj_name, traj_name, mpc_hz), bbox_inches='tight')

# Power Output
plt.figure()
plt.plot(sample_times, kite_data[:,5] * control_data[:,1])
plt.title('Generated Power')
plt.xlabel('Time (s)')
plt.ylabel('Power (W)')
plt.savefig('data/%s/power_%s_%dhz.png' % ( traj_name, traj_name, mpc_hz), bbox_inches='tight')

plt.show()

