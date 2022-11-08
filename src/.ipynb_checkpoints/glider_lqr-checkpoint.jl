# one reference trajectory
using StaticArrays
using LinearAlgebra
using Debugger 


glider = Glider(0.08, 0.0015, 0.0885, 0.0147, 0, 0.022, 0.27)

x0 = [-3.5, 0.1, 0, 0, 7, 0, 0]
xf = [0., 0, π/4, 0, 0, 0, 0]

h = 0.01

time = 6  # per segment
N = time * 100 + 1  # time ÷ h

u_guess = [40]
u0 = QuadrotorDynamics.trim_control(glider, x0, u_guess)
# Bryson's rule: max position/angle deviation ~5, max angular rate ~5 rps (10pi rad/sec)
Q = 1 / 5^2 * Matrix(I, 7, 7)
R = 1 / (10π)^2 * Matrix(I, 1, 1)

regressor_matrices = []
aero_vectors = []
ref_traj = QuadrotorDynamics.reference_trajectory(glider, x0, xf, N; h=h)
ref_control = [u0 for i=1:length(ref_traj)]
ref_time = range(0; length=length(ref_traj), step=h)
A, B = QuadrotorDynamics.dynamics_jacobian(glider, ref_traj[1], u0; h=h)
# xs, ẋs, us = QuadrotorDynamics.lqr_traj(glider, A, B, Q, R, ref_traj[1], ref_traj, ref_control, ref_time; h=h)
xs, ẋs, us = simulate(glider, x0, ref_control, time*1.0)

error = norm(xs[end] - ref_traj[end])
pos_error = norm(xs[end][1:3] - ref_traj[end][1:3])

aero_params, regressor_matrix, aero_vector = QuadrotorDynamics.fit_aero(glider, xs, us, ẋs)
x_aero_mat = QuadrotorDynamics.symmetric_matrix_2d(aero_params[1:3])
z_aero_mat = QuadrotorDynamics.symmetric_matrix_2d(aero_params[4:6])
