# one reference trajectory
using StaticArrays
using LinearAlgebra
using Debugger 
using Statistics


rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, 0.05, 0.03, rotor, false)

x_list = [[0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0., 1.2, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0.5, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# x_list = [[3., 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1., 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [2., -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# x_list = [[0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1., 2, -3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [-2., 0, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
h = 0.01

time = 4  # per segment
N = time * 100 + 1  # time ÷ h


u_hover = QuadrotorDynamics.hover_control(quad)
# Bryson's rule: max position deviation ~5, max rad/sec ~1500
Q = 1 / 5^2 * Matrix(I, 13, 13)
R = 1 / 1500^2 * Matrix(I, 4, 4)

regressor_matrices = []
aero_vectors = []
ref_traj = cat(dims=1, [QuadrotorDynamics.reference_trajectory(quad, x_list[i], x_list[i+1], N; h=h) for i=1:length(x_list)-1]...)
ref_control = [u_hover for i=1:length(ref_traj)]
ref_time = range(0; length=length(ref_traj), step=h)
A, B = QuadrotorDynamics.dynamics_jacobian(quad, ref_traj[1], u_hover; h=h)
xs, ẋs, us = QuadrotorDynamics.lqr_traj(quad, A, B, Q, R, ref_traj[1], ref_traj, ref_control, ref_time; h=h)

error = norm(xs[end] - ref_traj[end])
pos_error = norm(xs[end][1:3] - ref_traj[end][1:3])

aero_params, regressor_matrix, aero_vector = QuadrotorDynamics.fit_aero(quad, xs, us, ẋs);
x_aero_mat = QuadrotorDynamics.symmetric_matrix(aero_params[1:6])
y_aero_mat = QuadrotorDynamics.symmetric_matrix(aero_params[7:12])
z_aero_mat = QuadrotorDynamics.symmetric_matrix(aero_params[13:18])

# x_aero_mat = QuadrotorDynamics.symmetric_matrix(aero_params[5:10])
# y_aero_mat = QuadrotorDynamics.symmetric_matrix(aero_params[15:20])
# z_aero_mat = QuadrotorDynamics.symmetric_matrix(aero_params[25:30])

ss_res = norm(regressor_matrix*aero_params-aero_vector)^2
aero_mean = mean(aero_vector)
ss_tot = norm(aero_vector .- aero_mean)^2
r2 = 1 - ss_res/ss_tot