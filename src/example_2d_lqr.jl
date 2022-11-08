# one reference trajectory
using StaticArrays
using LinearAlgebra
using Debugger 
using Statistics


mass = 1.0
width = 0.3 
J = 0.2*mass*width^2
height = 0.03  # height
drag_coefficient = 1e-4
lift_coefficient = 1e-4

quad = QuadRotor2D(mass, width, J, height, drag_coefficient, lift_coefficient)

# x_list = [[0., 0, 0, 0, 0, 0],
#             [0., 1, 0, 0, 0, 0],
#             [0., 0, 0, 0, 0, 0],
#             [1., 0, 0, 0, 0, 0],
#             [0., 0, 0, 0, 0, 0],
#             [1., 1, 0, 0, 0, 0],
#             [0., 0, 0, 0, 0, 0]]

# x_list = [[0., 0, 0, 0, 0, 0],
#             [0.5, 1, 0, 0, 0, 0],
#             [-0.2, 0.5, 0, 0, 0, 0],
#             [0.4, -1, 0, 0, 0, 0],
#             [-2., 0, 0, 0, 0, 0],
#             [0., 0, 0, 0, 0, 0]]

x_list = [[0., 0, 0, 0, 0, 0],
            [1., 0.1, 0, 0, 0, 0],
            [-0.2, -0.1, 0, 0, 0, 0]]

h = 0.01

time = 6  # per segment
N = time * 100 + 1  # time ÷ h


u_hover = QuadrotorDynamics.hover_control(quad)
# Bryson's rule: max position deviation ~5, max force ~50 N
Q = 1 / 5^2 * Matrix(I, 6, 6)
R = 1 / 50^2 * Matrix(I, 2, 2)

regressor_matrices = []
aero_vectors = []
ref_traj = cat(dims=1, [QuadrotorDynamics.reference_trajectory(quad, x_list[i], x_list[i+1], N; h=h) for i=1:length(x_list)-1]...)
ref_control = [u_hover for i=1:length(ref_traj)]
ref_time = range(0; length=length(ref_traj), step=h)
A, B = QuadrotorDynamics.dynamics_jacobian(quad, ref_traj[1], u_hover; h=h)
xs, ẋs, us = QuadrotorDynamics.lqr_traj(quad, A, B, Q, R, ref_traj[1], ref_traj, ref_control, ref_time; h=h)

error = norm(xs[end] - ref_traj[end])
pos_error = norm(xs[end][1:3] - ref_traj[end][1:3])

aero_params, regressor_matrix, aero_vector = QuadrotorDynamics.fit_aero(quad, xs, us, ẋs)
x_aero_mat = QuadrotorDynamics.symmetric_matrix_2d(aero_params[1:3])
z_aero_mat = QuadrotorDynamics.symmetric_matrix_2d(aero_params[4:6])

# with bias and linear terms added in
# x_aero_mat = QuadrotorDynamics.symmetric_matrix_2d(aero_params[5:7])
# z_aero_mat = QuadrotorDynamics.symmetric_matrix_2d(aero_params[12:14])

ss_res = norm(regressor_matrix*aero_params-aero_vector)^2
aero_mean = mean(aero_vector)
ss_tot = norm(aero_vector .- aero_mean)^2
r2 = 1 - ss_res/ss_tot
