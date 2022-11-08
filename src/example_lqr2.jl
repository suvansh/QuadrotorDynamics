using StaticArrays
using LinearAlgebra
using Debugger
using Statistics


rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, 0.05, 0.03, rotor, false)

# x_list = [[3., 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1., 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [2., -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
x_list = [[0., 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1., 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
h = 0.01

time = 5  # per segment
N = 501  # time ÷ h


u_hover = QuadrotorDynamics.hover_control(quad)
# Bryson's rule: max position deviation ~5, max rad/sec ~1500
Q = 1 / 5^2 * Matrix(I, 13, 13)
R = 1 / 1500^2 * Matrix(I, 4, 4)

errors = []
pos_errors = []
regressor_matrices = []
aero_vectors = []
for i=1:length(x_list)-1
    x0, xn = x_list[i], x_list[i+1]
    ref_traj = QuadrotorDynamics.reference_trajectory(quad, x0, xn, N; h=h)
    ref_control = [u_hover for i=1:length(ref_traj)]
    ref_time = range(0; length=length(ref_traj), step=h)
    A, B = QuadrotorDynamics.dynamics_jacobian(quad, x0, u_hover; h=h)
    
    xs, ẋs, us = QuadrotorDynamics.lqr_traj(quad, A, B, Q, R, ref_traj[1], ref_traj, ref_control, ref_time; h=h)
    push!(errors, norm(xs[end] - ref_traj[end]))
    push!(pos_errors, norm(xs[end][1:3] - ref_traj[end][1:3]))
    _, regressor_matrix, aero_vector = QuadrotorDynamics.fit_aero(quad, xs, us, ẋs);
    push!(regressor_matrices, regressor_matrix)
    push!(aero_vectors, aero_vector)
end

regressor_matrix = cat(dims=1, regressor_matrices...)
aero_vector = cat(dims=1, aero_vectors...)
aero_params = regressor_matrix \ aero_vector

ss_res = norm(regressor_matrix*aero_params-aero_vector)^2
aero_mean = mean(aero_vector)
ss_tot = norm(aero_vector .- aero_mean)^2
r2 = 1 - ss_res/ss_tot