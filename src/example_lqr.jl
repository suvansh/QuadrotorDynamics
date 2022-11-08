using StaticArrays
using LinearAlgebra
using Debugger 

rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, 0.05, 0.03, rotor, false)

x0 = [3., 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
xn = [0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
h = 0.01
# time = 0.5
# N = 51

time = 5
N = 501  # time ÷ h


u_hover = QuadrotorDynamics.hover_control(quad)

ref_traj = QuadrotorDynamics.reference_trajectory(quad, x0, xn, N; h=h)
ref_control = [u_hover for i=1:length(ref_traj)]
ref_time = range(0; length=length(ref_traj), step=h)
A, B = QuadrotorDynamics.dynamics_jacobian(quad, x0, u_hover; h=h)
# Bryson's rule: max position deviation ~5, max rad/sec ~1500
Q = 1 / 5^2 * Matrix(I, 13, 13)
R = 1 / 1500^2 * Matrix(I, 4, 4)
xs, ẋs, us = QuadrotorDynamics.lqr_traj(quad, A, B, Q, R, ref_traj[1], ref_traj, ref_control, ref_time; h=h)
error = norm(xs[end] - ref_traj[end])
pos_error = norm(xs[end][1:3] - ref_traj[end][1:3])
# drag_coeff = -6.780374999998809e-5
# true_drag = [drag_coeff; 0; 0; 0; 0; 0;
#                 0; 0; 0; drag_coeff; 0; 0;
#                 0; 0; 0; 0; 0; drag_coeff]
aero_params, regressor_matrix, aero_vector = QuadrotorDynamics.fit_aero(quad, xs, us, ẋs);