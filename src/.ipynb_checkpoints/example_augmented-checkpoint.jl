using StaticArrays
using LinearAlgebra
using ForwardDiff
using Debugger


rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, 0.05, 0.03, rotor, false)

x_list = [[0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0., 1.2, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]#,
        # [0.5, 0.5, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
h = 0.01
time = 2  # per segment
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
Ajac, Bjac = QuadrotorDynamics.dynamics_jacobian(quad, ref_traj[1], u_hover; h=h)
xs, ẋs, us = QuadrotorDynamics.lqr_traj(quad, Ajac, Bjac, Q, R, ref_traj[1], ref_traj, ref_control, ref_time; h=h)
catter = (x...) -> cat(x...; dims=2)
us = reduce(catter, us)'

p = 3  # z dim - hyperparameter
n = length(x_list[1])  # x dim
m = length(u_hover)  # u dim

z0 = randn(p)
A = 0.01 * randn(p, p)
B = 0.01 * randn(p, n+m)
C = 0.01 * randn(6, p)
total_time = time * (length(x_list) - 1)
function loss(x1, x2)
    diff = x1 - x2
    return dot(diff, diff)
end
# xs, ẋs, _, zs, żs = QuadrotorDynamics.augmented_simulate(quad, x_list[1], us, z0, A, B, C, total_time; h=h)

α = 1e-7
max_iters = 20
println("initial loss ", loss(x_list[end], QuadrotorDynamics.augmented_simulate(quad, x_list[1], us, z0, A, B, C, total_time; h=h)[1][end]))
println(A)
println(B)
println(C)
println()
for i=1:max_iters
    global A, B, C
    A_grad = ForwardDiff.gradient(A_ -> loss(x_list[end], QuadrotorDynamics.augmented_simulate(quad, x_list[1], us, z0, A_, B, C, total_time; h=h)[1][end]), A)
    B_grad = ForwardDiff.gradient(B_ -> loss(x_list[end], QuadrotorDynamics.augmented_simulate(quad, x_list[1], us, z0, A, B_, C, total_time; h=h)[1][end]), B)
    C_grad = ForwardDiff.gradient(C_ -> loss(x_list[end], QuadrotorDynamics.augmented_simulate(quad, x_list[1], us, z0, A, B, C_, total_time; h=h)[1][end]), C)
    println(A_grad)
    println(B_grad)
    println(C_grad)
    println()
    A -= α * A_grad
    B -= α * B_grad
    C -= α * C_grad
    last_pos = QuadrotorDynamics.augmented_simulate(quad, x_list[1], us, z0, A, B, C, total_time; h=h)[1][end]
    curr_loss = loss(x_list[end], last_pos)
    println("loss ", curr_loss)
    println(A)
    println(B)
    println(C)
    println()
    if curr_loss < 2 || i == max_iters
        println(last_pos)
        break
    end
end