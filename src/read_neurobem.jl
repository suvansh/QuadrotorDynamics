import Pkg; Pkg.activate(".."); Pkg.instantiate()
using QuadrotorDynamics
using JLD2
using Statistics
using StaticArrays
using LinearAlgebra

data_dir = "/home/sqs/Downloads/processed_data/"

params_path = "../params_5iters_30iiters_3traj_lambda5e3_accelerated_chunk5.jld2"
chunk_length = 5

@load "../params_5iters_30iiters_3traj_lambda5e3_accelerated_chunk5.jld2"#params_path
"""
params
traj_paths
all_mses
all_force_mses
all_torque_mses
"""
all_params = params
params = all_params[end]

rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, 0.05, 0.03, rotor, false)

function loss(x1, x2)
    diff = x1 - x2
    return dot(diff, diff)
end
function loss(t::Tuple{Number, Number})
    return loss(t[1], t[2])
end


traj_full_paths = [data_dir * traj_path for traj_path in traj_paths]
xchunks = []
ẋchunks = []
uchunks = []
h = 0
for (traj_idx, traj_full_path) in enumerate(traj_full_paths)
    global h
    xs, ẋs, us, h = QuadrotorDynamics.read_neuroBEM(traj_full_path)
    println("traj_idx $traj_idx\tlength $(size(xs, 1))\ttimestep $h")
    QuadrotorDynamics.chunk_traj(xs, ẋs, us, xchunks, ẋchunks, uchunks, chunk_length)  # pushes to chunks list inplace
end
chunks = collect(zip(xchunks, ẋchunks, uchunks))


n = size(xchunks[1], 2)  # x dim
m = size(uchunks[1], 2)  # u dim
p = 3  # latent z dim

catter = (arr...) -> cat(arr...; dims=1)
xs = reduce(catter, xchunks)
us = reduce(catter, uchunks)
ẋs = reduce(catter, ẋchunks)
aero_params, regressor_matrix, aero_vector = QuadrotorDynamics.fit_aero(quad, eachrow(xs), eachrow(us), eachrow(ẋs))


force_mses = []
torque_mses = []
mses = []
println(size(chunks))
@run for (xc, ẋc, uc) in chunks
    true_forces, true_torques = QuadrotorDynamics.get_force_torque(quad, xc, uc, ẋc) 
    x0 = xc[1,:]
    z0, A, B, C = params[1:p], reshape(params[p .+ (1:p^2)], (p, p)), reshape(params[(p+p^2) .+ (1:(p*(n+m)))], (p, n+m)), reshape(params[(p+p^2+p*(n+m)) .+ (1:(6*p))], (6, p))
	time = h * (size(xc, 1) - 1)
    
    xs_sim, ẋs_sim = QuadrotorDynamics.augmented_simulate(quad, x0, uc, z0, A, B, C, time; h=h, grouped_aero_params=aero_params)[1:2]
    sim_forces, sim_torques = QuadrotorDynamics.get_force_torque(quad, xs_sim, uc, ẋs_sim)

	xdiff = [norm(xsi-xci) for (xsi, xci) in zip(xs_sim, eachrow(xc)[2:end])]  # xs_sim doesn't start with x0
	ẋdiff = [norm(ẋsi-ẋci) for (ẋsi, ẋci) in zip(ẋs_sim, eachrow(ẋc))]
    @bp
    force_mse = mean([loss(f, t_f) for (f, t_f) in zip(sim_forces, true_forces)])
    torque_mse = mean([loss(t, t_t) for (t, t_t) in zip(sim_torques, true_torques)])
    mse = mean([loss(xi, xsim) for (xi, xsim) in zip(eachrow(xc), xs_sim)])
    push!(force_mses, force_mse)
    push!(torque_mses, torque_mse)
    push!(mses, mse)
end

mean_force_mse = mean(force_mses)
mean_torque_mse = mean(torque_mses)
mean_mse = mean(mses)
