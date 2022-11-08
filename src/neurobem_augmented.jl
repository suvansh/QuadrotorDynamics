using StaticArrays
using LinearAlgebra
using ForwardDiff
using Optim, LineSearches
using Statistics
using JLD2

println("Running with $(Threads.nthreads()) threads")

rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(0.772, SMatrix{3,3}(Diagonal([ 0.0025, 0.0021, 0.0043 ])), -0.007, 0.315, 0.05, 0.03, rotor, false)

data_dir = "/home/sqs/Downloads/processed_data/"
save_dir = "/home/sqs/Research/QuadrotorDynamics/"
chunk_length = 10  # length of traj chunk to fit to at a time
# xs, ẋs, us, h = QuadrotorDynamics.read_neuroBEM("/Users/sanjeev/Downloads/processed_data/merged_2021-02-05-14-01-47_seg_3.csv")  # 0.4210
# xs, ẋs, us, h = QuadrotorDynamics.read_neuroBEM("/Users/sanjeev/Downloads/processed_data/merged_2021-02-03-17-33-11_seg_1.csv")  # 0.9646
traj_paths = ["merged_2021-02-23-22-33-54_seg_1.csv"

        #### LONG ONES
        "merged_2021-02-03-17-33-11_seg_1.csv"
        ## # below are randomly chosen
        "merged_2021-02-18-15-47-24_seg_2.csv"
        #"merged_2021-02-23-10-52-00_seg_2.csv"
        #### END LONG ONES

        "merged_2021-02-23-18-38-21_seg_3.csv"
        "merged_2021-02-23-18-50-03_seg_1.csv"
        "merged_2021-02-23-19-45-06_seg_3.csv"
        "merged_2021-02-23-22-41-46_seg_3.csv"
        "merged_2021-02-23-22-03-52_seg_3.csv"
        "merged_2021-02-23-22-19-12_seg_1.csv"
        "merged_2021-02-23-22-26-25_seg_1.csv"
        "merged_2021-02-23-22-26-25_seg_3.csv"
        "merged_2021-02-23-10-43-02_seg_1.csv"
        "merged_2021-02-18-17-05-14_seg_1.csv"
        "merged_2021-02-18-17-48-50_seg_1.csv"
        "merged_2021-02-23-18-43-00_seg_1.csv"
    ]
println(length(traj_paths))
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

# xs, ẋs, us, h = QuadrotorDynamics.read_neuroBEM("/Users/sanjeev/Downloads/processed_data/merged_2021-02-23-22-33-54_seg_1.csv")
n = size(xchunks[1], 2)  # x dim
m = size(uchunks[1], 2)  # u dim
p = 5  # latent z dim

catter = (arr...) -> cat(arr...; dims=1)
xs = reduce(catter, xchunks)
us = reduce(catter, uchunks)
ẋs = reduce(catter, ẋchunks)
aero_params, regressor_matrix, aero_vector = QuadrotorDynamics.fit_aero(quad, eachrow(xs), eachrow(us), eachrow(ẋs))
# grouped_aero_params = QuadrotorDynamics.group_aero_params(aero_params)



# z0 = 1e-3 * randn(p)
# A = 1e-3 * randn(p, p)
# B = 1e-3 * randn(p, n+m)
# C = 1e-3 * randn(6, p)
# total_time = h * (size(xs, 1) - 1)
function loss(x1, x2)
    diff = x1 - x2
    return dot(diff, diff)
end
function loss(t::Tuple{Number, Number})
    return loss(t[1], t[2])
end
# # xs, ẋs, _, zs, żs = QuadrotorDynamics.augmented_simulate(quad, x_list[1], us, z0, A, B, C, total_time; h=h)

# α = 1e-7
# max_iters = 50
# # println("initial loss ", loss(xf, QuadrotorDynamics.simulate(quad, x0, [u for u in eachrow(us)], total_time; h=h)[1][end]))  # test just simulate instead of augmented
# # println("non aug sim")
# # last_pos = QuadrotorDynamics.augmented_simulate(quad, x0, us, zero(z0), zero(A), zero(B), zero(C), total_time; h=h, grouped_aero_params=aero_params)[1][end]
# println("aug sim")
# xs_aug = QuadrotorDynamics.augmented_simulate(quad, x0, us, z0, A, B, C, total_time; h=h, grouped_aero_params=aero_params)[1]
# last_pos = xs_aug[end]
# # println("xf ", xf)
# println(xs_aug)


function optimize_chunk(xs, us, ẋs, h; eval=false, λ=0, ẋ_λ=0, p=3, iters=500, aero_params=nothing, initial_params=nothing, optim_type=nothing, use_force_torque=false, wrench="linear", wrench_model=nothing)
    """ nt is number of discrete time steps """
    n = size(xs, 2)  # x dim
    m = size(us, 2)  # u dim
    x0, xf = xs[1,:], xs[end,:]
    true_forces, true_torques = QuadrotorDynamics.get_force_torque(quad, xs, us, ẋs)
    time = h * (size(xs, 1) - 1)

    function augmented_objective_state(params; λ=λ, ẋ_λ=ẋ_λ)
        z0, A, B, C = params[1:p], reshape(params[p .+ (1:p^2)], (p, p)), reshape(params[(p+p^2) .+ (1:(p*(n+m)))], (p, n+m)), reshape(params[(p+p^2+p*(n+m)) .+ (1:(6*p))], (6, p))
        xs_sim, ẋs_sim = QuadrotorDynamics.augmented_simulate(quad, x0, us, z0, A, B, C, time; h=h, grouped_aero_params=aero_params, wrench=wrench, wrench_model=wrench_model)[1:2]
        if !isnothing(initial_params)
            reg = λ * norm(params[p+1:end] - initial_params[p+1:end], 2)^2
        else
            reg = 0
        end
        return mean([loss(xi, xsi) for (xi, xsi) in zip(eachrow(xs), xs_sim)]) + reg + ẋ_λ*mean([loss(ẋi, ẋsi) for (ẋi, ẋsi) in zip(eachrow(ẋs), ẋs_sim)])
    end
    function augmented_objective_force_torque(params; λ=λ, ẋ_λ=ẋ_λ)
        z0, A, B, C = params[1:p], reshape(params[p .+ (1:p^2)], (p, p)), reshape(params[(p+p^2) .+ (1:(p*(n+m)))], (p, n+m)), reshape(params[(p+p^2+p*(n+m)) .+ (1:(6*p))], (6, p))
		for i=1:size(us, 1)
			x_sim, ẋ_sim = QuadrotorDynamics.augmented_simulate(quad, xs[i,:], us[i:i,:], z0, A, B, C, h; h=h, grouped_aero_params=aero_params, wrench=wrench, wrench_model=wrench_model)[1:2]
			sim_force, sim_torque = QuadrotorDynamics.get_force_torque(quad, x_sim, us[i:i,:], ẋ_sim)
			push!(force_losses, loss(sim_force[1], true_forces[i]))
			push!(torque_losses, loss(sim_torque[1], true_torques[i]))
		end
        if !isnothing(initial_params)
            reg = λ * norm(params[p+1:end] - initial_params[p+1:end], 2)^2
        else
            reg = 0
        end
        force_loss = mean(force_losses)
        torque_loss = mean(torque_losses)
        return force_loss + 10*torque_loss + reg, force_loss, torque_loss
    end
    augmented_objective = use_force_torque ? (args...) -> augmented_objective_force_torque(args...)[1] : augmented_objective_state

	if eval
		if !isnothing(initial_params)
			return augmented_objective(initial_params; λ=0, ẋ_λ=0)
		else
			return nothing
		end
	end


    if isnothing(initial_params)
        z0 = 1e-3 * randn(p)
        A = 1e-3 * randn(p, p)
        B = 1e-3 * randn(p, n+m)
        C = 1e-3 * randn(6, p)
        params = cat(z0, reshape(A, p^2), reshape(B, p*(n+m)), reshape(C, 6*p), dims=1)
    else
        params = copy(initial_params)
    end
    if optim_type == "accelerated"
        # Default nonlinear procenditioner for `OACCEL`
        nlprecon = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
                                   linesearch=LineSearches.Static())
        # Default size of subspace that OACCEL accelerates over is `wmax = 10`
        oacc10 = OACCEL(nlprecon=nlprecon, wmax=10)

        results = optimize(augmented_objective, params, oacc10, Optim.Options(iterations=iters),
                            autodiff=:forward)
    else
        results = optimize(augmented_objective, params, LBFGS(),
                            Optim.Options(iterations=iters),
                            autodiff=:forward)
    end
    if use_force_torque
        return results, augmented_objective_force_torque(results.minimizer, λ=0)...  # return just mse without regularization too, and force and torque losses
    else
		minim = results.minimizer
		z0, A, B, C = minim[1:p], reshape(minim[p .+ (1:p^2)], (p, p)), reshape(minim[(p+p^2) .+ (1:(p*(n+m)))], (p, n+m)), reshape(minim[(p+p^2+p*(n+m)) .+ (1:(6*p))], (6, p))
		force_losses = []
		torque_losses = []
		for i=1:size(us, 1)
			x_sim, ẋ_sim = QuadrotorDynamics.augmented_simulate(quad, xs[i,:], us[i:i,:], z0, A, B, C, h; h=h, grouped_aero_params=aero_params, wrench=wrench, wrench_model=wrench_model)[1:2]
			sim_force, sim_torque = QuadrotorDynamics.get_force_torque(quad, x_sim, us[i:i,:], ẋ_sim)
			push!(force_losses, loss(sim_force[1], true_forces[i]))
			push!(torque_losses, loss(sim_torque[1], true_torques[i]))
		end
		force_mse = mean(force_losses)
		torque_mse = mean(torque_losses)
		#println("Force MSE: $(force_mse), Torque MSE: $(torque_mse)")
        return results, augmented_objective(minim, λ=0, ẋ_λ=0), force_mse, torque_mse  # return just mse without regularization too
    end
end

# NOTE: unfinished. won't work because each step gets its own z0 so too many free parameters. will overfit.
# function optimize_1step(xs, us, h; λ=0, p=3, iters=500, aero_params=nothing, initial_params=nothing)
#     n = size(xs, 2)  # x dim
#     m = size(us, 2)  # u dim
#     function augmented_objective(params; λ=λ)
#         z0, A, B, C = params[1:p], reshape(params[p .+ (1:p^2)], (p, p)), reshape(params[(p+p^2) .+ (1:(p*(n+m)))], (p, n+m)), reshape(params[(p+p^2+p*(n+m)) .+ (1:(6*p))], (6, p))
#         xs_sim = QuadrotorDynamics.augmented_simulate(quad, x0, us, z0, A, B, C, time; h=h, grouped_aero_params=aero_params)[1]
#         xz, _ = QuadrotorDynamics.augmented_step(quad, x, u, z, A, B, C; h::Float64=0.01, grouped_aero_params=nothing)
#         if !isnothing(initial_params)
#             reg = λ * norm(params[p+1:end] - initial_params[p+1:end], 2)^2
#         else
#             reg = 0
#         end
#         return mean([loss(xi, xsi) for (xi, xsi) in zip(eachrow(xs), xs_sim)]) + reg
#     end
#     if isnothing(initial_params)
#         z0 = 1e-3 * randn(p)
#         A = 1e-3 * randn(p, p)
#         B = 1e-3 * randn(p, n+m)
#         C = 1e-3 * randn(6, p)
#         params = cat(z0, reshape(A, p^2), reshape(B, p*(n+m)), reshape(C, 6*p), dims=1)
#     else
#         params = copy(initial_params)
#     end
#     results = optimize(augmented_objective, params, LBFGS(),
#                     Optim.Options(iterations=iters),
#                     autodiff=:forward)
#     return results, augmented_objective(results.minimizer, λ=0)  # return just mse without regularization too
# end

function optimize_over_time_horizon(xs, us, h, nt; p=3, iters=500, aero_params=nothing, initial_params=nothing, optim_type=nothing)
    """ nt is number of discrete time steps """
    n = size(xs, 2)  # x dim
    m = size(us, 2)  # u dim
    x0, xf = xs[1,:], xs[nt,:]
    time = h * ((isnothing(nt) ? size(xs, 1) : nt) - 1)
    function augmented_objective(params)
        z0, A, B, C = params[1:p], reshape(params[p .+ (1:p^2)], (p, p)), reshape(params[(p+p^2) .+ (1:(p*(n+m)))], (p, n+m)), reshape(params[(p+p^2+p*(n+m)) .+ (1:(6*p))], (6, p))
        xs_sim = QuadrotorDynamics.augmented_simulate(quad, x0, us, z0, A, B, C, time; h=h, grouped_aero_params=aero_params)[1]
        return mean([loss(xi, xsi) for (xi, xsi) in zip(eachrow(xs), xs_sim)])
        # return loss(xf, xs_sim[end])
    end
    if isnothing(initial_params)
        z0 = 1e-3 * randn(p)
        A = 1e-3 * randn(p, p)
        B = 1e-3 * randn(p, n+m)
        C = 1e-3 * randn(6, p)
        params = cat(z0, reshape(A, p^2), reshape(B, p*(n+m)), reshape(C, 6*p), dims=1)
    else
        params = copy(initial_params)
    end
    if optim_type == "accelerated"
        # Default nonlinear preconditioner for `OACCEL`
        nlprecon = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
                                   linesearch=LineSearches.Static())
        # Default size of subspace that OACCEL accelerates over is `wmax = 10`
        oacc10 = OACCEL(nlprecon=nlprecon, wmax=10)

        results = optimize(augmented_objective, params, oacc10, Optim.Options(iterations=iters),
                            autodiff=:forward)
    else
        results = optimize(augmented_objective, params, LBFGS(),
                        Optim.Options(iterations=iters),
                        autodiff=:forward)
    end
    return results
end

### NEW APPROACH (consensus ADMM)
use_force_torque = true
consensus_iters = 10
λ = 5e-3
ẋ_λ = 1

mean_params = nothing
all_params = []
all_mses = []
all_force_mses = []
all_torque_mses = []
iiters = 50
ft_suffix = use_force_torque ? "_force_torque" : ""
for i=1:consensus_iters
    println("iter ", i)
    global mean_params
    chunk_params = []
    chunk_mses = []
    chunk_force_mses = []
    chunk_torque_mses = []
    Threads.@threads for (xc, ẋc, uc) in chunks
        results, mse, force_mse, torque_mse = optimize_chunk(xc, uc, ẋc, h; λ=λ, ẋ_λ=ẋ_λ, p=p, iters=iiters, aero_params=aero_params, initial_params=mean_params, optim_type="accelerated", use_force_torque=use_force_torque)
        push!(chunk_force_mses, force_mse)
        push!(chunk_torque_mses, torque_mse)
        push!(chunk_mses, mse)
        push!(chunk_params, results.minimizer)  # z0 and ABC matrices for this chunk
        println("\tmse: $(mse)")
    end
	println("chunk params size")
	println(size(chunk_params))
    mean_params = mean(chunk_params)
    push!(all_params, mean_params)
    push!(all_mses, chunk_mses)
    push!(all_force_mses, chunk_force_mses)
    push!(all_torque_mses, chunk_torque_mses)
	try
		println("\taverage mse for iter $(i): $(mean(chunk_mses))")
		println("\taverage force mse for iter $(i): $(mean(chunk_force_mses))")
		println("\taverage torque mse for iter $(i): $(mean(chunk_torque_mses))")
	catch
		println("error printing average mses for iter $(i)")
	end
end
# eval all chunks with fixed final mean params
Threads.@threads for i=1:size(us, 1)
	if use_force_torque
		final_mse, final_force_mse, final_torque_mse = optimize_chunk(xs[i:i,:], us[i:i,:], ẋs[i:i,:], h; eval=true, aero_params=aero_params, initial_params=mean_params, use_force_torque=use_force_torque)
	else
		final_mse = optimize_chunk(xs[i:i,:], us[i:i,:], ẋs[i:i,:], h; eval=true, aero_params=aero_params, initial_params=mean_params, use_force_torque=use_force_torque)
		final_force_mse, final_torque_mse = nothing, nothing
	end
end

# all_mses is an array of arrays with all chunk mses for all iters of this consensus ADMM procedure
# mean_params is the final zABC matrices
# all_params is full history of mean_params

print("Saving to: ")
println(save_dir*"params_$(consensus_iters)iters_$(iiters)iiters_$(length(traj_paths))traj_lambda5e3_p$(p)_accelerated_chunk$(chunk_length)$(ft_suffix).jld2")
@save save_dir*"params_$(consensus_iters)iters_$(iiters)iiters_$(length(traj_paths))traj_lambda5e3_p$(p)_accelerated_chunk$(chunk_length)$(ft_suffix).jld2" params=all_params traj_paths=traj_paths all_mses=all_mses all_force_mses=all_force_mses all_torque_mses=all_torque_mses final_mse=final_mse final_force_mse=final_force_mse final_torque_mse=final_torque_mse



### MY OLD APPROACH (bad)
# results1 = optimize_over_time_horizon(xs, us, h, 50; iters=500, aero_params=aero_params, optim_type="accelerated")
# results2 = optimize_over_time_horizon(xs, us, h, 80; iters=500, aero_params=aero_params, initial_params=results1.minimizer, optim_type="accelerated")
# results3 = optimize_over_time_horizon(xs, us, h, 110; iters=800, aero_params=aero_params, initial_params=results2.minimizer, optim_type="accelerated")
# results4 = optimize_over_time_horizon(xs, us, h, 140; iters=1000, aero_params=aero_params, initial_params=results3.minimizer, optim_type="accelerated")
# results5 = optimize_over_time_horizon(xs, us, h, 170; iters=1000, aero_params=aero_params, initial_params=results4.minimizer)
# results6 = optimize_over_time_horizon(xs, us, h, 200; iters=1000, aero_params=aero_params, initial_params=results5.minimizer)
# params = results4.minimizer
# z0, A, B, C = params[1:p], reshape(params[p .+ (1:p^2)], (p, p)), reshape(params[(p+p^2) .+ (1:(p*(n+m)))], (p, n+m)), reshape(params[(p+p^2+p*(n+m)) .+ (1:(6*p))], (6, p))




### GRADIENT DESCENT
# for i=1:max_iters
#     global A, B, C, z0, last_pos
#     z0_grad = ForwardDiff.gradient(z0_ -> loss(xf, QuadrotorDynamics.augmented_simulate(quad, x0, us, z0_, A, B, C, total_time; h=h)[1][end]), z0)
#     A_grad = ForwardDiff.gradient(A_ -> loss(xf, QuadrotorDynamics.augmented_simulate(quad, x0, us, z0, A_, B, C, total_time; h=h)[1][end]), A)
#     B_grad = ForwardDiff.gradient(B_ -> loss(xf, QuadrotorDynamics.augmented_simulate(quad, x0, us, z0, A, B_, C, total_time; h=h)[1][end]), B)
#     C_grad = ForwardDiff.gradient(C_ -> loss(xf, QuadrotorDynamics.augmented_simulate(quad, x0, us, z0, A, B, C_, total_time; h=h)[1][end]), C)
#     # println("z0 ", z0_grad)
#     # println("A ", A_grad)
#     # println("B ", B_grad)
#     # println("C ", C_grad)
#     # println()

#     z0 -= α * z0_grad
#     A -= α * A_grad
#     B -= α * B_grad
#     C -= α * C_grad
#     last_pos = QuadrotorDynamics.augmented_simulate(quad, x0, us, z0, A, B, C, total_time; h=h)[1][end]
#     curr_loss = loss(xf, last_pos)
#     println("loss ", curr_loss)

#     # println(z0)
#     # println(A)
#     # println(B)
#     # println(C)
#     # println()

#     if curr_loss < 2 || i == max_iters
#         println(last_pos)
#         break
#     end
# end
