using StaticArrays
using LinearAlgebra
using ForwardDiff
using Optim
using Statistics
using JLD2

rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, 0.05, 0.03, rotor, false)

data_dir = "/Users/sanjeev/Google Drive/My Drive/CMU/Research/QuadrotorDynamics/exps/good_exps/"
chunk_length = 10  # length of traj chunk to fit to at a time
# xs, ẋs, us, h = QuadrotorDynamics.read_cf_log(data_dir * "20220812_135052/merged.csv")


traj_paths = [
    # "20220812_135052/merged.csv",
    # "20220812_135239/merged.csv"
    "20220812_155559/merged.csv"
    ]
traj_full_paths = [data_dir * traj_path for traj_path in traj_paths]
xchunks = []
ẋchunks = []
uchunks = []
h = 0
for traj_full_path in traj_full_paths
    global h
    xs, ẋs, us, h = QuadrotorDynamics.read_cf_log(traj_full_path)
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


function loss(x1, x2)
    diff = x1 - x2
    return dot(diff, diff)
end
function loss(t::Tuple{Number, Number})
    return loss(t[1], t[2])
end


function optimize_chunk(xs, us, h; λ=0, p=3, iters=500, aero_params=nothing, initial_params=nothing)
    """ nt is number of discrete time steps """
    n = size(xs, 2)  # x dim
    m = size(us, 2)  # u dim
    x0, xf = xs[1,:], xs[end,:]
    time = h * (size(xs, 1) - 1)
    function augmented_objective(params; λ=λ)
        z0, A, B, C = params[1:p], reshape(params[p .+ (1:p^2)], (p, p)), reshape(params[(p+p^2) .+ (1:(p*(n+m)))], (p, n+m)), reshape(params[(p+p^2+p*(n+m)) .+ (1:(6*p))], (6, p))
        xs_sim = QuadrotorDynamics.augmented_simulate(quad, x0, us, z0, A, B, C, time; h=h, grouped_aero_params=aero_params)[1]
        if !isnothing(initial_params)
            reg = λ * norm(params[p+1:end] - initial_params[p+1:end], 2)^2
        else
            reg = 0
        end
        return mean([loss(xi, xsi) for (xi, xsi) in zip(eachrow(xs), xs_sim)]) + reg
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
    results = optimize(augmented_objective, params, LBFGS(),
                    Optim.Options(iterations=iters),
                    autodiff=:forward)
    return results, augmented_objective(results.minimizer, λ=0)  # return just mse without regularization too
end

### NEW APPROACH (consensus ADMM)
consensus_iters = 5
λ = 5e-2

mean_params = nothing
all_params = []
all_mses = []
for i=1:consensus_iters
    println("iter ", i)
    global mean_params
    chunk_params = []
    chunk_mses = []
    # Threads.@threads
    for (xc, ẋc, uc) in chunks
        results, mse = optimize_chunk(xc, uc, h; λ=λ, p=p, iters=10, aero_params=aero_params, initial_params=mean_params)
        push!(chunk_mses, mse)
        push!(chunk_params, results.minimizer)  # z0 and ABC matrices for this chunk
    end
    mean_params = mean(chunk_params)
    push!(all_params, mean_params)
    push!(all_mses, chunk_mses)
end
# all_mses is an array of arrays with all chunk mses for all iters of this consensus ADMM procedure
# mean_params is the final zABC matrices
# all_params is full history of mean_params

# @save "params_5iters_50iiters_1traj_cf.jld2" params=mean_params
