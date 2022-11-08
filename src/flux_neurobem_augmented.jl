using StaticArrays
using LinearAlgebra
using ForwardDiff
using Optim, LineSearches
using Statistics
using JLD2
using Flux
using Flux.Data: DataLoader
using Debugger
using BetaML


println("Running with $(Threads.nthreads()) threads")

rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(0.772, SMatrix{3,3}(Diagonal([ 0.0025, 0.0021, 0.0043 ])), -0.007, 0.315, 0.05, 0.03, rotor, false)

# data_dir = "/home/sqs/Downloads/processed_data/"
# save_dir = "/home/sqs/Research/QuadrotorDynamics/"
data_dir = "/Users/sanjeev/Downloads/processed_data/"
save_dir = "/Users/sanjeev/Google Drive/My Drive/CMU/Research/QuadrotorDynamics/"
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
println("$(length(traj_paths)) trajs")
traj_full_paths = [data_dir * traj_path for traj_path in traj_paths]
xchunks = []
ẋchunks = []
uchunks = []
h = 0
println("reading neurobem trajs")
for (traj_idx, traj_full_path) in enumerate(traj_full_paths)
    global h
    xs, ẋs, us, h = QuadrotorDynamics.read_neuroBEM(traj_full_path)
    println("traj_idx $traj_idx\tlength $(size(xs, 1))\ttimestep $h")
    QuadrotorDynamics.chunk_traj(xs, ẋs, us, xchunks, ẋchunks, uchunks, chunk_length; transpose=true)  # pushes to chunks list inplace
end
chunks = collect(zip(xchunks, ẋchunks, uchunks))
chunks_train, chunks_test = partition(chunks, [0.75, 0.25])
train_loader = DataLoader(chunks_train; batchsize=-1, parallel=true, shuffle=true)
test_loader = DataLoader(chunks_test; batchsize=-1, parallel=true, shuffle=true)

n = size(xchunks[1], 2)  # x dim
m = size(uchunks[1], 2)  # u dim
p = 5  # latent z dim

catter = (arr...) -> cat(arr...; dims=2)
xs = reduce(catter, xchunks)'
us = reduce(catter, uchunks)'
ẋs = reduce(catter, ẋchunks)'
println(size(xs))
aero_params, regressor_matrix, aero_vector = QuadrotorDynamics.fit_aero(quad, eachrow(xs), eachrow(us), eachrow(ẋs))


function loss(x1, x2)
    diff = x1 - x2
    return dot(diff, diff)
end
function loss(t::Tuple{Number, Number})
    return loss(t[1], t[2])
end

function data_loss(params, data; wrench="linear", wrench_model=nothing)
    xc, ẋc, uc = data
    time = h * (size(xc, 1) - 1)
    z0, A, B, C = params[1:p], reshape(params[p .+ (1:p^2)], (p, p)), reshape(params[(p+p^2) .+ (1:(p*(n+m)))], (p, n+m)), reshape(params[(p+p^2+p*(n+m)) .+ (1:(6*p))], (6, p))
    @run i=1:1
        @bp
    end
	xs_sim, ẋs_sim = QuadrotorDynamics.augmented_simulate(quad, xc[:,1], uc, z0, A, B, C, time; h=h, grouped_aero_params=aero_params, wrench=wrench, wrench_model=wrench_model)[1:2]
	training_loss = mean([loss(xi, xsi) for (xi, xsi) in zip(xc, xs_sim)]) + mean([loss(ẋi, ẋsi) for (ẋi, ẋsi) in zip(ẋc, ẋs_sim)])
    return training_loss
end


function neurobem_train!(params, train_data, test_data, opt; wrench="linear", wrench_model=nothing)
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss, i
    total_train_loss = total_test_loss = 0
    for (i, data) in enumerate(train_data)
        train_loss = data_loss(params[1], data; wrench=wrench, wrench_model=wrench_model)
	    gs = gradient(params) do
            #train_loss = data_loss(params[1], data; wrench=wrench, wrench_model=wrench_model)
            return train_loss
        end
        total_train_loss += train_loss
        update!(opt, params, gs)
    end
    avg_train_loss = total_train_loss / i
    println("Train loss: $(avgl_train_loss)")
    push!(train_losses, avg_train_loss)
    for (i, data) in enumerate(test_data)
        test_loss = data_loss(params[1], data; wrench=wrench, wrench_model=wrench_model)
        total_test_loss += test_loss
    end
    avg_test_loss = total_test_loss / i
    println("Test loss: $(avg_test_loss)")
    push!(test_losses, avg_test_loss)
end

function initialize_params(p, n, m)
    z0 = 1e-3 * randn(p)
    A = 1e-3 * randn(p, p)
    B = 1e-3 * randn(p, n+m)
    C = 1e-3 * randn(6, p)  # 6 for wrench
    params = cat(z0, reshape(A, p^2), reshape(B, p*(n+m)), reshape(C, 6*p), dims=1)
    return Flux.params(params)
end

use_force_torque = false
ft_suffix = use_force_torque ? "force_torque" : ""
wrench = "linear"
η = 0.01
train_losses = []
test_losses = []
params = initialize_params(p, n, m)
opt = Flux.Adam(η)
#Flux.@epochs 5 
neurobem_train!(params, train_loader, test_loader, opt; wrench=wrench)
@run for i=1:2
    @bp
end

print("Saving to: ")
println(save_dir*"flux_params_$(wrench)_$(length(traj_paths))traj_η$(η)_chunk$(chunk_length)$(ft_suffix).jld2")
@save save_dir*"flux_params_$(wrench)_$(length(traj_paths))traj_η$(η)_chunk$(chunk_length)$(ft_suffix).jld2" params=params traj_paths=traj_paths train_losses=train_losses test_losses=test_losses
