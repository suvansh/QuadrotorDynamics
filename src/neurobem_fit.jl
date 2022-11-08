# one reference trajectory
# import Pkg; Pkg.activate(joinpath(@__DIR__,".."))
# using QuadrotorDynamics
using StaticArrays
using LinearAlgebra
using Statistics
using Plots


# xs, ẋs, us, h = QuadrotorDynamics.read_neuroBEM("/Users/sanjeev/Downloads/processed_data/merged_2021-02-03-17-33-11_seg_1.csv")  # 0.9646
# xs, ẋs, us, h = QuadrotorDynamics.read_neuroBEM("/Users/sanjeev/Downloads/processed_data/merged_2021-02-05-14-01-47_seg_3.csv")  # 0.4210
# xs, ẋs, us, h = QuadrotorDynamics.read_neuroBEM("/Users/sanjeev/Downloads/processed_data/merged_2021-02-03-16-12-22_seg_3.csv")

function get_r2(aero_params, regressor_matrix, aero_vector)
    ss_res = norm(regressor_matrix * aero_params - aero_vector)^2
    aero_mean = mean(aero_vector)
    ss_tot = norm(aero_vector .- aero_mean)^2
    r2 = 1 - ss_res/ss_tot
    return r2
end

function fit_neuroBEM(quad, neurobem_path; get_r2s=false)
    # fit all trajs, get R2 for each traj
    all_xs_list = []
    all_ẋs_list = []
    all_us_list = []
    for traj_path in filter(path -> endswith(path, ".csv"), readdir(neurobem_path; join=true))
        xs, ẋs, us, _ = QuadrotorDynamics.read_neuroBEM(traj_path)
        push!(all_xs_list, xs)
        push!(all_ẋs_list, ẋs)
        push!(all_us_list, us)
    end
    catter = (x...) -> cat(x...; dims=1)
    all_xs = reduce(catter, all_xs_list)
    all_ẋs = reduce(catter, all_ẋs_list)
    all_us = reduce(catter, all_us_list)
    
    aero_params, regressor_matrix, aero_vector = QuadrotorDynamics.fit_aero(quad, eachrow(all_xs), eachrow(all_us), eachrow(all_ẋs))
    grouped_aero_params = QuadrotorDynamics.group_aero_params(aero_params)

    if get_r2s
        r2_overall = get_r2(aero_params, regressor_matrix, aero_vector)

        # get r2 for each traj
        r2s = []
        for (xs, ẋs, us) in zip(all_xs_list, all_ẋs_list, all_us_list)
            _, regressor_matrix_i, aero_vector_i = QuadrotorDynamics.fit_aero(quad, eachrow(xs), eachrow(us), eachrow(ẋs))
            push!(r2s, get_r2(aero_params, regressor_matrix_i, aero_vector_i))
        end
        return grouped_aero_params, r2_overall, r2s
    else
        return grouped_aero_params
    end    
end

function fit_neuroBEM_each(quad, neurobem_path)
    # fit each traj separately and get its R2
    r2s = []
    for traj_path in filter(path -> endswith(path, ".csv"), readdir(neurobem_path; join=true))
        xs, ẋs, us, _ = QuadrotorDynamics.read_neuroBEM(traj_path)
        aero_params, regressor_matrix, aero_vector = QuadrotorDynamics.fit_aero(quad, eachrow(xs), eachrow(us), eachrow(ẋs))
        r2 = get_r2(aero_params, regressor_matrix, aero_vector)
        push!(r2s, r2)
    end
    return r2s
end

function plot_traj(traj_path)
    xs, _, _, _ = QuadrotorDynamics.read_neuroBEM(traj_path)
    plt3d = Plots.plot(xs[:,1],xs[:,2], xs[:,3], seriestype=:scatter, markersize=5)
    display(plt3d)
end

if abspath(PROGRAM_FILE) == @__FILE__
    rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
    quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, 0.05, 0.03, rotor, false)
    neurobem_path = "/Users/sanjeev/Downloads/processed_data"
    # grouped_aero_params, r2_overall, r2s = fit_neuroBEM(quad, neurobem_path; get_r2s=true)
    r2s = fit_neuroBEM_each(quad, neurobem_path)
    # plot_traj(filter(path -> endswith(path, ".csv"), readdir(neurobem_path; join=true))[232])
end
