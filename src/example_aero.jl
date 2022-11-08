using StaticArrays
using LinearAlgebra
include("fit_drag.jl")

rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, 0.05, 0.03, rotor, false)
x0 = [1., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
u0 = 1000*ones(4)
u_hover = [844.7311405540217, -844.7311405540215, 844.7311405540217, -844.7311405540215]
u_sin = [[200*cos(.005*t); 200*cos(.005*t); 200*cos(.005*t); 200*cos(.005*t)] .+ u_hover for t=1:2500];
us_both = cat(dims=1, [u_hover for i=1:100], u_sin);
us = u_sin;
xs, ẋs = simulate(quad, x0, us, 10., h=0.01);

aero_params, regressor_matrix, aero_vector = fit_aero(quad, xs, us, ẋs)