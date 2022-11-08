using StaticArrays
using LinearAlgebra

rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, 0.05, rotor, false)
# x0 = State([1., 0, 0], [1., 0, 0, 0], [0., 0, 0], [0., 0, 0])
# u0 = Control(1000*ones(4))
x0 = [1., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
u0 = 1000*ones(4)
u_hover = [844.7311405540217, -844.7311405540215, 844.7311405540217, -844.7311405540215]
xs, norm_flappings, rel_norm_flappings = simulate(quad, x0, [u_hover for i=1:100], 1., h=0.01, return_flapping_norm=true)