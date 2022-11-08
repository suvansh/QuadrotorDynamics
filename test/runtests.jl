using QuadrotorDynamics
using StaticArrays
using LinearAlgebra
using Test

@testset "QuadrotorDynamics.jl" begin
    rotor = Rotor(2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
    quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, rotor, false)
    x0 = State([1., 0, 0], [1., 0, 0, 0], [0., 0, 0], [0., 0, 0])
    u0 = Control([0.1, 0.1, 0.1, 0.1])
    hover_u = Control(sqrt.(QuadrotorDynamics.Γ(quad) \ [quad.m * QuadrotorDynamics.g; zeros(3)]) .* [1; -1; 1; -1])
    xs = simulate(quad, x0, [hover_u for i=1:100], 1.0, h=0.01)
    @test all(norm(x.p - x0.p) < 1e-10 for x in xs)
end
