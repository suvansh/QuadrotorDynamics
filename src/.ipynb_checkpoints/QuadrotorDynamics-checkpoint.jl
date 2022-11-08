module QuadrotorDynamics

using LinearAlgebra
using StaticArrays
using ForwardDiff
using Debugger

export QuadRotor,
    QuadRotor2D,
    Glider,
    Rotor,
    State,
    Control,
    step,
    simulate,
    dynamics


include(joinpath("model", "constants.jl"))
include(joinpath("model", "rotor.jl"))
include(joinpath("model", "quadrotor.jl"))
include(joinpath("model", "glider.jl"))
include(joinpath("model", "state.jl"))
include(joinpath("model", "control.jl"))
include("utils.jl")
include("dynamics.jl")
include("augmented_dynamics.jl")
include("simulate.jl")
include("lqr_control.jl")
include("fit_aero.jl")
include("neurobem_fit.jl")

end
