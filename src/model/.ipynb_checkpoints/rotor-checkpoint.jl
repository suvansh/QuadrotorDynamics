struct Rotor
    flapping::Bool
    num_blades::Int
    r::Float64
    blade_chord::Float64
    flapping_hinge_offset::Float64
    blade_m::Float64
    hub_clamp_mass::Float64
    blade_root_clamp_displacement::Float64
    thrust_constant::Float64
    drag_constant::Float64
    θt::Float64  # blade tip angle
    θ0::Float64  # blade root angle
    lift_slope_gradient::Float64
end

function blade_inertia(rotor::Rotor)
    0.25 * rotor.blade_m * (rotor.r - rotor.blade_root_clamp_displacement)^2
end

function root_clamp_inertia(rotor::Rotor)
    0.25 * rotor.hub_clamp_mass * (rotor.blade_root_clamp_displacement)^2
end

function static_blade_moment(rotor::Rotor)
    0.5 * g * (rotor.hub_clamp_mass * rotor.blade_root_clamp_displacement + rotor.blade_m * rotor.r)
end

function rotor_inertia(rotor::Rotor)
    rotor.num_blades * (blade_inertia(rotor) + root_clamp_inertia(rotor))
end

function rotor_area(rotor::Rotor)
    π * rotor.r^2
end

function σ(rotor::Rotor)  # rotor solidity ratio
    rotor.blade_chord * rotor.num_blades / (π * rotor.r)
end

function θ1(rotor::Rotor)  # blade twist angle
    rotor.θt - rotor.θ0
end

function θ75(rotor::Rotor)  # 3/4 blade angle
    rotor.θ0 + 0.75 * θ1(rotor)
end

function θi(rotor::Rotor)  # blade ideal root approximation
    rotor.θt * rotor.r / rotor.flapping_hinge_offset
end

function γ(rotor::Rotor)  # lock number
    ρ * rotor.lift_slope_gradient * rotor.blade_chord * rotor.r^4 / (blade_inertia(rotor) + root_clamp_inertia(rotor))
end

function thrust_coefficient(rotor::Rotor)
    """ returns c_T = C_T⋅ρ⋅Aᵣ⋅r² such that thrust from the rotor is T = c_T⋅ω² """
    rotor.thrust_constant * ρ * rotor_area(rotor) * rotor.r^2
end

function drag_coefficient(rotor::Rotor)
    """ returns c_Q = C_Q⋅ρ⋅Aᵣ⋅r² such that drag from the rotor is Q = c_Q⋅ω² """
    rotor.drag_constant * ρ * rotor_area(rotor) * rotor.r^3
end
