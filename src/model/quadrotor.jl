struct QuadRotor
    m::Float64  # mass
    J::SMatrix{3,3,Float64,9}  # moment of inertia
    rotor_height::Float64  # height of rotor above center of gravity
    rotor_distance::Float64  # distance from rotor to central axis
    drag_coefficient::Float64
    lift_coefficient::Float64
    rotor::Rotor  # represents one of the 4 rotors
    ground_check::Bool  # whether to prevent the quadrotor from going through the ground
end

function Γ(qr::QuadRotor)
    """ returns Γ such that [thrust; τ] = Γ⋅[ω₁²; ω₂²; ω₃²; ω₄²] """
    c_T = thrust_coefficient(qr.rotor)
    c_Q = drag_coefficient(qr.rotor)
    dc_T = qr.rotor_distance * c_T
    [c_T * ones(4)';
        0 dc_T 0 -dc_T;
        -dc_T 0 dc_T 0;
        c_Q * [-1 1 -1 1]]
end


struct QuadRotor2D
    m::Float64
    width::Float64
    J::Float64
    height::Float64
    drag_coefficient::Float64
    lift_coefficient::Float64
end