struct Glider
    m::Float64    # mass
    J::Float64    # moment of inertia
    S_w::Float64  # surface area of wing
    S_e::Float64  # surface area of elevator
    l_w::Float64  # distance from front of glider to middle of wing
    l_e::Float64  # distance from back of glider to middle of elevator
    l::Float64    # length of glider
end