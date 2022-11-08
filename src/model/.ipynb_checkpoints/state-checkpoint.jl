# world frame has z pointing down
mutable struct State
    p::SVector{3,Float64}  # world frame position
    q::SVector{4,Float64}  # quaternion representing rotation from body frame to world frame
    v::SVector{3,Float64}  # world frame linear velocity
    Ω::SVector{3,Float64}  # world frame angular velocity
end

Base.:+(x::State, y::State) = State(x.p + y.p, x.q + y.q, x.v + y.v, x.Ω + y.Ω)

Base.:*(x::State, c::Number) = State(x.p * c, x.q * c, x.v * c, x.Ω * c)

Base.:*(c::Number, x::State) = x * c
