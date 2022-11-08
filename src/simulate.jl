function step(model, x, u; h::Float64=0.01, grouped_aero_params=nothing, printing=false)
    ẋ = dynamics(model, x, u; printing=printing, grouped_aero_params=grouped_aero_params)
    xn = rk4(x, x_ -> dynamics(model, x_, u; grouped_aero_params=grouped_aero_params), h)
    if model isa QuadRotor && model.ground_check && xn[3] > 0
        xn[3] = 0
    end
    return xn, ẋ
end


function simulate(model, x0, us, time::Number; h::Float64=0.01, controller=nothing, grouped_aero_params=nothing, printing=false)
    t = 0
    i = 1
    xs = [x0]
    ẋs = []
    us_ = []
    while t < time
        if isnothing(controller)
            u = us[i]
        else
            u = QuadrotorDynamics.get_control(controller, xs[end], t)
        end
        x, ẋ = step(model, xs[end], u; h=h, printing=printing, grouped_aero_params=grouped_aero_params)
        push!(xs, x)
        push!(ẋs, ẋ)
        push!(us_, u)
        t += h
        i += 1
    end
    return xs, ẋs, us_
end
