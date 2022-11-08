function augmented_dynamics(qr::QuadRotor, x, u, z, A, B, C; grouped_aero_params=nothing, return_force_torque=false, printing=false, wrench="linear", wrench_model=nothing)
    w = wrench == "linear" ? C*z : wrench_model(z)
    ẋ, addl_force, addl_torque = dynamics(qr, x, u; w=w, grouped_aero_params=grouped_aero_params, return_force_torque=true, printing=printing)
    ż = A * z + B * [x; u]
    if return_force_torque
        return [ẋ; ż], addl_force, addl_torque
    else
        return [ẋ; ż]
    end
end

function augmented_step(qr::QuadRotor, x, u, z, A, B, C; h::Float64=0.01, grouped_aero_params=nothing, return_force_torque=false, wrench="linear", wrench_model=nothing)
    xz_dot, force, torque = augmented_dynamics(qr, x, u, z, A, B, C; grouped_aero_params=grouped_aero_params, return_force_torque=true, printing=false, wrench=wrench, wrench_model=wrench_model)
    xzn = rk4([x; z], xz -> augmented_dynamics(qr, xz[1:13], u, xz[14:end], A, B, C; grouped_aero_params=grouped_aero_params, printing=false, wrench=wrench, wrench_model=wrench_model), h)
    if qr isa QuadRotor && qr.ground_check && xn[3] > 0
        xn[3] = 0
    end
    if return_force_torque
        return xzn, xz_dot, force, torque
    else
        return xzn, xz_dot
    end
end

function augmented_simulate(qr::QuadRotor, x0, us, z0, A, B, C, time::Number; h::Float64=0.01, grouped_aero_params=nothing, return_force_torque=false, controller=nothing, wrench="linear", wrench_model=nothing)
    t = 0
    i = 1
    xs = []
    ẋs = []
    us_ = []
    zs = []
    żs = []
    forces = []
    torques = []
    while t < time
        if isnothing(controller)
            u = us[i, :]
        else
            u = QuadrotorDynamics.get_control(controller, xs[end], t)
        end
        x_curr = length(xs) > 0 ? xs[end] : x0
        z_curr = length(zs) > 0 ? zs[end] : z0
        xz, xz_dot, force, torque = augmented_step(qr, x_curr, u, z_curr, A, B, C; h=h, grouped_aero_params=grouped_aero_params, return_force_torque=true, wrench=wrench, wrench_model=wrench_model)
        x, z = xz[1:13], xz[14:end]
        ẋ, ż = xz_dot[1:13], xz_dot[14:end]
        push!(xs, x)
        push!(ẋs, ẋ)
        push!(us_, u)
        push!(zs, z)
        push!(żs, ż)
        push!(forces, force)
        push!(torques, torque)
        t += h
        i += 1
    end
    if return_force_torque
        return xs, ẋs, us_, zs, żs, forces, torques
    else
        return xs, ẋs, us_, zs, żs
    end
end
