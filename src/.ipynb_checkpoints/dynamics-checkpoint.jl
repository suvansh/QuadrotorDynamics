function dynamics(qr::QuadRotor, x, u; w=nothing, grouped_aero_params=nothing, return_flapping_norm=false, return_force_torque=false, printing=false, ϵ=1e-7)
    p, q, v, Ω = x[1:3], x[4:7], x[8:10], x[11:13]
    vb = A(q)' * v  # body-frame velocity
    # per rotor forces/torques
    D = [
        qr.rotor_distance*[1 0 -1 0;
                            0 1 0 -1];
        qr.rotor_height * ones(4)'
    ]
    elt = Float64
    if eltype(x) != Float64
        elt = eltype(x)
    elseif eltype(u) != Float64
        elt = eltype(u)
    end
    # w is optional world-frame input wrench
    if isnothing(w)
        w_τ = zeros(elt, 3)
        w_F = zeros(elt, 3)
    else
        w_τ = w[4:6]
        w_F = w[1:3]
        # @assert norm(w_F) < ϵ  # TODO remove
    end
    T = zeros(elt, 3, 4)
    Q = zeros(elt, 3, 4)
    τ = zeros(elt, 3, 4)
    a1s = zeros(elt, 4)
    b1s = zeros(elt, 4)
    norm_flapping = zeros(elt, 4)
    rel_norm_flapping = zeros(elt, 4)
    for i=1:4
        Vr = cross(Ω, D[:,i]) + v
        μ = norm(Vr[1:2]) / (abs(u[i]) * qr.rotor.r)
        lc = Vr[3] / (abs(u[i]) * qr.rotor.r)
        li = μ
        αs = atan(lc, μ)
        j = atan(Vr[2], Vr[1])
        J = [cos(j) -sin(j);
            sin(j) cos(j)]

        # forces and torques
        if qr.rotor.flapping
            # flapping
            β = [((8/3*qr.rotor.θ0 + 2 * θ1(qr.rotor)) * μ - 2 * lc * μ) / (1 - μ^2/2);  # longitudinal flapping
                0;]
            β = J' * β
            a1s[i] = β[1] - 16/γ(qr.rotor)/abs(u[i])*Ω[2]
            b1s[i] = β[2] - 16/γ(qr.rotor)/abs(u[i])*Ω[1]
            flapping = [-cos(b1s[i]) * sin(a1s[i]); sin(b1s[i]); -cos(a1s[i]) * cos(b1s[i])]
        else
            flapping = [0; 0; -1]
        end
        T[:, i] = thrust_coefficient(qr.rotor) * u[i]^2 * flapping
        rel_norm_flapping[i] = acos(cos(a1s[i]) * cos(b1s[i]))  # distance between z axis and blade flapping vec along the unit sphere
        Q[:, i] = -drag_coefficient(qr.rotor) * u[i] * abs(u[i]) * e3
        τ[:, i] = cross(T[:, i], D[:, i])
    end
    
    # whole body force    
    # cross sectional area facing velocity vector. the 3 faces of the rect prism quadrotor are oriented along the body frame axes,
    # so the body-frame velocity coordinates give us the dot product of the area normals (elementary vectors) with the velocity vector,
    # and we divide by velocity norm to get the cos of the angle between vel and area normals. multiplied onto the nominal area,
    # we get the cross-sectional area for each face. then sum.
    A_x = abs((2 * qr.rotor_distance) * qr.rotor_height)
    A_y = abs((2 * qr.rotor_distance) * qr.rotor_height)
    A_z = (2 * qr.rotor_distance)^2
    if norm(v) < ϵ
        cross_A = A_z  # doesn't really matter since drag will be 0
    else
        vB = A(q)' * v  # velocity in body frame
        vBh = abs.(vB ./ norm(vB))  # element-wise absolute value of normalized vector velocity in body frame
        cross_A = vBh[1] * A_x + vBh[2] * A_y + vBh[3] * A_z
    end
    
    
    
    
    
    
    
    
    # DRAG test
    drag = -0.5 * ρ * qr.drag_coefficient * cross_A * norm(v) * v  # magnitude 0.5 ρ C A ||v||^2, direction opposite of v, world frame
    quad_normal = A(q) * e3  # body frame unit z vector in world frame coords
    if norm(v) < ϵ
        # no velocity so no lift
        lift = zeros(3)
    else
        lift_direction = quad_normal - dot(quad_normal, v) / dot(v, v) * v  # vector rejection of quadrotor normal onto v (drag direction)
        if norm(lift_direction) < ϵ
            # drag is in quadrotor body z direction and is equal to total aero force
            lift = zeros(3)
        else
            lift_direction ./= norm(lift_direction)
            lift = 0.5 * ρ * qr.drag_coefficient * cross_A * dot(v, v) * lift_direction
        end
    end
    grouped_aero = drag + lift  # if there's no grouped aero params, this will get used instead
    
    if !isnothing(grouped_aero_params)
        grouped_aero = [[vb[1]^2 2*vb[1]*vb[2] 2*vb[1]*vb[3] vb[2]^2 2*vb[2]*vb[3] vb[3]^2] * grouped_aero_params[1:6];
                    [vb[1]^2 2*vb[1]*vb[2] 2*vb[1]*vb[3] vb[2]^2 2*vb[2]*vb[3] vb[3]^2] * grouped_aero_params[7:12];
                    [vb[1]^2 2*vb[1]*vb[2] 2*vb[1]*vb[3] vb[2]^2 2*vb[2]*vb[3] vb[3]^2] * grouped_aero_params[13:18]]
#         if length(grouped_aero_params) == 9
#             x_aero_const, x_aero_vec, x_aero_mat, y_aero_const, y_aero_vec, y_aero_mat, z_aero_const, z_aero_vec, z_aero_mat = grouped_aero_params
#             grouped_aero = [x_aero_const + dot(vb, x_aero_vec) + dot(vb, x_aero_mat * vb);
#                     y_aero_const + dot(vb, y_aero_vec) + dot(vb, y_aero_mat * vb);
#                     z_aero_const + dot(vb, z_aero_vec) + dot(vb, z_aero_mat * vb)]
#         elseif length(grouped_aero_params) == 3
#             x_aero_mat, y_aero_mat, z_aero_mat = grouped_aero_params
            
#             grouped_aero = [dot(vb, x_aero_mat * vb);
#                             dot(vb, y_aero_mat * vb);
#                             dot(vb, z_aero_mat * vb)]
#         end
        grouped_v̇ = g * e3 + (1 / qr.m) * (grouped_aero + w_F + A(q) * sum(T, dims=2)[:,1])  # A is rotation from body to world frame
            
        if printing 
            println("grouped__aero ", grouped_aero)
            println("vb ", vb)
        end
    end
    
    
    
    # if isnothing(grouped_aero_params)
    #     drag = -0.5 * ρ * qr.drag_coefficient * cross_A * norm(v) * v  # magnitude 0.5 ρ C A ||v||^2, direction opposite of v, world frame
    #     quad_normal = A(q) * e3  # body frame unit z vector in world frame coords
    #     if norm(v) < ϵ
    #         # no velocity so no lift
    #         lift = zeros(3)
    #     else
    #         lift_direction = quad_normal - dot(quad_normal, v) / dot(v, v) * v  # vector rejection of quadrotor normal onto v (drag direction)
    #         if norm(lift_direction) < ϵ
    #             # drag is in quadrotor body z direction and is equal to total aero force
    #             lift = zeros(3)
    #         else
    #             lift_direction ./= norm(lift_direction)
    #             lift = 0.5 * ρ * qr.drag_coefficient * cross_A * dot(v, v) * lift_direction
    #         end
    #     end
    #     aero = drag + lift
    # else
    #     x_aero_const, x_aero_vec, x_aero_mat, y_aero_const, y_aero_vec, y_aero_mat, z_aero_const, z_aero_vec, z_aero_mat = grouped_aero_params
    #     aero = [x_aero_const + dot(vb, x_aero_vec) + dot(vb, x_aero_mat * vb);
    #             y_aero_const + dot(vb, y_aero_vec) + dot(vb, y_aero_mat * vb);
    #             z_aero_const + dot(vb, z_aero_vec) + dot(vb, z_aero_mat * vb)]
    # end
    if printing
        println("original_aero ", aero)
    end
    
    
    
    
    

    ṗ = v
    q̇ = 0.5 * R(q) * [0; Ω]
    v̇ = g * e3 + (1 / qr.m) * (grouped_aero + w_F + A(q) * sum(T, dims=2)[:,1])  # A is rotation from body to world frame
    if printing
        println("aero_v̇ ", v̇)
        println("grouped_v̇ ", grouped_v̇)
    end
    Ω̇ = qr.J \ (cross(-Ω, qr.J * Ω) + w_τ + sum(τ, dims=2)[:,1] + sum(Q, dims=2)[:,1])
    if qr.ground_check && p[3] > 0
        ṗ[3] = 0
    end
    if return_flapping_norm
        return [ṗ; q̇; v̇; Ω̇], norm_flapping, rel_norm_flapping
    elseif return_force_torque
        addl_force = grouped_aero + w_F  # world frame
        addl_torque = w_τ
        return [ṗ; q̇; v̇; Ω̇], addl_force, addl_torque
    else
        return [ṗ; q̇; v̇; Ω̇]
    end
end

function dynamics(qr::QuadRotor2D, x, u; ϵ=1e-7, printing=false)
    # planar quadrotor dynamics

     # unpack state
    px,pz,θ,vx,vz,ω = x
    # θ is positive when quad is tilted to the right
    
    # aero forces
    # Ax = qr.width*qr.height
    # Az = qr.width^2
    v = [vx; vz]
    vn = norm(v)
    if vn < ϵ
        drag = lift = 0
    else
        vB = rotate2d(v, θ)  # body frame velocity
        # vBh = abs.(vB / norm(vB))  # element-wise absolute value of normalized body frame velocity
        # A = vBh' * [Ax; Az]  # cross-sectional area in direction of velocity
        A = qr.width^2  # planform area
        
        α = atan(-vB[2], abs(vB[1]))
        
        
        aero_wind_x = vB[1] < 0 ? qr.drag_coefficient : -qr.drag_coefficient  # set handedness of wind frame
        aero_wind_y = vB[1] == 0 ? 0 : qr.lift_coefficient  # if velocity is in quadrotor normal direction, no lift (?)
        aero_wind = 0.5 * ρ * A * vn^2 * [aero_wind_x; aero_wind_y]  # wind frame
        aero_body = rotate2d(aero_wind, -α)  # body frame
        aero_world = rotate2d(aero_body, -θ)  # world frame
        if printing
            print(aero_world, " ", θ, " ", α, " ", vB)
            println(" sim")
        end
        
        
        drag_coefficient = 2π * α * sin(α)
        lift_coefficient = 2π * α
        drag = -0.5 * ρ * A * drag_coefficient * vn * v 
        quad_normal = rotate2d([0; 1], θ)  # quadrotor normal in world frame
        lift_direction = quad_normal - dot(quad_normal, v) / vn^2 * v  # vector rejection of quad normal onto v
        if norm(lift_direction) < ϵ  # velocity in quadrotor normal direction; no lift
            lift = zeros(2)
        else
            lift_direction /= norm(lift_direction)  # normalize
            lift = 0.5 * ρ * A * lift_coefficient * vn^2 * lift_direction
        end
        aero_world = lift + drag  # world frame
        if printing
            print(aero_world)
            # print(α)
            println(" sim")
        end
    end
    return [vx,vz,ω,(1/qr.m)*((u[1] + u[2])*sin(θ) + aero_world[1]), (1/qr.m)*((u[1] + u[2])*cos(θ) + aero_world[2]) - g, (qr.width/(2*qr.J))*(u[2]-u[1])]
end

function dynamics(gl::Glider, x, u; printing=false)
    px, pz, θ, ϕ, vx, vz, θ̇ = x
    ϕ̇ = u[1]
    
    xzwdot = [vx; vz] + (gl.l_w*θ̇) .* [-sin(θ); cos(θ)]
    xzedot = [vx; vz] + (gl.l*θ̇) .* [sin(θ); -cos(θ)] + (gl.l_e * (θ̇ + ϕ̇)) .* [sin(θ + ϕ); -cos(θ + ϕ)]
    
    α_w = θ - atan(xzwdot[2], xzwdot[1])
    α_e = θ + ϕ - atan(xzedot[2], xzedot[1])
    
    Fw = ρ * gl.S_w * sin(α_w) * dot(xzwdot, xzwdot)
    Fe = ρ * gl.S_e * sin(α_e) * dot(xzedot, xzedot)
    
    ax = (-Fw*sin(θ) - Fe*sin(θ+ϕ)) / gl.m
    az = (Fw*cos(θ) + Fe*cos(θ+ϕ)) / gl.m - g
    θ̈ = gl.J \ (Fw*gl.l_w - Fe*(gl.l*cos(ϕ) + gl.l_e))
    
    return [vx, vz, θ̇, ϕ̇, ax, az, θ̈]
end