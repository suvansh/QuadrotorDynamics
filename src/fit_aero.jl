function dynamics_aero(qr::QuadRotor, x, u, ẋ; return_torque=false)
    p, q, v, Ω = x[1:3], x[4:7], x[8:10], x[11:13]
    ṗ, q̇, v̇, Ω̇ = ẋ[1:3], ẋ[4:7], ẋ[8:10], ẋ[11:13]
    vb = A(q)' * v  # body velocity
    # per rotor forces/torques
    D = [
        qr.rotor_distance*[1 0 -1 0;
                            0 1 0 -1];
        qr.rotor_height * ones(4)'
    ]
    T = zeros(eltype(x), 3, 4)
    Q = zeros(eltype(x), 3, 4)
    τ = zeros(eltype(x), 3, 4)
    a1s = zeros(eltype(x), 4)
    b1s = zeros(eltype(x), 4)
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
        T[:, i] = QuadrotorDynamics.thrust_coefficient(qr.rotor) * u[i]^2 * flapping
        Q[:, i] = -QuadrotorDynamics.drag_coefficient(qr.rotor) * u[i] * abs(u[i]) * QuadrotorDynamics.e3
        τ[:, i] = cross(T[:, i], D[:, i])
    end

    aero = -sum(T, dims=2)[:,1] + QuadrotorDynamics.A(q)' * qr.m * (v̇ - QuadrotorDynamics.g * QuadrotorDynamics.e3)  # body frame aero force
    torque_residual = qr.J * Ω̇ - cross(-Ω, qr.J * Ω) - sum(τ, dims=2)[:,1] - sum(Q, dims=2)[:,1]
    # println("fit aero ", aero)
    # println("vb ", vb)
    # println("original_v̇ ", v̇)

    regressor = kron(Matrix(I, 3, 3), [vb[1]^2 2*vb[1]*vb[2] 2*vb[1]*vb[3] vb[2]^2 2*vb[2]*vb[3] vb[3]^2])
    # adding lin
    # regressor = kron(Matrix(I, 3, 3), [1 v[1] v[2] v[3] v[1]^2 2*v[1]*v[2] 2*v[1]*v[3] v[2]^2 2*v[2]*v[3] v[3]^2])
    # regressor = kron(Matrix(I, 3, 3), [1 vb[1] vb[2] vb[3] vb[1]^2 2*vb[1]*vb[2] 2*vb[1]*vb[3] vb[2]^2 2*vb[2]*vb[3] vb[3]^2])

    # ṗ = v
    # q̇ = 0.5 * R(q) * [0; Ω]
    # v̇ = g * e3 + (1 / qr.m) * (drag + A(q) * sum(T, dims=2)[:,1])  # A is rotation from body to world frame
    # Ω̇ = qr.J \ (cross(-Ω, qr.J * Ω) + sum(τ, dims=2)[:,1] + sum(Q, dims=2)[:,1])
    # if qr.ground_check && p[3] > 0
    #     ṗ[3] = 0
    # end
    # return [ṗ; q̇; v̇; Ω̇]
    if return_torque
        return aero, regressor, torque_residual
    else
        return aero, regressor
    end
end


function dynamics_aero(qr::QuadRotor2D, x, u, ẋ)
    # planar quadrotor dynamics

    # unpack state
    px,pz,θ,vx,vz,ω = x
    _, _, _, ax, az, aω = ẋ

    vbx, vbz = rotate2d([vx; vz], -θ)
    aero = [ax - (1/qr.m)*(u[1] + u[2])*sin(θ);
                az - (1/qr.m)*(u[1] + u[2])*cos(θ) + g]
    regressor = kron(Matrix(I, 2, 2), [1 vbx vbz vbx^2 2*vbx*vbz vbz^2])
    # print(aero)
    # println(" fit")
    return aero, regressor
end

function dynamics_aero(glider::Glider, x, u, ẋ)
    # glider dynamics

    # unpack state
    px, pz, θ, ϕ, vx, vz, θ̇ = x
    _, _, _, _, ax, az, θ̈ = ẋ

    vbx, vbz = rotate2d([vx; vz], -θ)
    aero = [-ax * glider.m;
                (az + g) * glider.m]
    regressor = kron(Matrix(I, 2, 2), [vbx^2 2*vbx*vbz vbz^2])
    # print(aero)
    # println(" fit")
    return aero, regressor
end


function fit_aero(model, xs, us, ẋs)
    aeros = []
    regressors = []
    for (x, u, ẋ) in zip(xs, us, ẋs)
        aero, regressor = dynamics_aero(model, x, u, ẋ)
        push!(aeros, aero)
        push!(regressors, regressor)
    end
    catter = (x...) -> cat(x...; dims=1)
    regressor_matrix = reduce(catter, regressors)
    aero_vector = reduce(catter, aeros)
    # regressor_matrix = cat(dims=1, regressors...)
    # aero_vector = cat(dims=1, aeros...)
    aero_params = regressor_matrix \ aero_vector
    return aero_params, regressor_matrix, aero_vector
end
