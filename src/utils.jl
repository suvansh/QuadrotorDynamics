using CSVFiles
using DataFrames
using Statistics
using LinearAlgebra
using BlockDiagonals


function hat(vec::Vector)
    [0 -vec[3] vec[2];
        vec[3] 0 -vec[1];
        -vec[2] vec[1] 0]
end

""" Quaternion utils """
H = [zeros(3)'; I]
T = [1 zeros(3)'; zeros(3) -Matrix(I, 3, 3)]

function L(q)
    [q[1] -q[2:4]';
        q[2:4] q[1]*I + hat(q[2:4])]
end

function R(q)
    [q[1] -q[2:4]';
        q[2:4] q[1]*I - hat(q[2:4])]
end

function A(q)
    """ rotation matrix associated with quaternion q """
    H' * L(q) * R(q)' * H
end

function qtoQ(q)
    return H'*T*L(q)*T*L(q)*H
end

function G(q)
    G = L(q)*H
end

function rptoq(ϕ)
    (1/sqrt(1+ϕ'*ϕ))*[1; ϕ]
end

function Drptoq(ϕ)
    (1/sqrt(1+ϕ'*ϕ))*[zeros(3)'; I(3)] - (((1+ϕ'*ϕ)^(-3/2))*[1; ϕ])*ϕ'
end

function qtorp(q)
    q[2:4]/q[1]
end

function E(q)
    E = BlockDiagonal([1.0*I(3), G(q), 1.0*I(6)])
end

""" Integration, etc. """

function rk4(x, f::Function, h::Float64)
    """ updates x using f = dx/dt for time h """
    t1 = f(x)
    t2 = f(x + 0.5 * h * t1)
    t3 = f(x + 0.5 * h * t2)
    t4 = f(x + h * t3)
    xnext = x + h/6 * (t1 + 2*t2 + 2*t3 + t4)
    if length(xnext) > 6
        # normalize quat
        xnext[4:7] /= norm(xnext[4:7])
    end
    return xnext
end

function hover_control(qr::QuadRotor)
    sqrt.(QuadrotorDynamics.Γ(qr) \ [qr.m * QuadrotorDynamics.g; zeros(3)]) .* [1; -1; 1; -1]
end

function hover_control(qr::QuadRotor2D)
    fill(qr.m * g / 2, 2)
end

function trim_control(gl::Glider, x_trim, u_guess;
                        verbose = false,
                        tol = 1e-4,
                        iters = 100)
    """
        trim_control(gl::Glider, x_trim, u_guess)

    Calculate the trim controls for the Glider model, given the position and velocity specified by `x_trim`.
    Find a control close to `u_guess` that minimizes the accelerations on the plane.
    """

    function kkt_conditions(gl::Glider, x, u, u_guess, λ, B)

        # TODO: Fill out these lines
        ∇ᵤL = u - u_guess + B' * λ
        c = dynamics(gl, x, u)[5:7]

        # Return the concatenated vector
        return [∇ᵤL; c]
    end

    function kkt_jacobian(gl::Glider, x, u, u_guess, λ, B, ρ=1e-5)
        n, m, l = length(x), length(u), size(B, 1)
        # TODO: Create the KKT matrix
        H = convert(Matrix{Float64}, [Diagonal(ones(m)) B';
                B zeros(l, l)])
        Hreg = H + Diagonal([ones(m); -ones(l)])*ρ
        return Symmetric(Hreg,:L)
    end

    utrim = copy(u_guess)
    m, l = length(utrim), 3
    λ = zeros(l)
    var = [utrim; λ]
    for itr=1:iters
        utrim = var[1:m]
        λ = var[m+1:m+l]
        B = ForwardDiff.jacobian(u_ -> dynamics(gl, x_trim, u_)[5:7], utrim)

        kkt = kkt_conditions(gl, x_trim, utrim, u_guess, λ, B)
        kktj = kkt_jacobian(gl, x_trim, utrim, u_guess, λ, B)
        var -= kktj \ kkt
        res = norm(kkt)
        if verbose
            print(itr, " ")
            println(res)
        end
        if res < tol
            break
        end
    end

    return utrim
end

function slerp(qa, qb, t::Float64)
    wa, xa, ya, za = qa
    wb, xb, yb, zb = qb
    cos_half_theta = wa*wb + xa*xb + ya*yb + za*zb
    half_theta = acos(cos_half_theta)
    sin_half_theta = sqrt(1 - cos_half_theta^2)
    if abs(cos_half_theta) >= 1
        ratioA = 1
        ratioB = 0
    elseif abs(sin_half_theta) < 1e-6
        ratioA = ratioB = 0.5
    else
        ratioA = sin((1-t) * half_theta) / sin_half_theta
        ratioB = sin(t * half_theta) / sin_half_theta
    end
    return qa * ratioA + qb * ratioB
end

function dynamics_jacobian(model, x, u; h::Float64=0.01)
    A = ForwardDiff.jacobian(x_ -> step(model, x_, u; h=h)[1], x)
    B = ForwardDiff.jacobian(u_ -> step(model, x, u_; h=h)[1], u)
    return A, B
end

function reference_trajectory(qr::QuadRotor, x0, xn, N; h::Float64=0.01)
    Xref = [fill(NaN, length(x0)) for k = 1:N]
    quat_diff = L(xn[4:7]) * T * x0[4:7]
    angle_diff = 2 * acos(quat_diff[1])
    if angle_diff < 0.0001
        axis_diff = [0; 0; 1]
    else
        axis_diff = quat_diff[2:4] ./ sqrt(1 - quat_diff[1]^2)
    end
    time = h * (N - 1)
    for k=1:N-1
        Xref[k] .= [x0[1:3] + (xn[1:3] - x0[1:3]) * (k-1)/(N-1);  # interpolate position
            slerp(x0[4:7], xn[4:7], (k-1)/(N-1));  # interpolate quaternion
            (xn[1:3] - x0[1:3]) / time;  # average linear velocity
            axis_diff * angle_diff / time  # average angular velocity
        ]
    end
    Xref[end] = xn
    return Xref
end

function reference_trajectory(qr::QuadRotor2D, x0, xn, N; h::Float64=0.01)
    Xref = [fill(NaN, length(x0)) for k = 1:N]
    time = h * (N - 1)
    for k=1:N-1
        Xref[k] .= [x0[1:3] + (xn[1:3] - x0[1:3]) * (k-1)/(N-1);  # interpolate position and angle
            (xn[1:3] - x0[1:3]) / time  # average linear and angular velocity
        ]
    end
    Xref[end] = xn
    return Xref
end

function reference_trajectory(glider::Glider, x0, xn, N; h::Float64=0.01)
    Xref = [fill(NaN, length(x0)) for k = 1:N]
    time = h * (N - 1)
    for k=1:N-1
        Xref[k] .= [x0[1:4] + (xn[1:4] - x0[1:4]) * (k-1)/(N-1);  # interpolate position and angle
            (xn[1:3] - x0[1:3]) / time  # average linear and angular velocity
        ]
    end
    Xref[end] = xn
    return Xref
end

function symmetric_matrix(params)
    return [params[1:3]';
                params[2] params[4:5]';
                params[3] params[5:6]']
end

function symmetric_matrix_2d(params)
    return [params[1:2]'; params[2:3]']
end

function rotate2d(vec, θ)
    """ rotates a 2d vector VEC counterclockwise by an angle θ. """
    [cos(θ) -sin(θ);
        sin(θ) cos(θ)] * vec
end

function read_neuroBEM(csv_path)
    data = Matrix(DataFrame(load(csv_path)))
    dt = median(diff(data[:, 1]))
    # q̇ = 0.5 * R(q) * [0; Ω]
    q̇s = reduce(hcat, [0.5 * R(data[i, 8:11]) * [0; data[i, 5:7]] for i=1:size(data, 1)])'
    vw = reduce(hcat, [A(data[i, 8:11]) * data[i, 15:17] for i=1:size(data, 1)])'
    aw = reduce(hcat, [A(data[i, 8:11]) * data[i, 12:14] for i=1:size(data, 1)])'
    xs = [data[:, 18:20] data[:, 8:11] vw data[:, 5:7]]
    ẋs = [vw q̇s aw data[:, 2:4]]
    us = data[:, 21:24]
    return xs, ẋs, us, dt
end

function read_cf_log(csv_path)
    data = Matrix(DataFrame(load(csv_path)))
    dt = median(diff(data[:, 1])) / 1000  # convert to s
    q̇s = reduce(hcat, [0.5 * R(data[i, [8, 9, 10, 11]]) * [0; data[i, [5, 6, 7]]] for i=1:size(data, 1)])'
    xs = [data[:, 20:22] data[:, 8:11] data[:, 2:4] data[:, 5:7]]
    ẋs = [data[:, 2:4] q̇s data[:, 17:19] data[:, 30:32]]
    us = data[:, 26:29]
    return xs, ẋs, us, dt
end

function chunk_traj(xs, ẋs, us, xchunks_out, ẋchunks_out, uchunks_out, chunk_length; transpose=false)
    n = size(xs, 1)
    num_chunks = n ÷ chunk_length
    for i=1:num_chunks
        xchunk = xs[1+(i-1)*chunk_length:i*chunk_length, :]
        ẋchunk = ẋs[1+(i-1)*chunk_length:i*chunk_length, :]
        uchunk = us[1+(i-1)*chunk_length:i*chunk_length, :]
        if transpose
            xchunk = xchunk'
            ẋchunk = ẋchunk'
            uchunk = uchunk'
        end
        push!(xchunks_out, xchunk)
        push!(ẋchunks_out, ẋchunk)
        push!(uchunks_out, uchunk)
    end
end

function group_aero_params(aero_params)
    if length(aero_params) == 30
        x_aero_const = aero_params[1]
        x_aero_vec = aero_params[2:4]
        x_aero_mat = symmetric_matrix(aero_params[5:10])
        y_aero_const = aero_params[11]
        y_aero_vec = aero_params[12:14]
        y_aero_mat = symmetric_matrix(aero_params[15:20])
        z_aero_const = aero_params[21]
        z_aero_vec = aero_params[22:24]
        z_aero_mat = symmetric_matrix(aero_params[25:30])
        grouped_aero_params = tuple(x_aero_const, x_aero_vec, x_aero_mat, y_aero_const, y_aero_vec, y_aero_mat, z_aero_const, z_aero_vec, z_aero_mat)
    elseif length(aero_params) == 18
        x_aero_mat = symmetric_matrix(aero_params[1:6])
        y_aero_mat = symmetric_matrix(aero_params[7:12])
        z_aero_mat = symmetric_matrix(aero_params[13:18])
        grouped_aero_params = tuple(x_aero_mat, y_aero_mat, z_aero_mat)
    end
    return grouped_aero_params
end

function get_force_torque(qr, xs, us, ẋs)
    """ return forces and torques between the states by inverting the dynamics """
    forces = []
    torques = []
	if length(size(xs)) == 2
		zipper = zip(eachrow(xs), eachrow(us), eachrow(ẋs))
	else
		zipper = zip(xs, eachrow(us), ẋs)
	end
    for (x, u, ẋ) in zipper
        println("GETFORCE")
        println(size(x))
        println(size(u))
        # force, _, torque = dynamics_aero(qr, x, u, ẋ; return_torque=true)
        force = qr.m * ẋ[8:10]
        torque = qr.J * ẋ[11:13]
        push!(forces, force)
        push!(torques, torque)
    end
    return forces, torques
end
