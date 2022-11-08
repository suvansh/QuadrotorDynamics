struct LQRController
    K::Matrix{Float64}
    Xref::Vector{Vector{Float64}}
    Uref::Vector{Vector{Float64}}
    times::Vector{Float64}
end

get_k(controller, t) = searchsortedlast(controller.times, t)

function get_control(ctrl::LQRController, x, t)
    k = get_k(ctrl, t)
    u = ctrl.Uref[k] - ctrl.K * (x - ctrl.Xref[k])
    return u
end

function riccati(A,B,Q,R,Qf,N)
    # initialize the output
    n,m = size(B)
    P = [zeros(n,n) for k = 1:N]
    K = [zeros(m,n) for k = 1:N-1]
    
    # Riccati recursion
    P[end] .= Qf
    for k = reverse(1:N-1) 
        K[k] .= (R + B'P[k+1]*B)\(B'P[k+1]*A)
        P[k] .= Q + A'P[k+1]*A - A'P[k+1]*B*K[k]
    end
    
    # return the feedback gains and ctg matrices
    return K,P
end

"""
    lqr(A,B,Q,R; kwargs...)

Calculate the infinite-horizon LQR gain from dynamics Jacobians `A` and `B` and
cost matrices `Q` and `R`. Returns the infinite-horizon gain `K` and cost-to-go `P`.

# Keyword Arguments
* `P`: Provide an initial guess for the infinite-horizon gain
* `max_iters`: maximum number of iterations
* `tol`: tolerance for solve
` `verbose`: print the number of iterations
"""
function lqr(A,B,Q,R; P=Matrix(Q), tol=1e-8, max_iters=400, verbose=false)
    # initialize the output
    n,m = size(B)
    K = zeros(m,n)
    
    # TODO: calculate the infinite-horizon LQR solution
    Ks,Ps = riccati(A,B,Q,R,Q,max_iters)
    
    # return the feedback gains and ctg matrices
    return Ks[1],Ps[1]
end


function lqr_traj(model, A, B, Q, R, x0, Xref, Uref, tref; h::Float64=0.01)
    K, _ = lqr(A, B, Q, R)
    ctrl = LQRController(K, Xref, Uref, tref)
    time = h * length(Xref)
    xs, ẋs, us = simulate(model, x0, Uref, time; controller=ctrl, printing=true)
    return xs, ẋs, us
end