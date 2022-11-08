using LinearAlgebra
using ForwardDiff
using BlockDiagonals
using ControlSystems

# https://github.com/RoboticExplorationLab/inertial-filter-examples/blob/main/quadrotor-vb-bias.ipynb

rotor = Rotor(false, 2, 0.165, 0.018, 0.0, 0.005, 0.010, 0.004, 0.0048, 0.0048*sqrt(0.0024), 6.8*π/180, 14.6*π/180, 5.5)
quad = QuadRotor(4., SMatrix{3,3}(Diagonal([0.082, 0.082, 0.149])), -0.007, 0.315, 0.05, 0.03, rotor, false)

u_hover = QuadrotorDynamics.hover_control(quad)
r0 = zeros(3)
q0 = [1.; 0; 0; 0]
v0 = zeros(3)
ω0 = zeros(3)
x0 = [r0; q0; v0; ω0]
h = 0.01  # timestep


A = ForwardDiff.jacobian(dx->QuadrotorDynamics.step(quad,dx,uhover;h=h)[1], x0)
B = ForwardDiff.jacobian(du->QuadrotorDynamics.step(quad,x0,du;h=h)[1], uhover)
Ã = Array(E(q0)'*A*E(q0))
B̃ = Array(E(q0)'*B);


# Cost weights
Qlqr = Array(I(Nx̃));
Rlqr = Array(.1*I(Nu));

#LQR Controller
K = dlqr(Ã,B̃,Qlqr,Rlqr);


#Feedback controller
function controller(x)
    
    q0 = x0[4:7]
    q = x[4:7]
    ϕ = qtorp(L(q0)'*q)
    
    Δx̃ = [x[1:3]-r0; ϕ; x[8:10]-v0; x[11:13]-ω0]
    
    u = uhover - K*Δx̃
end


function filter_state_prediction(xf,uf,Pf,h)
    rf = xf[1:3] #inertial frame
    qf = xf[4:7] #body to inertial
    vf = xf[8:10] #body frame
    
    ab = xf[11:13] #accel bias
    ωb = xf[14:16] #gyro bias
    
    af = uf[1:3] #body frame
    ωf = uf[4:6] #body frame
    
    Qf = qtoQ(qf) #body to inertial
    
    #IMU Prediction
    y = rptoq(-0.5*h*(ωf-ωb))
    Y = qtoQ(y) #rotation from this time step to the next one
    rp = rf + h*Qf*vf
    qp = L(qf)*rptoq(0.5*h*(ωf-ωb))
    vpk = (vf + h*(af-ab - Qf'*[0; 0; g])) #new velocity in old body frame
    vp = Y*vpk #new velocity in new body frame
    
    #Jacobian
    dvdq = H'*(L(qf)*L(H*vf)*T + R(qf)'*R(H*vf))*G(qf) #This is the derivative of Q(q)*v w.r.t. q
    dgdq = H'*(L(qf)'*L(H*[0; 0; g]) + R(qf)*R(H*[0; 0; g])*T)*G(qf) #This is the derivative of Q(q)'*g w.r.t. q
    dvdb = 0.5*h*H'*(L(y)*L(H*vpk)*T + R(y)'*R(H*vpk))*Drptoq(-0.5*h*(ωf-ωb)) #This is the derivative of Y(ω-b)*vpk w.r.t. b
    Af = [I(3) h*dvdq h*Qf zeros(3,6);
          zeros(3,3) Y zeros(3,6) -0.5*h*G(qp)'*L(qf)*Drptoq(0.5*h*(ωf-ωb));
          zeros(3,3) -h*Y*dgdq Y -h*Y dvdb;
          zeros(6,9) I(6)]
    
    #Covariance Prediction
    Pp = Af*Pf*Af' + Vf
    
    return [rp; qp; vp; ab; ωb], Pp
end

#Filter measurement Jacobian
Cf = [I(3) zeros(3,12); zeros(3,3) I(3) zeros(3,9)];

function filter_mocap_update(xf,Pf,mocap)
    rf = xf[1:3] #inertial frame
    qf = xf[4:7] #body to inertial
    vf = xf[8:10] #body frame
    
    ab = xf[11:13] #accel bias
    ωb = xf[14:16] #gyro bias
    
    rm = mocap[1:3] #body frame
    qm = mocap[4:7] #body frame
    
    z = [rm-rf; qtorp(L(qf)'*qm)] #Innovation
    S = Cf*Pf*Cf' + Wf; #Innovation variance
    Lf = Pf*Cf'*S^-1; #Kalman Filter gain
    
    Δx̃ = Lf*z
    
    xn = [rf + Δx̃[1:3]; L(qf)*rptoq(Δx̃[4:6]); vf + Δx̃[7:9]; ab + Δx̃[10:12]; ωb + Δx̃[13:15]]
    Pn = (I-Lf*Cf)*Pf*(I-Lf*Cf)' + Lf*Wf*Lf'
    
    return xn, Pn
end


#Measurement Functions

function imu_measurement(x,u)
    return [(1/m)*[zeros(2,4); kt*ones(1,4)]*u; x[11:13]] + IMU_BIAS + sqrt(Vf)[1:6,1:6]*randn(6)
end

function gyro_measurement(x)
    return x[11:13] + IMU_BIAS[4:6] + sqrt(Vf)[4:6,4:6]*randn(3)
end

function mocap_measurement(x)
    #return x[1:7]
    return [x[1:3] + sqrt(Wf)[1:3,1:3]*randn(3); L(x[4:7])*rptoq(sqrt(Wf)[4:6,4:6]*randn(3))]
end


#Simulation Setup

#true state
xhist = zeros(Nx,Nt)
xhist[:,1] = [r0+randn(3); L(q0)*rptoq(0.2*randn(3)); v0; ω0] 

#controls
uhist = zeros(Nu,Nt)

#measurements
IMU_BIAS = 0.1*randn(6)
imuhist = zeros(6,Nt)
mocaphist = zeros(7,Nt)

#filter state
xf = zeros(16,Nt)
xf[:,1] .= [x0[1:10]; zeros(6)];

#controller state
xc = zeros(Nx,Nt)
xc[:,1] .= x0;

#filter covariance
Pf = zeros(15,15,Nt)
Pf[:,:,1] .= 1.0*I(15)

Wf = 0.0001*I(6) #measurement coviariance (mocap)
Vf = Diagonal([0.0001*ones(9); 1e-6*ones(6)]); #process covariance (IMU)


#Closed-Loop Simulation

for k = 1:(Nt-1)
    
    uhist[:,k] .= controller(xc[:,k])
    
    xhist[:,k+1] .= quad_dynamics_rk4(xhist[:,k],uhist[:,k])
    
    imuhist[:,k] .= imu_measurement(xhist[:,k],uhist[:,k])
    xpred, Ppred = filter_state_prediction(xf[:,k],imuhist[:,k],Pf[:,:,k],h)
    
    #Using the predicted state from the filter in the controller.
    #This is equivalent to a one time-step delay in the Mocap measurements
    
    xc[:,k+1] .= [xpred[1:10]; gyro_measurement(xhist[:,k+1])-xpred[14:16]]
    
    mocaphist[:,k+1] .= mocap_measurement(xhist[:,k+1])
    xf[:,k+1], Pf[:,:,k+1] = filter_mocap_update(xpred,Ppred,mocaphist[:,k+1])
    
    #Using the updated state from the filter in the controller.
    #This is equivalent to no delay on the Mocap measurements
    #xc[:,k+1] = [xf[1:10,k+1]; gyro_measurement(xhist[:,k+1])-xpred[14:16]]
end
