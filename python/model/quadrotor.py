import torch
from torch import nn, \
    tensor, stack, \
    cos, sin, atan2, abs, norm, cross
from utils import R, A, symmetrize

class Rotor(nn.Module):
    def __init__(self,
        r,  # rotor radius
        θ0,  # blade root angle
        θt,  # blade tip angle
        lift_slope_gradient,
        blade_chord,
        blade_root_clamp_displacement,
        blade_m,
        hub_clamp_mass,
        config
    ):
        self.r = r
        self.θ0 = θ0
        self.θt = θt
        self.config = config

        # learnable parameters
        self.thrust_coefficient = nn.Parameter(torch.rand(1))
        self.drag_coefficient = nn.Parameter(torch.rand(1))

    @property
    def θ1(self):
        return self.θt - θ0

    @property
    def blade_inertia(self):
        return 0.25 * self.blade_m * (self.r - self.blade_root_clamp_displacement)**2

    @property
    def root_clamp_inertia(self):
        return 0.25 * self.hub_clamp_mass * self.blade_root_clamp_displacement**2

    @property
    def γ(self):
        return self.config["ρ"] * self.lift_slope_gradient * self.blade_chord * self.r**4 / (self.blade_inertia + self.root_clamp_inertia)


class QuadRotor(nn.Module):
    def __init__(self,
        m,  # mass
        J: torch.FloatTensor,  # moment of inertia (3x3 torch tensor)
        h,  # height of rotor above center of gravity
        d,  # distance from rotor to central axis
        rotor: Rotor,
        config
        ):
        self.m = m
        self.J = J
        self.h = h
        self.d = d
        self.rotor = rotor
        self.config = config
        self.zdim = self.config["zdim"]
        self.xdim = 13
        self.udim = 4

        self.D = torch.vstack(
            (self.d * torch.array([[1, 0, -1, 0],
                                    [0, 1, 0, -1]]),
            self.h * torch.ones(1, 4))
        )

        # learnable parameters
        if self.config["aero_format"] == "tensor":
            self.aero_params = nn.Parameter(torch.randn(3, 3, 3))
        else:
            self.aero_params = nn.Parameter(torch.randn(18))

        if self.config["augmented_format"] == "linear":
            self.A = nn.Parameter(torch.randn(self.zdim, self.zdim))
            self.B = nn.Parameter(torch.randn(self.zdim, self.xdim + self.udim))
            self.C = nn.Parameter(torch.randn(6, self.zdim))
        elif self.config["augmented_format"] == "nn":
            if self.config["num_hidden_layers"] < 1:
                raise ValueError("Minimum value of config `num_hidden_layers` is 1.")
            activation = {
                "relu": nn.ReLU,
                "leakyrelu": nn.LeakyReLU,
                "gelu": nn.GELU,
                "tanh": nn.Tanh,
            }[self.config["activation"].lower()]
            layers = [nn.Linear(self.zdim, self.config["hidden_size"]),
                        activation()]
            if self.config.get("dropout"):
                layers.append(nn.Dropout(self.config["dropout"]))
            if self.config.get("batch_norm"):
                layers.append(nn.BatchNorm1d(self.config["hidden_size"]))
            for i in range(1, self.config["num_hidden_layers"]):
                layers.append(nn.Linear(self.config["hidden_size"],
                                        self.config["hidden_size"]))
                layers.append(activation())
                if self.config.get("dropout"):
                    layers.append(nn.Dropout(self.config["dropout"]))
                if self.config.get("batch_norm"):
                    layers.append(nn.BatchNorm1d(self.config["hidden_size"]))
            layers.append(nn.Linear(self.config["hidden_size"], 6))
            self.wrench_net = nn.Sequential(*layers)



    def get_wrench(self, z):
        if z is None:
            return zeros(6)
        if self.config["augmented_format"] == "linear":
            return self.C * z
        elif self.config["augmented_format"] == "nn":
            return self.wrench_net(z)

    def aero_mul(self, vec):
        """
            aero_params represent 3 symmetric 3x3 matrices,
            each of which yields a scalar when quadratically multiplied by VEC
            (vec @ sym_mat @ vec).
            This function returns the 3 such scalars stacked as a vector.
        """
        if self.config["aero_format"] == "tensor":
            sym_tensor = self.aero_params.transpose(2, 1) + self.aero_params
        else:
            sym_tensor = torch.stack(tuple(map(symmetrize, torch.split(self.aero_params, 6))))
        return vec @ sym_tensor @ vec


    def dynamics(self, x, u, z=None):
        p, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]
        vb = A(q).T * v

        """ augmented dynamics additive wrench """
        w = self.get_wrench(z)
        w_F, w_τ = torch.split(w, 3)

        """ flapping """
        T, Q, τ = [], [], []
        for i in range(4):
            if self.config["flapping"]:
                Vr = cross(ω, self.D[:, i]) + v
                μ = norm(Vr[1:2]) / (abs(u[i]) * self.rotor.r)
                lc = Vr[3] / (abs(u[i]) * self.rotor.r)
                li = μ
                αs = atan2(lc, μ)
                j = atan2(Vr[2], Vr[1])
                J = tensor([
                    [cos(j), -sin(j)],
                    [sin(j), cos(j)]
                ])
                β = J.T @ tensor([((8/3*self.rotor.θ0 + 2 * self.rotor.θ1) * μ - 2 * lc * μ) / (1 - μ**2/2),  # longitudinal flapping
                    0])
                a1 = β[1] - 16 * ω[2]/(self.rotor.γ * abs(u[i]))
                b1 = β[2] - 16 * ω[1]/(self.rotor.γ * abs(u[i]))
                flapping = tensor([-cos(b1) * sin(a1), sin(b1), -cos(a1) * cos(b1)])
            else:
                flapping = torch.tensor([0, 0, -1])
            T.append(qr.rotor.thrust_coefficient * u[i]**2 * flapping)  # coeff absorbs ρAr^2
            Q.append(-qr.rotor.drag_coefficient * u[i] * abs(u[i]) * torch.tensor([0, 0, 1]))  # coeff absorbs ρAr^3
            τ.append(cross(T[-1], D[:, i]))
        T = torch.dstack(T)
        Q = torch.dstack(Q)
        τ = torch.dstack(τ)

        """ aero force fit """
        aero = self.aero_mul(vb)

        """ dynamics """
        ṗ = v
        q̇ = 0.5 * R(q) * tensor([0, ω])
        v̇ = self.config["g"] * tensor([0, 0, 1]) + (1 / self.m) * A(q) @ (aero + w_F + T.sum(axis=1))  # A is rotation from body to world frame
        ω̇ = qr.J \ (cross(-ω, self.J @ ω) + w_τ + τ.sum(axis=1) + Q.sum(axis=1))

        ẋ = hstack((ṗ, q̇, v̇, ω̇))
        if z is None:
            ż = None
        else:
            ż = self.augmented_dynamics(x, u, z)
        return ẋ, ż

    def augmented_dynamics(self, x, u, z):
        return A @ z + B @ hstack((x, u))
