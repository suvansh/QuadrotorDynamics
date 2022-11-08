config = {
    "experiment_name": "my_example_exp",

    # data
    "chunk_length": 5,  # trajs split into chunks of this length to eval aug dynamics sim

    # model
    "flapping": False,  # True/False
    "zdim": 3,
    "aero_format": "tensor",  # tensor, compressed
    "augmented_format": "nn",  # nn, linear, none

    # nn
    "num_hidden_layers": 1,  # >= 1
    "activation": "relu",  # relu, leakurelu, gelu, tanh
    "hidden_size": 32,
    "dropout": 0.3,  # optional
    "batch_norm": True,  # True/False

    # physical constants
    "g": 9.81,  # acceleration due to gravity, m⋅s⁻²
    "ρ": 1.23,  # density of air, kg⋅m⁻³
}
