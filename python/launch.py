import itertools
import math
import argparse
import json
from rlkit.launchers.launcher_util import run_experiment
from utils import read_all_neuroBEM
from sklearn.model_selection import train_test_split
import torch


def sweep_hyperparameters(hyperparameters_dict):
    """
    adapted from https://github.com/rail-berkeley/rlkit/blob/6195cf76e4237d9d3a7e13d203bf927f82ab5858/rlkit/util/hyperparameter.py
    :param hyperparameters: A dictionary of the form
        ```
        {
            'hp_1': [value1, value2, value3],
            'hp_2': [value1, value2, value3],
            ...
        }
    """
    named_hyperparameters = []
    for name, values in hyperparameters.items():
        if type(values) is list:
            named_hyperparameters.append(
                [(name, v) for v in values]
            )
        else:  # single value
            named_hyperparameters.append(
                [(name, values)]
            )
    for tuples in itertools.product(*named_hyperparameters):
        yield dict(tuples)


def prepare_train_test():
    xs, ẋs, us, dt = read_all_neuroBEM(variant, save_path="neurobem_all.pt")
    xs_train, xs_test, ẋs_train, ẋs_test, us_train, us_test = \
        train_test_split(xs, ẋs, us, test_size=0.2, random_state=42)
    train_data = torch.utils.data.TensorDataset(xs_train, ẋs_train, us_train)
    test_data = torch.utils.data.TensorDataset(xs_test, ẋs_test, us_test)
    return train_data, test_data


def get_experiment(train_data, test_data):
    """ used to return a function that will use TRAIN_DATA and TEST_DATA
        and can be passed to run_experiment directly """
    # rotor
    r = 0.165
    θ0 = 6.8*math.pi/180
    θt = 14.6*math.pi/180
    lift_slope_gradient = 5.5
    blade_chord = 0.018
    blade_root_clamp_displacement = 0.004
    blade_m = 0.005
    hub_clamp_mass = 0.010

    # quad
    m = 0.772
    J = torch.diag([0.0025, 0.0021, 0.0043])
    h = -0.007
    d = 0.315

    def experiment(variant):
        rotor = Rotor(r, θ0, θt, lift_slope_gradient, blade_chord,
                        blade_root_clamp_displacement, blade_m, hub_clamp_mass,
                        variant)
        quad = QuadRotor(m, J, h, d, rotor, variant)
        train(quad, train_data, test_data)
    return experiment


def train(quad, train_data, test_data):
    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to json file containing hyperparameter dict")
    args = parser.parse_args()

    # get data to pass to experiment
    train_data, test_data = prepare_train_test()
    experiment = get_experiment(train_data, test_data)

    exp_name = sys.argv[1]
    with open(args.config_path) as f:
        hyperparameters_dict = json.load(f)
    exp_name = hyperparameters_dict.get("experiment_name", "exp")
    for hyperparams in sweep_hyperparameters(hyperparameters_dict):
        run_experiment(experiment,
            variant=hyperparams,
            exp_prefix=exp_name,
            mode="local",  # TODO maybe run on ec2 later?
            unpack_variant=False
        )
