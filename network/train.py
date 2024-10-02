"""
Train an agent on NetworkEnv environment.

Optimization goals:
- UAV trajectory optimization
- Power allocation optimization
- STAR-RIS phase shifts & amplitude adjustments optimization
"""

import argparse
import os
import pprint
import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
import yaml
#from network import *
from network_env import NetworkEnv
from config import *
from sb3_plus import MultiOutputPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from termcolor import colored
#from utils.callbacks import DisplayCallback

cfg = get_cfg()
SEED = cfg.exp_seed
MODEL = "MOPPO"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--param-file",
        type=str,
        default=None,
        help="The path to the parameter file.",
    )
    parser.add_argument(
        "--ris-opt",
        action="store_true",
        default=False,
        help="Whether to optimize the RIS phase shifts and amplitude adjustments.",
    )
    parser.add_argument(
        "--uav-opt",
        action="store_true",
        default=False,
        help="Whether to optimize the UAV trajectory.",
    )
    parser.add_argument(
        "--ma-scheme",
        type=str,
        default="noma",
        choices=["noma", "oma"],
        help="The multiple access scheme to be used. Choices are noma and oma.",
    )
    parser.add_argument(
        "--magnitude",
        action="store_true",
        default=False,
        help="Whether to use the magnitude in channel model. (Upper-cap)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="The total number of timesteps for training.",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=1,
        help="The frequency of logging information. (In episodes)",
    )
    parser.add_argument(
        "--disp-freq",
        type=int,
        default=cfg.Ts,
        help="The frequency of displaying information.",
    )
    parser.add_argument(
        "--verbose", type=int, default=0, help="The verbosity level of the training."
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        default=False,
        help="Whether to display the progress bar.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="The seed for numpy, torch, and the gym environment.",
    )
    return parser.parse_args()


def load_hyperparams(file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # If the file doesn't exist, create it with default values
    if not os.path.exists(file_path):
        default_params = {
            "NetworkEnv-v0": {
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "policy_kwargs": {
                    "net_arch": {
                        "pi": [64, 64],
                        "vf": [64, 64]
                    }
                }
            }
        }
        with open(file_path, "w") as file:
            yaml.dump(default_params, file)
        print(f"Created default hyperparameter file at {file_path}")
    
    # Load and return the hyperparameters
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert (
        args.ris_opt or args.uav_opt
    ), "At least one optimization goal should be selected."

    # disable warnings
    warnings.filterwarnings("ignore")

    cfg = get_cfg()
    print(colored("Network Configuration", "green"))
    pprint.pprint(cfg.to_dict(), compact=True)

    gym.register(id="NetworkEnv-v0", entry_point=NetworkEnv)

    # environments
    env = gym.make(
        "NetworkEnv-v0",
        cfg=cfg,
        ris_opt=args.ris_opt,
        uav_opt=args.uav_opt,
        ma_scheme=args.ma_scheme,
        magnitude=args.magnitude,
    )

    try:
        check_env(env)
    except Exception as e:
        print(colored("Environment check failed", "red"))
        print(e)
        exit(1)

    if args.param_file is None:
        try:
            args.param_file = f"./params/{MODEL.lower()}.yaml"
            print(
                colored(
                    f"\nUsing default hyperparameter file -> {args.param_file}",
                    "yellow",
                )
            )
            hyperparams = load_hyperparams(args.param_file)["NetworkEnv-v0"]
        except FileNotFoundError:
            print(colored("\nNo hyperparameter file found", "red"))
            exit(1)
    else:
        hyperparams = load_hyperparams(args.param_file)["NetworkEnv-v0"]
        print(colored(f"\nUsing hyperparameter file -> {args.param_file}", "yellow"))

    print(colored("Hyperparameters", "green"))
    pprint.pprint(hyperparams, compact=True)

    if hyperparams.get("noise_type") is not None:
        noise_type = hyperparams["noise_type"].strip()
        noise_std = hyperparams["noise_std"]

        # Save for later (hyperparameter optimization)
        assert isinstance(
            env.action_space, spaces.Box
        ), f"Action noise can only be used with Box action space, not {env.action_space}"
        n_actions = env.action_space.shape[0]

        if "normal" in noise_type:
            hyperparams["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions),
            )
        elif "ornstein-uhlenbeck" in noise_type:
            hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions),
            )
        else:
            raise RuntimeError(f"Unknown noise type -> {noise_type}")

        print(f"Applying {noise_type} noise with std {noise_std}")

        del hyperparams["noise_type"]
        del hyperparams["noise_std"]

    for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
        if kwargs_key in hyperparams.keys() and isinstance(
            hyperparams[kwargs_key], str
        ):
            hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

    ris_sv = "ris" if args.ris_opt else "no_ris"
    uav_sv = "uav" if args.uav_opt else "no_uav"
    save_dir = f"./logs/{MODEL}_{args.ma_scheme}_{ris_sv}_{uav_sv}"
    os.makedirs(save_dir, exist_ok=True)

    env = Monitor(
        env, filename=f"{save_dir}/monitor.csv", info_keywords=("distance", "sum_rate")
    )
    print(colored("\nTraining {} agent".format(MODEL), "green"))

    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    # Train the agent
    model_class = MultiOutputPPO
    model = model_class(
        "MultiOutputPolicy",
        env,
        verbose=args.verbose,
        device=device,
        tensorboard_log=save_dir,
        seed=args.seed,
        **hyperparams,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=args.progress_bar,
        log_interval=args.log_freq,
        #callback=DisplayCallback(mean_win=cfg.Ts, disp_freq=args.disp_freq),
    )

    # Save the model
    model.save(f"{save_dir}/model")


if __name__ == "__main__":
    main()
