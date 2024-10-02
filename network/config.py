"""
Config file for network
"""

import numpy as np
from box import Box

from comyx.propagation import get_noise_power
from comyx.utils import db2pow, dbm2pow, pow2db, pow2dbm


def get_cfg() -> Box:
    """
    Default configuration for the network; modify this function to return a
    custom configuration.

    Returns:
        Default configuration for the network.
    """

    cfg = Box()
    cfg.exp_seed = 3  # experiment seed

    # training params
    cfg.Ts = 200  # number of time steps
    cfg.max_height = 80  # maximum height of the UAV
    cfg.min_height = 40  # minimum height of the UAV

    cfg.n_envs = 4  # number of environments

    cfg.grid_lim = 75  # grid limit
    cfg.dist_constant = 0.75  # distance constant
    cfg.rew_thresh = 25  # reward threshold

    # grid and positions
    cfg.grid = [[-75, 75], [-75, 75]]

    cfg.bs1_pos = [-30, 30, 20]
    cfg.bs2_pos = [30, 30, 20]
    cfg.bs3_pos = [20, -30, 20]

    cfg.uc_pos = [-45, 25, 1] # cell center user
    cfg.unc_pos = [15, 30, 1] # ??????
    cfg.uf_pos = [-25, -25, 1] # edge user

    cfg.init_ris_pos = [-5, 0, 40]
    cfg.tris_loc = [15, 10, 3]

    # network params
    cfg.Pt = 15  # dBm
    cfg.Pt_lin = dbm2pow(cfg.Pt)  # Watt
    cfg.bandwidth = 10e6  # bandwidth in Hz
    cfg.frequency = 2.4e9  # carrier frequency
    cfg.temperature = 300  # Kelvin
    cfg.n_antennas = 1  # number of antennas
    cfg.n_elements = 500  # number of elements
    cfg.sigma2_dbm = get_noise_power(cfg.temperature, cfg.bandwidth, 12)  # dBm
    cfg.sigma2 = dbm2pow(cfg.sigma2_dbm)  # Watt
    cfg.radius = 30  # m
    cfg.n_users = 3  # number of users

    # metrics
    cfg.thresh_edge = 0.2  # edge threshold (bps/Hz)
    cfg.thresh_center = 0.5  # center threshold (bps/Hz)

    # channel shapes
    cfg.shape_bu = (cfg.n_antennas, cfg.n_antennas, cfg.Ts)
    cfg.shape_br = (cfg.n_elements, cfg.n_antennas, cfg.Ts)
    cfg.shape_ru = (cfg.n_elements, cfg.n_antennas, cfg.Ts)
    cfg.shape_ris = cfg.n_elements

    # fmt: off
    # fading & pathloss parameters
    cfg.K = 3  # Rician K factor

    cfg.nlos_fading_args = {"type": "rayleigh", "sigma": np.sqrt(1 / 2)}
    cfg.los_fading_args = {
        "type": "rician",
        "K": db2pow(cfg.K),
        "sigma": np.sqrt(1 / 2),
    }

    cfg.bu_pathloss_args = {
        "type": "reference", "alpha": 3, "p0": 30,
        "frequency": cfg.frequency}  # p0 is the reference power in dBm

    cfg.br_pathloss_args = {
        "type": "reference", "alpha": 2.2, "p0": 30,
        "frequency": cfg.frequency,
    }

    cfg.ru_pathloss_args = {
        "type": "reference", "alpha": 2.2, "p0": 30,
        "frequency": cfg.frequency,
    }

    cfg.edge_pathloss_args = {
        "type": "reference", "alpha": 3.3, "p0": 30,
        "frequency": cfg.frequency,
    }

    cfg.ici_pathloss_args = {
        "type": "reference", "alpha": 3.7, "p0": 30,
        "frequency": cfg.frequency,
    }
    # fmt: on

    return cfg