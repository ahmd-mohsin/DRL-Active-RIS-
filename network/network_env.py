"""
Models the Multi-Cell CoMP-NOMA Network with UAV-mounted ris.
Uses The Farama Foundation's Gymnasium to model the environment.

The core goal (optimization objective) is to maximize the sum-rate of the
network by optimizing the UAV's trajectory, the power allocation factors at the
BSs, and the amplitude adjustments and phase shifts of both the reflection and
transmission coefficients at the ris.

Each node is initialized with a single antenna, and the ris is initialized
with K elements. Furthermore, the UAV is initialized at the origin with an
altitude of H=75m.

Optimization Combinations:
- Optimal RIS & UAV Trajectory + NOMA PA
- Optimal RIS & UAV Trajectory + OMA (No PA)
- Optimal RIS & Hovering UAV + NOMA PA
- Random RIS & Hovering UAV + NOMA PA
"""

import gymnasium as gym
import numpy as np
from box import Box
from gymnasium import spaces

from comyx.network import RIS, BaseStation, UserEquipment
from comyx.utils import get_distance


from config import *
from utils import *

cfg = get_cfg()
gym.logger.set_level(40)

ACTION_MAP = {
    0: "left",
    1: "right",
    2: "up",
    3: "down",
    4: "stay",
}


class NetworkEnv(gym.Env):
    """
    Models the Multi-Cell CoMP-NOMA Network with UAV-mounted RIS.
    """

    def __init__(
        self,
        cfg: Box = cfg,
        ris_opt: bool = True,
        uav_opt: str = True,
        ma_scheme: bool = "noma",
        magnitude: bool = False,
        use_comp: bool = True,
        no_ris: bool = False,
    ):
        """
        Initializes the environment.

        Args:
            cfg: Configuration dictionary.
            ris_opt: Whether to optimize the RIS phase shifts and
              amplitude adjustments.
            uav_opt: Whether to optimize the UAV trajectory.
            ma_scheme: The multiple access scheme to be used. [noma, oma]
            magnitude: Whether to use the magnitude in channel model.
            use_comp: Whether to use CoMP.
            no_ris: Whether to ignore the RIS.
        """
        super().__init__()

        self.cfg = cfg
        self.magnitude = magnitude

        # initialize the network and useful variables
        self._init_network()
        self.rate_c = 0.0
        self.rate_nc = 0.0
        self.rate_f = 0.0
        self.time_step = 0.0
        self.oob = False  # out of bounds
        self.terminate_oob = False
        self.use_comp = use_comp
        self.no_ris = no_ris

        # optimization schemes
        self.ris_opt = ris_opt
        self.uav_opt = uav_opt
        self.ma_scheme = ma_scheme

        if self.no_ris:
            assert (
                not self.ris_opt
            ), "RIS optimization is not available when no_ris is True."

        # mean rate window
        self.mean_win = self.cfg.Ts
        self.mean_rc = np.zeros(self.mean_win)
        self.mean_rnc = np.zeros(self.mean_win)
        self.mean_rf = np.zeros(self.mean_win)
        self.dist = np.zeros(self.mean_win)
        self.pos = np.zeros((self.mean_win, 3))

        # observation space
        low = np.concatenate(
            [
                np.zeros(1),  # power allocation factors
                np.finfo(np.float32).min * np.ones(2),  # positions
                np.finfo(np.float32).min * np.ones(3),  # rates
            ]
        )
        high = np.concatenate(
            [
                np.ones(1),
                np.finfo(np.float32).max * np.ones(2),
                np.finfo(np.float32).max * np.ones(3),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Update RIS initialization to use cfg.n_elements
        self.aRIS = RIS("UAV", cfg.n_elements, position=cfg.init_ris_pos)
        self.tRIS = RIS("tRIS", cfg.n_elements, position=cfg.tris_loc)

        # Ensure phase shifts are initialized correctly
        self.aRIS.phase_shifts = np.zeros(cfg.n_elements)
        self.tRIS.phase_shifts = np.zeros(cfg.n_elements)

        # action space
        if self.ris_opt and self.uav_opt and self.ma_scheme == "noma":
            self.action_space = spaces.Dict(
                {
                    "uav": spaces.Discrete(5),  # left, right, up, down, stay
                    "bs1": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                    "aris": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(self.cfg.n_elements,),
                        dtype=np.float32,
                    ),
                    "tris": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(self.cfg.n_elements,),
                        dtype=np.float32,
                    ),
                }
            )
        elif (
            self.ris_opt and self.uav_opt and self.ma_scheme == "oma"
        ):  # no power allocation
            self.action_space = spaces.Dict(
                {
                    "uav": spaces.Discrete(5),  # left, right, up, down, stay
                    "aris": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(self.cfg.n_elements,),
                        dtype=np.float32,
                    ),
                    "tris": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(self.cfg.n_elements,),
                        dtype=np.float32,
                    ),
                }
            )
        elif (
            self.ris_opt and (not self.uav_opt) and self.ma_scheme == "noma"
        ):  # hovering UAV
            self.action_space = spaces.Dict(
                {
                    "bs1": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                    "aris": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(self.cfg.n_elements,),
                        dtype=np.float32,
                    ),
                    "tris": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(self.cfg.n_elements,),
                        dtype=np.float32,
                    ),
                }
            )
            #self.cfg.init_ris_pos = self.cfg.uav_pos
        elif (
            (not self.ris_opt) and (self.uav_opt) and self.ma_scheme == "noma"
        ):  # hovering UAV
            self.action_space = spaces.Dict(
                {
                    "uav": spaces.Discrete(5),  # left, right, up, down, stay
                    "bs1": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                }
            )
            self.cfg.init_ris_pos = self.cfg.uav_pos
        else:
            raise ValueError("Invalid optimization scheme!")

    def _init_network(self):
        """
        Initializes the network nodes and links.
        """
        # define base stations
        self.BS1 = BaseStation(
            "BS1",
            self.cfg.n_antennas,
            position=self.cfg.bs1_pos,
            radius=self.cfg.radius,
        )
        self.BS2 = BaseStation(
            "BS2",
            self.cfg.n_antennas,
            position=self.cfg.bs2_pos,
            radius=self.cfg.radius,
        )
        self.BS3 = BaseStation(
            "BS3",
            self.cfg.n_antennas,
            position=self.cfg.bs3_pos,
            radius=self.cfg.radius,
        )

        # define user equipments
        self.U_c = UserEquipment("U_c", self.cfg.n_antennas, position=self.cfg.uc_pos)
        self.U_nc = UserEquipment(
            "U_nc", self.cfg.n_antennas, position=self.cfg.unc_pos
        )
        self.U_f = UserEquipment("U_f", self.cfg.n_antennas, position=self.cfg.uf_pos)

        # define the RIS
        init_ris_pos = self.cfg.init_ris_pos
        self.aRIS = RIS("UAV", self.cfg.n_elements, position=init_ris_pos)
        self.tRIS = RIS("tRIS", self.cfg.n_elements, position=self.cfg.tris_loc)

        self.links = init_links(
            self.cfg,
            self.BS1,
            self.BS2,
            self.BS3,
            self.aRIS,
            self.tRIS,
            self.U_c,
            self.U_nc,
            self.U_f,
        )

        self.link_count = len(self.links)

    def _get_obs(self):
        """
        Returns the current observation.
        """
        pos = np.array(
            [
                self.aRIS.position[0],
                self.aRIS.position[1],
            ],
            dtype=np.float32,
        )
        rates = np.array(
            [self.rate_c, self.rate_nc, self.rate_f], dtype=np.float32
        ).squeeze()

        # Flatten pa_factor to make it a 1D array
        pa_factor = np.array([self.BS1.alpha_f], dtype=np.float32).flatten()

        return np.concatenate(
            [
                pa_factor,
                pos,
                rates,
            ]
        )

    def _get_info(self):
        """
        Returns the current information.
        """

        return {
            "time_step": self.time_step,
            "position": self.aRIS.position,
            "distance": get_distance(self.aRIS.position[:2], self.U_f.position[:2]),
            "rate_c": np.mean(self.mean_rc),
            "rate_nc": np.mean(self.mean_rnc),
            "rate_f": np.mean(self.mean_rf),
            "sum_rate": (
                np.mean(self.mean_rc) + np.mean(self.mean_rnc) + np.mean(self.mean_rf)
            ),
        }

    def _get_reward(self):
        """
        Returns the current reward.
        """
        # calculate the sum-rate
        get_sinr(
            self.U_c,
            self.U_nc,
            self.U_f,
            self.BS1,
            self.BS2,
            self.BS3,
            self.channels,
            self.cfg,
            ma_scheme=self.ma_scheme,
        )
        self.rate_c, self.rate_nc, self.rate_f = get_rates(
            self.U_c, self.U_nc, self.U_f, ma_scheme=self.ma_scheme
        )
        sum_rate = self.rate_c + self.rate_nc + self.rate_f

        self.mean_rc[self.time_step % self.mean_win] = self.rate_c
        self.mean_rnc[self.time_step % self.mean_win] = self.rate_nc
        self.mean_rf[self.time_step % self.mean_win] = self.rate_f

        r_sum = sum_rate

        # distance incentive
        dist_uav_uf = get_distance(self.aRIS.position[:2], self.U_f.position[:2])

        dist_inct = 0
        if dist_uav_uf < self.cfg.rew_thresh:
            dist_inct = self.cfg.dist_constant / (dist_uav_uf / self.cfg.grid_lim)

        dist_inct = np.clip(dist_inct, 0, 15)

        if self.terminate_oob:
            r_sum -= 7

        return float(r_sum + dist_inct)

    def _step_ris_uav_noma(self, action):
        # UAV actions
        # Update RIS actions to use cfg.n_elements
        self.aRIS.phase_shifts = rescale(action["aris"][:self.cfg.n_elements], -np.pi, np.pi)
        self.tRIS.phase_shifts = rescale(action["tris"][:self.cfg.n_elements], -np.pi, np.pi)
        uav_action = ACTION_MAP[action["uav"]]
        x, y, z = self.aRIS.position

        if uav_action == "left":
            next_pos = (x - 1, y, z)
        elif uav_action == "right":
            next_pos = (x + 1, y, z)
        elif uav_action == "up":
            next_pos = (x, y + 1, z)
        elif uav_action == "down":
            next_pos = (x, y - 1, z)
        else:
            next_pos = (x, y, z)

        if not (  # do not move out of the grid
            (x <= self.cfg.grid[0][0])
            or (y <= self.cfg.grid[1][0])
            or (x >= self.cfg.grid[0][1])
            or (y >= self.cfg.grid[1][1])
        ):
            self.aRIS.position = next_pos
            self.oob = False
        else:
            self.oob = True
            self.terminate_oob = True

        self.dist[self.time_step % self.mean_win] = get_distance(
            self.aRIS.position[:2], self.U_f.position[:2]
        )
        self.pos[self.time_step % self.mean_win] = self.aRIS.position

        # BS actions
        self.BS1.alpha_f = rescale(action["bs1"], 0.5, 1.0)

        # RIS actions
        self.aRIS.phase_shifts = rescale(action["aris"], -np.pi, np.pi)
        self.tRIS.phase_shifts = rescale(action["tris"], -np.pi, np.pi)

    def _step_ris_uav_oma(self, action):
        # UAV actions
        # Update RIS actions to use cfg.n_elements
        self.aRIS.phase_shifts = rescale(action["aris"][:self.cfg.n_elements], -np.pi, np.pi)
        self.tRIS.phase_shifts = rescale(action["tris"][:self.cfg.n_elements], -np.pi, np.pi)
        uav_action = ACTION_MAP[action["uav"]]
        x, y, z = self.aRIS.position

        if uav_action == "left":
            next_pos = (x - 1, y, z)
        elif uav_action == "right":
            next_pos = (x + 1, y, z)
        elif uav_action == "up":
            next_pos = (x, y + 1, z)
        elif uav_action == "down":
            next_pos = (x, y - 1, z)
        else:
            next_pos = (x, y, z)

        if not (  # do not move out of the grid
            (x <= self.cfg.grid[0][0])
            or (y <= self.cfg.grid[1][0])
            or (x >= self.cfg.grid[0][1])
            or (y >= self.cfg.grid[1][1])
        ):
            self.aRIS.position = next_pos
            self.oob = False
        else:
            self.oob = True
            self.terminate_oob = True

        self.dist[self.time_step % self.mean_win] = get_distance(
            self.aRIS.position, self.U_f.position
        )
        self.pos[self.time_step % self.mean_win] = self.aRIS.position

        # BS actions
        self.BS1.alpha_f = 0.0  # no power allocation / does not matter

        # RIS actions
        self.aRIS.phase_shifts = rescale(action["aris"], -np.pi, np.pi)
        self.tRIS.phase_shifts = rescale(action["tris"], -np.pi, np.pi)

    def _step_ris_no_uav_noma(self, action):
        # Update RIS actions to use cfg.n_elements
        self.aRIS.phase_shifts = rescale(action["aris"][:self.cfg.n_elements], -np.pi, np.pi)
        self.tRIS.phase_shifts = rescale(action["tris"][:self.cfg.n_elements], -np.pi, np.pi)
        self.oob = False
        self.terminate_oob = False

        self.dist[self.time_step % self.mean_win] = get_distance(
            self.aRIS.position[:2], self.U_f.position[:2]
        )
        self.pos[self.time_step % self.mean_win] = self.aRIS.position

        # BS actions
        self.BS1.alpha_f = rescale(action["bs1"], 0.5, 1.0)

        # RIS actions
        self.aRIS.phase_shifts = rescale(action["aris"], -np.pi, np.pi)
        self.tRIS.phase_shifts = rescale(action["tris"], -np.pi, np.pi)

    def _step_uav_noma(self, action):
        # UAV actions
        uav_action = ACTION_MAP[action["uav"]]
        x, y, z = self.aRIS.position

        if uav_action == "left":
            next_pos = (x - 1, y, z)
        elif uav_action == "right":
            next_pos = (x + 1, y, z)
        elif uav_action == "up":
            next_pos = (x, y + 1, z)
        elif uav_action == "down":
            next_pos = (x, y - 1, z)
        else:
            next_pos = (x, y, z)

        if not (  # do not move out of the grid
            (x <= self.cfg.grid[0][0])
            or (y <= self.cfg.grid[1][0])
            or (x >= self.cfg.grid[0][1])
            or (y >= self.cfg.grid[1][1])
        ):
            self.aRIS.position = next_pos
            self.oob = False
        else:
            self.oob = True
            self.terminate_oob = True

        self.dist[self.time_step % self.mean_win] = get_distance(
            self.aRIS.position[:2], self.U_f.position[:2]
        )
        self.pos[self.time_step % self.mean_win] = self.aRIS.position

        # BS actions
        self.BS1.alpha_f = 0.0  # no power allocation / does not matter

    def step(self, action):
        """
        Executes the given action and returns the new state, reward, and
        termination flag.

        Actions from the policy are between -1 and 1. We scale them to the
        appropriate range before executing them.
        """

        # action
        if self.ris_opt and self.uav_opt and self.ma_scheme == "noma":
            self._step_ris_uav_noma(action)
        elif self.ris_opt and self.uav_opt and self.ma_scheme == "oma":
            self._step_ris_uav_oma(action)
        elif self.ris_opt and (not self.uav_opt) and self.ma_scheme == "noma":
            self._step_ris_no_uav_noma(action)
        elif (not self.ris_opt) and self.uav_opt and self.ma_scheme == "noma":
            self._step_uav_noma(action)
        else:
            raise ValueError("Invalid optimization scheme!")

        # update the links
        update_links(self.links, ex_pathloss=False, ex_rvs=True)
        self.channels = get_channels(
            self.cfg,
            self.links,
            self.aRIS,
            self.tRIS,
            self.time_step,
            self.magnitude,
            self.no_ris,
        )

        # calculate the reward
        reward = self._get_reward()

        # update the time step
        self.time_step += 1

        # check trunaction & termination
        truncated = self.time_step >= self.cfg.Ts
        terminated = self.terminate_oob

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def reset(self, seed=None, **kwargs):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        self.seed_fac = self.np_random.integers(0, 2**32 - 1)

        self.BS1.alpha_f = 0.5 + np.finfo(np.float32).eps

        # coefficients
        self.aRIS.amplitudes = np.ones(self.cfg.n_elements)
        self.aRIS.phase_shifts = np.zeros(self.cfg.shape_ris)

        self.tRIS.amplitudes = np.ones(self.cfg.n_elements)
        self.tRIS.phase_shifts = np.zeros(self.cfg.shape_ris)

        self.aRIS.position = self.cfg.init_ris_pos

        # rates
        self.rate_c = 0.0
        self.rate_nc = 0.0
        self.rate_f = 0.0

        self.time_step = 0
        self.oob = False  # out of bounds
        self.terminate_oob = False

        # update the links
        update_links(self.links, ex_pathloss=False, ex_rvs=True, seed_fac=self.seed_fac)
        self.channels = get_channels(
            self.cfg,
            self.links,
            self.aRIS,
            self.tRIS,
            self.time_step,
            self.magnitude,
            self.no_ris,
        )

        return self._get_obs(), self._get_info()