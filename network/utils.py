"""
Utility functions for the environment
"""

import numpy as np
from box import Box
from numba import jit

from comyx.network import RIS, BaseStation, Link, UserEquipment
from comyx.utils import db2pow, generate_seed


def next_position(
    curr_loc, speed, azimuthal_angle, polar_angle=None, mode="horizontal"
):
    """
    Calculates the next position of a UAV in a 2D or 3D plane based on mode.

    Args:
        curr_loc: List representing the current location of the UAV (x, y, z).
        speed: Speed of the UAV, i.e., the radial distance change per unit time.
        azimuthal_angle: Azimuthal angle in radians (-pi to pi).
        polar_angle: Polar angle in radians (-pi/2 to pi/2), optional.
          Required if mode is "full".
        mode: Either "horizontal" (default) for 2D movement or "full" for 3D movement.

    Returns:
        List representing the next position of the UAV (x, y, z).
    """

    if mode == "horizontal":
        x_new = curr_loc[0] + speed * np.cos(azimuthal_angle)
        y_new = curr_loc[1] + speed * np.sin(azimuthal_angle)
        return [x_new, y_new, curr_loc[2]]  # Keep z-coordinate unchanged

    elif mode == "full":
        if polar_angle is None:
            raise ValueError("Polar angle is required for full 3D movement.")

        x0, y0, z0 = curr_loc

        delta_x = speed * np.cos(polar_angle) * np.cos(azimuthal_angle)
        delta_y = speed * np.cos(polar_angle) * np.sin(azimuthal_angle)
        delta_z = speed * np.sin(polar_angle)

        new_x = x0 + delta_x
        new_y = y0 + delta_y
        new_z = z0 + delta_z

        return [new_x, new_y, new_z]

    else:
        raise ValueError("Invalid mode. Choose 'horizontal' or 'full'.")


def init_links(
    cfg: Box,
    BS1: BaseStation,
    BS2: BaseStation,
    BS3: BaseStation,
    aRIS: RIS,
    tRIS: RIS,
    U_c: UserEquipment,
    U_nc: UserEquipment,
    U_f: UserEquipment,
):
    """
    Initializes the links in the network.
    """
    nlos_fading_args = cfg.nlos_fading_args
    los_fading_args = cfg.los_fading_args
    bu_pathloss_args = cfg.bu_pathloss_args
    br_pathloss_args = cfg.br_pathloss_args
    ru_pathloss_args = cfg.ru_pathloss_args
    ici_pathloss_args = cfg.ici_pathloss_args

    # fmt: off
    # center links
    link_1c = Link(
        BS1, U_c,
        nlos_fading_args, bu_pathloss_args,
        shape=cfg.shape_bu, seed=generate_seed("link_1c"),
    )
    link_2nc = Link(
        BS2, U_nc,
        nlos_fading_args, bu_pathloss_args,
        shape=cfg.shape_bu, seed=generate_seed("link_2c"),
    )

    # ici links
    link_2f = Link(
        BS2, U_f,
        nlos_fading_args, ici_pathloss_args,
        shape=cfg.shape_bu, seed=generate_seed("link_2f"),
    )
    link_3c = Link(
        BS3, U_c,
        nlos_fading_args, ici_pathloss_args,
        shape=cfg.shape_bu, seed=generate_seed("link_3c"),
    )
    
    # uav (aRIS) links
    # assuming aerial ris as active ris
    link_1ar = Link(
        BS1, aRIS,
        los_fading_args, br_pathloss_args,
        shape=cfg.shape_br, seed=generate_seed("link_1r"),
        rician_args={"K": db2pow(cfg.K), "order": "post"}
    )
    link_3ar = Link(
        BS3, aRIS,
        los_fading_args, br_pathloss_args,
        shape=cfg.shape_br, seed=generate_seed("link_2r"),
        rician_args={"K": db2pow(cfg.K), "order": "post"}
    )

    link_arc = Link(
        aRIS, U_c,
        los_fading_args, ru_pathloss_args,
        shape=cfg.shape_ru, seed=generate_seed("link_r1c"),
        rician_args={"K": db2pow(cfg.K), "order": "pre"}
    )
    link_arf = Link(
        aRIS, U_f,
        los_fading_args, ru_pathloss_args,
        shape=cfg.shape_ru, seed=generate_seed("link_rf"),
        rician_args={"K": db2pow(cfg.K), "order": "pre"}
    )

    # tRIS links
    link_2tr = Link(
        BS2, tRIS,
        los_fading_args, br_pathloss_args,
        shape=cfg.shape_br, seed=generate_seed("link_2tr"),
        rician_args={"K": db2pow(cfg.K), "order": "post"}
    )

    link_trnc = Link(
        tRIS, U_nc,
        los_fading_args, ru_pathloss_args,
        shape=cfg.shape_ru, seed=generate_seed("link_tr2nc"),
        rician_args={"K": db2pow(cfg.K), "order": "pre"}
    )

    link_trf = Link(
        tRIS, U_f,
        los_fading_args, ru_pathloss_args,
        shape=cfg.shape_ru, seed=generate_seed("link_trf"),
        rician_args={"K": db2pow(cfg.K), "order": "pre"}
    )
    
    links = {
        "link_1c": link_1c, "link_2nc": link_2nc,
        "link_2f": link_2f, "link_3c": link_3c,
        "link_1ar": link_1ar, "link_3ar": link_3ar,
        "link_arc": link_arc, "link_arf": link_arf,
        "link_2tr": link_2tr, "link_trnc": link_trnc,
        "link_trf": link_trf,
    }
    # fmt: on

    return links


class ActiveRIS(RIS):
    def __init__(self, name, n_elements, position):
        super().__init__(name, n_elements, position)
        self.amplification_factors = np.ones(n_elements)

    def apply_phase_and_amplification(self, input_signal):
        # Apply both phase shifts and amplification
        return input_signal * self.amplification_factors * np.exp(1j * self.phase_shifts)

def csc_channel(
    to_ris: Link,
    from_ris: Link,
    amplitudes: np.ndarray,
    phase_shifts: np.ndarray,
    t: int,
    magnitude: bool = False,
):
    """Calculate the effective channel gain."""

    if not magnitude:
        csc = (
            from_ris.channel_gain[..., t].T
            @ np.diag(np.sqrt(amplitudes) * np.exp(1j * phase_shifts))
            @ to_ris.channel_gain[..., t]
        )
    else:
        csc = (
            np.abs(from_ris.channel_gain[..., t].T)
            @ np.diag(np.sqrt(amplitudes))
            @ np.abs(to_ris.channel_gain[..., t])
        )
        return np.abs(csc)

    return csc


def eff_channel(
    direct: Link,
    to_ris: Link,
    from_ris: Link,
    amplitudes: np.ndarray,
    phase_shifts: np.ndarray,
    t: int,
    magnitude: bool = False,
):
    """Calculate the effective channel gain."""

    if not magnitude:
        csc = (
            from_ris.channel_gain[..., t].T
            @ np.diag(np.sqrt(amplitudes) * np.exp(1j * phase_shifts))
            @ to_ris.channel_gain[..., t]
        )
    else:
        csc = (
            np.abs(from_ris.channel_gain[..., t].T)
            @ np.diag(np.sqrt(amplitudes))
            @ np.abs(to_ris.channel_gain[..., t])
        )
        return np.abs(direct.channel_gain[..., t]) + np.abs(csc)

    return direct.channel_gain[..., t] + csc


def update_links(links, ex_pathloss=False, ex_rvs=True, seed_fac=None):
    """
    Updates the channel of the links in the network.
    """
    if seed_fac is None:
        for link in links.values():
            link.update_channel(ex_pathloss=ex_pathloss, ex_rvs=ex_rvs)
    else:
        for link in links.values():
            link.update_channel(ex_pathloss=ex_pathloss, ex_rvs=ex_rvs, seed=seed_fac)
            seed_fac += 1


def get_channels(
    cfg: Box,
    links: dict,
    aRIS: ActiveRIS,
    tRIS: ActiveRIS,
    t: int,
    magnitude: bool = False,
    no_ris=False,
):
    """
    Compute the effective channel gains with active RIS.
    """

    link_1c, link_2nc, link_2f, link_3c = (
        links["link_1c"],
        links["link_2nc"],
        links["link_2f"],
        links["link_3c"],
    )
    link_1ar, link_3ar, link_arc, link_arf = (
        links["link_1ar"],
        links["link_3ar"],
        links["link_arc"],
        links["link_arf"],
    )
    link_2tr, link_trnc, link_trf = (
        links["link_2tr"],
        links["link_trnc"],
        links["link_trf"],
    )

    # interference
    h_2f = link_2f.channel_gain[..., t]
    h_3c = link_3c.channel_gain[..., t]

    # fmt: off
    if not no_ris:
        H_1c = eff_channel(
            link_1c, link_1ar,
            link_arc, aRIS.amplification_factors,
            aRIS.phase_shifts, t, magnitude
        )
        H_2nc = eff_channel(
            link_2nc, link_2tr,
            link_trnc, tRIS.amplification_factors,
            tRIS.phase_shifts, t, magnitude,
        )

        H_1f = csc_channel(
            link_1ar, link_arf, aRIS.amplification_factors,
            aRIS.phase_shifts, t, magnitude
        )
        H_2f = eff_channel(
            link_2f, link_2tr,
            link_trf, tRIS.amplification_factors,
            tRIS.phase_shifts, t, magnitude
        )
        H_3f = csc_channel(
            link_3ar, link_arf, aRIS.amplification_factors,
            aRIS.phase_shifts, t, magnitude
        )
    # fmt: on
    else:
        H_1c = link_1c.channel_gain[..., t]
        H_2nc = link_2nc.channel_gain[..., t]
        H_1f = np.zeros_like(H_1c)
        H_2f = link_2f.channel_gain[..., t]
        H_3f = np.zeros_like(H_1c)

    channels = {
        "H_1c": H_1c,
        "H_2nc": H_2nc,
        "H_1f": H_1f,
        "H_2f": H_2f,
        "H_3f": H_3f,
        "h_2f": h_2f,
        "h_3c": h_3c,
    }

    # Add noise introduced by active RIS
    if not no_ris:
        aRIS_noise = np.random.normal(0, np.sqrt(cfg.sigma2), (cfg.n_elements, 1)) * aRIS.amplification_factors.reshape(-1, 1)
        tRIS_noise = np.random.normal(0, np.sqrt(cfg.sigma2), (cfg.n_elements, 1)) * tRIS.amplification_factors.reshape(-1, 1)
        channels['aRIS_noise'] = aRIS_noise
        channels['tRIS_noise'] = tRIS_noise

    return channels


def get_sinr(
    U_c: UserEquipment,
    U_nc: UserEquipment,
    U_f: UserEquipment,
    BS1: BaseStation,
    BS2: BaseStation,
    BS3: BaseStation,
    channels: dict,
    cfg: Box,
    ma_scheme: str = "noma",
) -> None:
    """
    Compute the SINR of the users in the network.
    """

    H_1c, H_2nc, H_1f, H_2f, H_3f, h_2f, h_3c = (
        channels["H_1c"],
        channels["H_2nc"],
        channels["H_1f"],
        channels["H_2f"],
        channels["H_3f"],
        channels["h_2f"],
        channels["h_3c"],
    )

    rho = cfg.Pt_lin / cfg.sigma2

    if ma_scheme == "noma":
        # center users
        U_c.sinr_pre = (
            (BS1.alpha_f)
            * np.abs(H_1c) ** 2
            / ((1 - BS1.alpha_f) * np.abs(H_1c) ** 2 + np.abs(h_3c) ** 2 + (1 / rho))
        )
        U_c.sinr = (
            (1 - BS1.alpha_f) * np.abs(H_1c) ** 2 / (np.abs(h_3c) ** 2 + (1 / rho))
        )

        U_nc.sinr = np.abs(H_2nc) ** 2 / ((1 / rho))

        # edge users
        U_f.sinr = ((BS1.alpha_f * np.abs(H_1f) ** 2) + (np.abs(H_2f) ** 2)) / (
            (1 - BS1.alpha_f) * np.abs(H_1f) ** 2 + (1 / rho)
        )

    elif ma_scheme == "oma":
        raise NotImplementedError("OMA is not implemented yet.")

    else:
        raise ValueError("Invalid multiple access scheme.")


def get_rates(
    U_c: UserEquipment,
    U_nc: UserEquipment,
    U_f: UserEquipment,
    ma_scheme: str = "noma",
):
    """
    Compute the rates of the users in the network.
    """

    if ma_scheme == "noma":
        rate_c = np.log2(1 + U_c.sinr*1.5)
        rate_nc = np.log2(1 + U_nc.sinr*1.5)
        rate_f = np.log2(1 + U_f.sinr*1.5)
    elif ma_scheme == "oma":
        rate_c = np.log2(1 + U_c.sinr*1.5) / 2
        rate_nc = np.log2(1 + U_nc.sinr*1.5) / 2
        rate_f = np.log2(1 + U_f.sinr*1.5) / 2
    else:
        raise ValueError("Invalid multiple access scheme.")

    return rate_c, rate_nc, rate_f


# JIT compiled as mc can be very large (>> 10000)
@jit(nopython=True)
def get_outage(sinr: np.ndarray, cfg: Box):
    """
    Compute the outage probability of the edge users.
    """
    outage = np.zeros((len(cfg.Pt), 1))
    rate = np.log2(1 + sinr)

    for i in range(len(cfg.Pt)):
        for k in range(cfg.Ts):
            if rate[i, k] < cfg.thresh_edge:
                outage[i] += 1

    return outage / cfg.Ts


@jit(nopython=True)
def get_outage_sic(sinr: np.ndarray, sinr_pre: np.ndarray, cfg: Box):
    """
    Compute the outage probability of the center users.
    """

    outage = np.zeros((len(cfg.Pt), 1))
    rate_pre = np.log2(1 + sinr_pre)
    rate = np.log2(1 + sinr)

    for i in range(len(cfg.Pt)):
        for k in range(cfg.Ts):
            if rate_pre[i, k] < cfg.thresh_edge or rate[i, k] < cfg.thresh_center:
                outage[i] += 1

    return outage / cfg.Ts


def get_outage(U_c: UserEquipment, U_nc: UserEquipment, U_f: UserEquipment, cfg: Box):
    """
    Compute the outage probability of the users in the network.
    """

    outage_c = get_outage_sic(
        np.log2(1 + U_c.sinr),
        np.log2(1 + U_c.sinr_pre),
        cfg.thresh_center,
        cfg.thresh_edge,
    )
    outage_nc = get_outage_sic(
        np.log2(1 + U_nc.sinr),
        np.log2(1 + U_nc.sinr_pre),
        cfg.thresh_center,
        cfg.thresh_edge,
    )
    outage_f = get_outage(np.log2(1 + U_f.sinr), cfg.thresh_edge)

    return outage_c, outage_nc, outage_f


def rescale(x, x_min, x_max):
    """
    Rescale the input in [-1, 1] to [x_min, x_max].

    Args:
        x: Input to be rescaled.
        x_min: Minimum value of the rescaled range.
        x_max: Maximum value of the rescaled range.
    """
    return (x + 1) * (x_max - x_min) / 2 + x_min
