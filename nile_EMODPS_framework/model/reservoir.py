# Reservoir class

from functools import cached_property
import numpy as np
from sympy import memoize_property


class Reservoir:
    """
    A class used to represent reservoirs of the problem

    Attributes
    ----------
    name : str
        Lowercase non-spaced name of the reservoir
    evap_rates : np.array
        (unit)
        Monthly evaporation rates of the reservoir throughout the run
    rating_curve : np.array (...x...)
        (unit) xUnit -> yUnit
        Vectors of water level versus corresponding discharge
    level_to_storage_rel : np.array (2x...)
        (unit) xUnit -> yUnit
        Vectors of water level versus corresponding water storage
    level_to_surface_rel : np.array (2x...)
        (unit) xUnit -> yUnit
        Vectors of water level versus corresponding surface area
    average_cross_section : float
        m2
        Average cross section of the reservoir. Used for approximation
        when relations are not given
    target_hydropower_production : np.array (12x1)
        TWh(?)
        Target hydropower production from the dam
    storage_vector : np.array (1xH)
        m3
        A vector that holds the volume of the water body in the reservoir
        throughout the simulation horizon
    level_vector : np.array (1xH)
        m
        A vector that holds the height of the water body in the reservoir
        throughout the simulation horizon
    release_vector : np.array (1xH)
        m3/s
        A vector that holds the release decisions from the reservoir
        throughout the simulation horizon
    hydropower_plants : list
        A list that holds the hydropower plant objects belonging to the
        reservoir
    actual_hydropower_production : np.array (1xH)
        (unit)
        

    Methods
    -------
    storage_to_level(h=float)
        Returns the level(height) based on volume
    level_to_storage(s=float)
        Returns the volume based on level(height)
    level_to_surface(h=float)
        Returns the surface area based on level
    integration()
        FILL IN LATER!!!!
    """

    def __init__(self, name):
        # Explanation placeholder
        self.name = name

        data_directory = "../NileData/"
        self.evap_rates = np.loadtxt(f"{data_directory}evap_{name}.txt")
        self.rating_curve = np.loadtxt(f"{data_directory}min_max_release_{name}.txt")
        self.level_to_storage_rel = np.loadtxt(f"{data_directory}lsto_rel_{name}.txt")
        self.level_to_surface_rel = np.loadtxt(f"{data_directory}lsur_rel_{name}.txt")
        self.average_cross_section = None  # To be set in the model main file
        self.target_hydropower_production = None  # To be set if obj exists
        self.storage_vector = np.empty(0)
        self.level_vector = np.empty(0)
        self.release_vector = np.empty(0)
        self.hydropower_plants = list()
        self.actual_hydropower_production = np.empty(0)
        self.hydropower_deficit = np.empty(0)
        self.filling_schedule = None
        # Basic memorization implementation
        self.level_to_surface_memo = dict()
        self.storage_to_level_memo = dict()
        self.level_to_minmax_memo = dict()

    def read_hydropower_target(self):
        self.target_hydropower_production = np.loadtxt(
            f"..NileData/{self.name}prod.txt"
        )

    def storage_to_level(self, s):
        rounded_s = s - (s % 1000)
        if rounded_s not in self.storage_to_level_memo:
            self.storage_to_level_memo[rounded_s] = np.interp(
                rounded_s, self.level_to_storage_rel[1], self.level_to_storage_rel[0]
            )

        return self.storage_to_level_memo[rounded_s]

    def level_to_storage(self, h):
        # interpolation when lsto_rel exists
        if self.level_to_storage_rel.size > 0:
            s = np.interp(h, self.level_to_storage_rel[0], self.level_to_storage_rel[1])
        # approximating with volume and cross section
        else:
            s = h * self.average_cross_section
        return s

    def level_to_surface(self, h):
        rounded_h = round(h, 2)
        if rounded_h not in self.level_to_surface_memo:
            self.level_to_surface_memo[rounded_h] = np.interp(
                rounded_h, self.level_to_surface_rel[0], self.level_to_surface_rel[1]
            )

        return self.level_to_surface_memo[rounded_h]

    def level_to_minmax(self, h):
        rounded_h = round(h, 2)
        if rounded_h not in self.level_to_minmax_memo:
            self.level_to_minmax_memo[rounded_h] = (
                np.interp(rounded_h, self.rating_curve[0], self.rating_curve[1]),
                np.interp(rounded_h, self.rating_curve[0], self.rating_curve[2]),
            )

        return self.level_to_minmax_memo[rounded_h]

    def integration(
        self,
        nu_of_days,
        policy_release_decision,
        net_secondly_inflow,
        current_month,
        integration_interval,
    ):
        """Converts the flows of the reservoir into storage. Time step
        fidelity can be adjusted within a for loop. The core idea is to
        arrive at m3 storage from m3/s flows.

        Parameters
        ----------

        Returns
        -------
        """

        total_seconds = 3600 * 24 * nu_of_days

        integration_step_possibilities = {
            "once-a-month": total_seconds,
            "weekly": total_seconds / 4,
            "daily": total_seconds / nu_of_days,
            "12-hours": total_seconds / (nu_of_days * 2),
            "6-hours": total_seconds / (nu_of_days * 4),
            "hourly": total_seconds / (nu_of_days * 24),
            "half-an-hour": total_seconds / (nu_of_days * 48),
        }
        integ_step = integration_step_possibilities[integration_interval]

        current_storage = self.storage_vector[-1]
        in_month_releases = np.empty(0)

        if self.filling_schedule is not None:
            releasable_excess = max(
                0, net_secondly_inflow - self.filling_schedule[current_month - 1]
            )
        else:
            releasable_excess = 1e12  # Big M

        for _ in np.arange(0, total_seconds, integ_step):
            level = self.storage_to_level(current_storage)
            surface = self.level_to_surface(level)

            evaporation = surface * (
                self.evap_rates[current_month - 1]
                / (100 * (total_seconds / integ_step))
            )

            min_possible_release, max_possible_release = self.level_to_minmax(level)

            max_possible_release = min(max_possible_release, releasable_excess)

            secondly_release = min(
                max_possible_release, max(min_possible_release, policy_release_decision)
            )
            in_month_releases = np.append(in_month_releases, secondly_release)

            total_addition = net_secondly_inflow * integ_step

            current_storage += (
                total_addition - evaporation - secondly_release * integ_step
            )

        self.storage_vector = np.append(self.storage_vector, current_storage)

        avg_monthly_release = np.mean(in_month_releases)
        self.release_vector = np.append(self.release_vector, avg_monthly_release)

        # Record level  based on storage for time t:
        self.level_vector = np.append(
            self.level_vector, self.storage_to_level(self.storage_vector[-1])
        )
