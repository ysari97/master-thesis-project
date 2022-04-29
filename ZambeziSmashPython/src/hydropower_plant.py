# Hydropower plant

from scipy.constants import gravitational_constant

class HydropowerPlant:

    def __init__(self, reservoir, identifier=None, release_share=None):
        
        self.reservoir = reservoir
        self.identifier = identifier
        self.release_share = release_share
        # Read the other parameters from file
        self.efficiency = float()
        self.max_turbine_flow = float()
        self.head_start_level = float()

    def calculate_hydropower_production(self, actual_release, reservoir_level,
        nu_of_days):
        
        if self.release_share != None: actual_release *= self.release_share

        m3_to_kg_factor = 1000
        hours_in_a_day = 24
        W_MW_conversion = 1e-6
        turbine_flow = min(actual_release, self.max_turbine_flow)
        head = reservoir_level - self.head_start_level
        hydropower_production = turbine_flow * head * m3_to_kg_factor * \
            gravitational_constant * self.efficiency * nu_of_days * \
            hours_in_a_day * W_MW_conversion #MWh

        return hydropower_production
